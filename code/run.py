# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Extended with LoRA + FAISS dual-loss training pipeline.
#
# Usage (full pipeline with LoRA + FAISS):
#   python run.py \
#     --output_dir=./saved_models/lora_faiss \
#     --model_type=roberta \
#     --tokenizer_name=microsoft/graphcodebert-base \
#     --model_name_or_path=microsoft/graphcodebert-base \
#     --do_train --do_eval --do_test \
#     --train_data_file=../dataset/train.jsonl \
#     --eval_data_file=../dataset/valid.jsonl \
#     --test_data_file=../dataset/test.jsonl \
#     --block_size 400 --train_batch_size 128 --eval_batch_size 128 \
#     --gnn ReGCN --learning_rate 5e-4 --epoch 100 \
#     --hidden_size 128 --num_GNN_layers 2 \
#     --format uni --window_size 5 \
#     --use_lora --lora_rank 8 --lora_alpha 16 \
#     --use_faiss --embed_dim 512 \
#     --contrastive_weight 0.3 --contrastive_loss supcon

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json


from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import *
from losses import SupervisedContrastiveLoss, TripletMarginLossWithMining
from faiss_index import FAISSIndexManager

cpu_cont = multiprocessing.cpu_count()

# transformers moved these around: AdamW and WEIGHTS_NAME were removed from the
# top-level namespace in recent releases (which is what hosted runtimes such as
# Colab install), so import them defensively rather than pinning an old version.
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          get_linear_schedule_with_warmup)

try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW

try:
    from transformers import WEIGHTS_NAME
except ImportError:
    try:
        from transformers.utils import WEIGHTS_NAME
    except ImportError:
        WEIGHTS_NAME = "pytorch_model.bin"

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


class InputFeatures(object):
    """A single training/test features for an example."""

    def __init__(self, input_tokens, input_ids, idx, label, vuln_type=""):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label
        # Optional CWE/vulnerability class, propagated into the FAISS metadata
        # so that FAISSIndexManager.compute_vuln_type() has something to vote
        # on. Without it every lookup returns ("unknown", 0.0).
        self.vuln_type = vuln_type


def convert_examples_to_features(js, tokenizer, args):
    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    vuln_type = js.get('vuln_type') or js.get('cwe') or js.get('cwe_id') or ""
    return InputFeatures(source_tokens, source_ids, js['idx'], js['target'],
                         vuln_type=str(vuln_type))


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, sample_percent=1.):
        self.examples = []
        with open(file_path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    js = json.loads(line)
                except json.JSONDecodeError as exc:
                    # A raw JSONDecodeError here points at the parser, not at
                    # the data; name the file and line so the real cause is
                    # obvious. dataset/test.jsonl shipped with line 1 missing
                    # its opening brace, which made --do_test die on startup.
                    raise ValueError(
                        f"{file_path}:{line_no} is not valid JSON ({exc}). "
                        f"Line starts with: {line[:60]!r}"
                    ) from exc
                missing = {"func", "target"} - set(js)
                if missing:
                    raise ValueError(
                        f"{file_path}:{line_no} is missing required field(s) "
                        f"{sorted(missing)}; found {sorted(js)}"
                    )
                self.examples.append(convert_examples_to_features(js, tokenizer, args))

        total_len = len(self.examples)
        num_keep = int(sample_percent * total_len)

        if num_keep < total_len:
            np.random.seed(10)
            np.random.shuffle(self.examples)
            self.examples = self.examples[:num_keep]

        if 'train' in file_path:
            logger.info("*** Total Sample ***")
            logger.info("\tTotal: {}\tselected: {}\tpercent: {}\t".format(total_len, num_keep, sample_percent))
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Sample ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

    def get_func_snippet(self, i, max_len=200):
        """Get truncated source code for FAISS metadata."""
        tokens = self.examples[i].input_tokens
        snippet = ' '.join(tokens[:50])  # first 50 tokens
        return snippet[:max_len]


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_training_checkpoint(args, model, optimizer, scheduler, epoch,
                             global_step, best_acc, best_metric=None,
                             epochs_without_improvement=0):
    """Write everything needed to resume training exactly where it stopped."""
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-last')
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, 'module') else model
    save_checkpoint_weights(os.path.join(checkpoint_dir, 'model.bin'),
                            model_to_save, args,
                            slim=(getattr(args, 'checkpoint_format', 'slim') == 'slim'))
    torch.save(optimizer.state_dict(),
               os.path.join(checkpoint_dir, 'optimizer.pt'))
    torch.save(scheduler.state_dict(),
               os.path.join(checkpoint_dir, 'scheduler.pt'))
    save_model_config(checkpoint_dir, args, model_to_save)

    with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as fh:
        json.dump({"epoch": epoch, "global_step": global_step,
                   "best_acc": best_acc, "best_metric": best_metric,
                   "epochs_without_improvement": epochs_without_improvement,
                   "early_stopping_metric": getattr(
                       args, "early_stopping_metric", "eval_loss")}, fh)

    logger.info("Saved resumable checkpoint to %s (epoch %d)",
                checkpoint_dir, epoch)


def load_training_state(output_dir):
    """Read training_state.json from checkpoint-last, or None."""
    path = os.path.join(output_dir, 'checkpoint-last', 'training_state.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def train(args, train_dataset, model, tokenizer):
    """Train the model with dual loss: classification + contrastive (if FAISS enabled)."""
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True)
    # global_step counts *optimizer* steps, so anything compared against it
    # must be expressed in the same unit. Using micro-batch counts here made
    # `global_step % save_steps` fire only every gradient_accumulation_steps
    # epochs, which silently disabled per-epoch evaluation, best-checkpoint
    # saving and early stopping whenever accumulation was on.
    steps_per_epoch = max(1, len(train_dataloader) // args.gradient_accumulation_steps)
    args.max_steps = args.epoch * steps_per_epoch
    args.save_steps = steps_per_epoch
    args.warmup_steps = steps_per_epoch
    args.logging_steps = steps_per_epoch
    args.num_train_epochs = args.epoch
    model.to(args.device)

    # ── Contrastive loss setup ──
    contrastive_loss_fn = None
    if args.use_faiss:
        if args.contrastive_loss == "triplet":
            contrastive_loss_fn = TripletMarginLossWithMining(margin=0.3)
        else:
            contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.07)
        logger.info(f"Using contrastive loss: {args.contrastive_loss}, weight={args.contrastive_weight}")

    # ── Optimizer: only trainable parameters ──
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)

    # ── Print trainable parameters info ──
    if hasattr(model, 'get_trainable_params_info'):
        info = model.get_trainable_params_info()
        logger.info(f"Trainable params: {info['trainable_params']:,} / {info['total_params']:,} "
                     f"({info['trainable_percent']}%)")
        if info.get('encoder_mode'):
            logger.info(f"Encoder mode: {info['encoder_mode']}")
        if info['lora_params'] > 0:
            logger.info(f"LoRA params: {info['lora_params']:,} "
                        f"(effective: {info.get('lora_params_effective', 0):,})")
            if info.get('lora_params_effective', 0) == 0:
                logger.warning(
                    "LoRA parameters are NOT reachable by the optimizer in "
                    "this configuration; they will not change during training."
                )

    # ── Verify every trainable tensor is really in the autograd graph ──
    # Guards against the class of bug where adapters are created and counted
    # but never wired into the forward pass.
    if not getattr(args, 'no_grad_check', False):
        try:
            probe = next(iter(train_dataloader))

            def _training_objective(output, lbl):
                if args.use_faiss and len(output) == 3 and \
                        output[2] is not None and contrastive_loss_fn is not None:
                    return ((1.0 - args.contrastive_weight) * output[0]
                            + args.contrastive_weight
                            * contrastive_loss_fn(output[2], lbl))
                return output[0]

            check_gradient_flow(model.module if hasattr(model, 'module') else model,
                                probe[0].to(args.device), probe[1].to(args.device),
                                loss_fn=_training_objective)
        except Exception as e:
            logger.warning("Gradient flow check skipped: %s", e)
        model.zero_grad(set_to_none=True)

    # Mixed precision. torch.amp is built in; NVIDIA apex is deprecated and
    # is effectively uninstallable on hosted runtimes such as Colab, so it is
    # only used when explicitly requested.
    amp = None
    scaler = None
    if args.fp16:
        if getattr(args, 'use_apex', False):
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "--use_apex requested but apex is not installed. Drop "
                    "--use_apex to use torch.amp instead.")
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level=args.fp16_opt_level)
            logger.info("Mixed precision: apex %s", args.fp16_opt_level)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=(args.device.type == 'cuda'))
            logger.info("Mixed precision: torch.amp (autocast + GradScaler)")

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    model_last = os.path.join(checkpoint_last, 'model.bin')
    if os.path.exists(model_last):
        load_checkpoint_weights(model_last, model, args.device, args)
        logger.info("Resumed model weights from %s", model_last)
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_acc = 0.0
    stop_early = False
    epochs_without_improvement = 0
    best_metric = float('inf') if args.early_stopping_metric == 'eval_loss' else 0.0
    resumed = load_training_state(args.output_dir)
    if resumed:
        best_acc = resumed.get("best_acc", 0.0)
        # Early-stopping state only means something for the metric it was
        # recorded against: restoring an eval_loss best of 0.61 and then
        # comparing eval_auc against it would be nonsense.
        # Absent key means the snapshot predates this field: we cannot know
        # which metric its best value belongs to, so the safe reading is
        # "unknown", not "same as now". Defaulting to the current metric would
        # restore a stale threshold and patience counter and could stop the
        # run on its very first epoch.
        prior_metric = resumed.get("early_stopping_metric")
        if prior_metric == args.early_stopping_metric:
            best_metric = resumed.get("best_metric", best_metric)
            epochs_without_improvement = resumed.get(
                "epochs_without_improvement", 0)
        else:
            logger.warning(
                "Early stopping state does not match metric %r (snapshot: "
                "%s); resetting its best value and patience counter.",
                args.early_stopping_metric, prior_metric or "not recorded")
        logger.info("Resuming at epoch %d (global_step %d, best_acc %.4f)",
                    args.start_epoch, global_step, best_acc)
    model.zero_grad()

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()

            # Forward — returns (cls_loss, prob, embeddings)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(inputs, labels)
            else:
                output = model(inputs, labels)

            if args.use_faiss and len(output) == 3:
                cls_loss, logits, embeddings = output

                # Contrastive loss on embeddings
                if embeddings is not None and contrastive_loss_fn is not None:
                    con_loss = contrastive_loss_fn(embeddings, labels)
                    alpha = 1.0 - args.contrastive_weight
                    loss = alpha * cls_loss + args.contrastive_weight * con_loss
                else:
                    loss = cls_loss
            else:
                # Backward compatible: old model returns (loss, prob)
                loss, logits = output[0], output[1]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            elif args.fp16 and amp is not None:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip once per optimizer step on the fully accumulated
                # gradients, not on every micro-step.
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.fp16 and amp is not None:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / max(global_step - tr_nb, 1)), 4)

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))

                        metric_name = args.early_stopping_metric
                        current = results[metric_name]
                        # An improvement must clear min_delta, otherwise noise
                        # in the fourth decimal keeps resetting the patience
                        # counter and the run never stops: eval_auc going
                        # 0.7139 -> 0.7143 is not progress.
                        delta = args.early_stopping_min_delta
                        improved = (current < best_metric - delta) \
                            if metric_name == 'eval_loss' \
                            else (current > best_metric + delta)

                        if improved:
                            best_metric = current
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 1

                        if results['eval_acc'] > best_acc:
                            best_acc = results['eval_acc']
                            logger.info("  " + "*" * 20)
                            logger.info("  Best acc:%s", round(best_acc, 4))
                            logger.info("  " + "*" * 20)

                            checkpoint_prefix = 'checkpoint-best-acc'
                            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            save_model_config(output_dir, args, model_to_save)
                            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                            info = save_checkpoint_weights(
                                output_dir, model_to_save, args,
                                slim=(args.checkpoint_format == 'slim'))
                            logger.info("Saving model checkpoint to %s (%s, "
                                        "%d tensors)", output_dir,
                                        info['format'], info['tensors'])

                        # Early stopping. Without it a run keeps going long
                        # after validation has turned: on this dataset
                        # eval_loss bottomed out at epoch 10 and rose for the
                        # next 17 epochs while train loss kept falling.
                        if args.early_stopping_patience > 0:
                            if epochs_without_improvement:
                                logger.info(
                                    "  No improvement in %s for %d/%d epochs "
                                    "(best %.4f, current %.4f)",
                                    metric_name, epochs_without_improvement,
                                    args.early_stopping_patience,
                                    best_metric, current)
                            if epochs_without_improvement >= args.early_stopping_patience:
                                logger.info(
                                    "Early stopping: %s has not improved for "
                                    "%d epochs (best %.4f). Best checkpoint is "
                                    "kept in checkpoint-best-acc.",
                                    metric_name,
                                    args.early_stopping_patience, best_metric)
                                stop_early = True

            if stop_early:
                break

        avg_loss = round(train_loss / tr_num, 5)
        logger.info("epoch {} loss {}".format(idx, avg_loss))

        # Resumable snapshot. Hosted runtimes (Colab in particular) disconnect
        # mid-run, and checkpoint-best-acc alone cannot resume: it carries no
        # optimizer, scheduler or epoch state, so restarting from it silently
        # begins a fresh run at epoch 0 with a reset learning-rate schedule.
        if getattr(args, 'save_every_epoch', True) and args.local_rank in [-1, 0]:
            save_training_checkpoint(args, model, optimizer, scheduler,
                                     epoch=idx, global_step=global_step,
                                     best_acc=best_acc,
                                     best_metric=best_metric,
                                     epochs_without_improvement=epochs_without_improvement)

        if stop_early:
            logger.info("Stopped after epoch %d of %d", idx,
                        int(args.num_train_epochs) - 1)
            break

    # ── After training: build FAISS index from best model ──
    if args.use_faiss:
        logger.info("Building FAISS index from training data...")
        # Load best checkpoint — FAISS must match the model that will be used at inference
        best_model_path = os.path.join(args.output_dir, 'checkpoint-best-acc', 'model.bin')
        if os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path} for FAISS index")
            load_checkpoint_weights(best_model_path, model, args.device, args)
        build_faiss_index(args, model, tokenizer, train_dataset)


def build_faiss_index(args, model, tokenizer, dataset):
    """
    Build FAISS index from a dataset using the model's embedding head.
    This is called after training and can also be called standalone.
    """
    model_to_use = model.module if hasattr(model, 'module') else model
    model_to_use.eval()

    embed_dim = getattr(args, 'embed_dim', 512)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset),
                            batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)

    all_embeddings = []
    all_metadata = []
    sample_counter = 0

    for batch in dataloader:
        inputs = batch[0].to(args.device)
        labels = batch[1].numpy()

        with torch.no_grad():
            embeddings = model_to_use.get_embedding(inputs)
            all_embeddings.append(embeddings.cpu().numpy())

        # Build metadata
        for i, label in enumerate(labels):
            all_metadata.append({
                "idx": dataset.examples[sample_counter].idx,
                "label": int(label),
                "func_snippet": dataset.get_func_snippet(sample_counter),
                "vuln_type": getattr(dataset.examples[sample_counter],
                                     "vuln_type", ""),
            })
            sample_counter += 1

    all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # Choose index type based on dataset size
    index_type = "ivf" if len(all_embeddings) > 50000 else "flat"
    manager = FAISSIndexManager(embed_dim=embed_dim, index_type=index_type)
    manager.build_index(all_embeddings, all_metadata)

    faiss_dir = os.path.join(args.output_dir, "faiss_index")
    manager.save(faiss_dir)
    logger.info(f"FAISS index saved to {faiss_dir} ({manager.size} vectors)")


def evaluate(args, model, tokenizer, eval_when_training=False):
    eval_output_dir = args.output_dir
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=args.num_workers, pin_memory=True)

    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            output = model(inputs, label)
            # Handle both old (loss, prob) and new (loss, prob, embeddings) format
            lm_loss = output[0]
            logit = output[1]
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    probs = logits[:, 0]
    preds = logits[:, 0] > 0.5
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    eval_f1 = f1_score(labels, preds, zero_division=0)
    eval_precision = precision_score(labels, preds, zero_division=0)
    eval_recall = recall_score(labels, preds, zero_division=0)
    try:
        eval_auc = roc_auc_score(labels, probs)
    except ValueError:
        eval_auc = 0.0

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "eval_f1": round(eval_f1, 4),
        "eval_precision": round(eval_precision, 4),
        "eval_recall": round(eval_recall, 4),
        "eval_auc": round(eval_auc, 4),
    }
    return result


def test(args, model, tokenizer):
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            output = model(inputs)
            # Handle new format: (prob, embeddings)
            if isinstance(output, tuple):
                logit = output[0]
            else:
                logit = output
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    probs = logits[:, 0]
    preds = probs > 0.5

    # Accuracy alone is not enough to report: on a 46/54 split it hides both
    # a degenerate all-negative classifier and a change in the ranking
    # quality. Mirror the metric set evaluate() already computes.
    test_acc = np.mean(labels == preds)
    try:
        test_auc = roc_auc_score(labels, probs)
    except ValueError:
        test_auc = 0.0
    test_metrics = {
        "test_acc": round(float(test_acc), 4),
        "test_auc": round(float(test_auc), 4),
        "test_f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "test_precision": round(
            float(precision_score(labels, preds, zero_division=0)), 4),
        "test_recall": round(
            float(recall_score(labels, preds, zero_division=0)), 4),
    }
    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(eval_dataset.examples, preds):
            if pred:
                f.write(example.idx + '\t1\n')
            else:
                f.write(example.idx + '\t0\n')

    return test_metrics


def load_best_checkpoint(args, model):
    """Load the best checkpoint, falling back sensibly when it is absent.

    A very short run (or one stopped before the first evaluation) never writes
    checkpoint-best-acc, and the unconditional torch.load used to abort the
    whole job with a bare FileNotFoundError after training had already
    succeeded.
    """
    for name in ('checkpoint-best-acc', 'checkpoint-last'):
        path = os.path.join(args.output_dir, name, 'model.bin')
        if os.path.exists(path):
            load_checkpoint_weights(path, model, args.device, args)
            logger.info("Loaded weights from %s", path)
            return name
    logger.warning(
        "Neither checkpoint-best-acc nor checkpoint-last exists in %s; "
        "evaluating the in-memory weights instead.", args.output_dir)
    return None


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default="../dataset/train.jsonl", type=str)
    parser.add_argument("--output_dir", default="./saved_models", type=str)

    ## Other parameters
    parser.add_argument("--eval_data_file", default="../dataset/valid.jsonl", type=str)
    parser.add_argument("--test_data_file", default="../dataset/test.jsonl", type=str)

    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str)

    parser.add_argument("--mlm", action='store_true')
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--block_size", default=-1, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--evaluate_during_training", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')

    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=1.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)

    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=50)
    parser.add_argument('--save_total_limit', type=int, default=None)
    parser.add_argument("--eval_all_checkpoints", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=42)
    parser.add_argument('--fp16', action='store_true',
                        help="Mixed precision via torch.amp (recommended on T4/Colab)")
    parser.add_argument('--use_apex', action='store_true',
                        help="Use NVIDIA apex instead of torch.amp (deprecated; "
                             "not installable on Colab)")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="DataLoader workers; use 2 on Colab free tier")
    parser.add_argument('--save_every_epoch', action='store_true', default=True,
                        help="Write checkpoint-last after every epoch so an "
                             "interrupted run can resume (essential on Colab)")
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--server_ip', type=str, default='')
    parser.add_argument('--server_port', type=str, default='')

    # Model architecture
    parser.add_argument("--model", default="GNNs", type=str)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--feature_dim_size", default=768, type=int)
    parser.add_argument("--num_GNN_layers", default=2, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--gnn", default="ReGCN", type=str, help="ReGCN or ReGGNN")

    # Graph construction
    parser.add_argument("--format", default="uni", type=str)
    parser.add_argument("--window_size", default=3, type=int)
    parser.add_argument("--remove_residual", default=False, action='store_true')
    parser.add_argument("--att_op", default='mul', type=str)
    parser.add_argument("--training_percent", default=1., type=float)
    parser.add_argument("--alpha_weight", default=1., type=float)

    # ── NEW: LoRA parameters ──
    parser.add_argument("--checkpoint_format", type=str, default="slim",
                        choices=["slim", "full"],
                        help="slim (default): store only trainable tensors -- "
                             "~3 MB instead of ~500 MB with LoRA, since the "
                             "frozen base weights are rebuilt from "
                             "--model_name_or_path. full: the whole state dict.")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Stop after N evaluations with no improvement "
                             "(0 = disabled). The best checkpoint is kept.")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001,
                        help="Smallest change in the watched metric that counts "
                             "as an improvement. Without it, noise in the last "
                             "decimals keeps resetting the patience counter.")
    parser.add_argument("--early_stopping_metric", type=str, default="eval_loss",
                        choices=["eval_loss", "eval_acc", "eval_auc", "eval_f1"],
                        help="Metric early stopping watches (default: eval_loss, "
                             "which turns earliest)")
    parser.add_argument("--encoder_mode", type=str, default="auto",
                        choices=["auto", "static", "contextual"],
                        help="How node features are produced. 'static': look "
                             "them up in a frozen copy of the embedding table "
                             "(original ReGVD; the transformer never runs, so "
                             "LoRA cannot train). 'contextual': run the "
                             "encoder and pool its hidden states, so gradients "
                             "reach the encoder and its LoRA adapters. "
                             "'auto' (default): contextual iff --use_lora.")
    parser.add_argument("--no_grad_check", action='store_true',
                        help="Skip the start-of-training gradient flow check")
    parser.add_argument("--use_lora", action='store_true',
                        help="Enable LoRA adapters on GraphCodeBERT "
                             "(implies --encoder_mode contextual unless "
                             "overridden)")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank (lower=fewer params, default=8)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA scaling factor (default=16)")

    # ── NEW: FAISS parameters ──
    parser.add_argument("--use_faiss", action='store_true',
                        help="Enable FAISS embedding head for nearest-neighbor search")
    parser.add_argument("--embed_dim", type=int, default=512,
                        help="FAISS embedding dimension (default=512)")
    parser.add_argument("--contrastive_weight", type=float, default=0.3,
                        help="Weight for contrastive loss (0.0=only BCE, 1.0=only contrastive)")
    parser.add_argument("--contrastive_loss", type=str, default="supcon",
                        choices=["supcon", "triplet"],
                        help="Type of contrastive loss: supcon or triplet")

    args = parser.parse_args()

    # Setup CUDA
    if args.local_rank == -1 or args.no_cuda:
        device = resolve_device(no_cuda=args.no_cuda, prefer=getattr(args, "device_override", None))
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // max(args.n_gpu, 1)
    args.per_gpu_eval_batch_size = args.eval_batch_size // max(args.n_gpu, 1)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed: %s, fp16: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    set_seed(args.seed)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.start_epoch = 0
    args.start_step = 0
    # Resume from checkpoint-last if one is present. The base encoder is still
    # loaded from model_name_or_path; the fine-tuned weights are restored from
    # the snapshot inside train(), which also restores optimizer and scheduler.
    resumed_state = load_training_state(args.output_dir)
    if resumed_state:
        args.start_epoch = resumed_state.get("epoch", -1) + 1
        args.start_step = resumed_state.get("global_step", 0)
        logger.info("Found checkpoint-last: resuming from epoch %d "
                    "(global_step %d)", args.start_epoch, args.start_step)
        checkpoint_config = load_model_config(
            os.path.join(args.output_dir, 'checkpoint-last'))
        if checkpoint_config:
            apply_model_config(args, checkpoint_config, override=True)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    if args.model_name_or_path:
        encoder = model_class.from_pretrained(args.model_name_or_path,
                                              from_tf=bool('.ckpt' in args.model_name_or_path),
                                              config=config,
                                              cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        encoder = model_class(config)

    # Build model
    if args.model == "devign":
        model = DevignModel(encoder, config, tokenizer, args)
    else:
        model = GNNReGVD(encoder, config, tokenizer, args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
        train_dataset = TextDataset(tokenizer, args, args.train_data_file, args.training_percent)
        if args.local_rank == 0:
            torch.distributed.barrier()
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        load_best_checkpoint(args, model)
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test and args.local_rank in [-1, 0]:
        load_best_checkpoint(args, model)
        model.to(args.device)
        test_result = test(args, model, tokenizer)
        logger.info("***** Test results *****")
        for key in sorted(test_result.keys()):
            logger.info("  %s = %s", key, str(round(test_result[key], 4)))

    return results


if __name__ == "__main__":
    main()
