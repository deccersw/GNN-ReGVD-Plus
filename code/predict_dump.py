"""Run a checkpoint over a JSONL file and dump one probability per sample.

`run.py --do_test` writes `predictions.txt`, which holds the decision (0/1) and
not the score. That is enough for accuracy and F1 at a fixed threshold and for
nothing else: no PR curve, no threshold chosen on validation, no bootstrap
interval, no paired ranking check. This script keeps the score, plus whatever
metadata the input row carried, so those analyses stay possible after the fact.

Usage:
    python code/predict_dump.py \
        --model_path code/saved_models/baseline/checkpoint-best-acc/model.bin \
        --input_file data/megavul/megavul_test.jsonl \
        --output_file runs/megavul_test.static.jsonl
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (RobertaConfig, RobertaForSequenceClassification,
                          RobertaTokenizer)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (GNNReGVD, load_checkpoint_weights,            # noqa: E402
                   load_model_config, resolve_device)
from run import TextDataset, set_seed                            # noqa: E402

logger = logging.getLogger(__name__)

#: Defaults for a model built from scratch. Every architecture key is
#: overwritten by the checkpoint's own sidecar below, so these only fill gaps.
ARCHITECTURE_DEFAULTS = dict(
    block_size=400, hidden_size=128, feature_dim_size=768, num_GNN_layers=2,
    gnn="ReGCN", format="uni", window_size=5, remove_residual=False,
    att_op="mul", num_classes=1, use_lora=False, lora_rank=8, lora_alpha=16,
    use_faiss=True, embed_dim=512, encoder_mode="auto", freeze_encoder=False,
    pos_weight=1.0, model_name_or_path="microsoft/graphcodebert-base",
)


def build_args(cli):
    args = argparse.Namespace(**ARCHITECTURE_DEFAULTS)
    args.block_size = cli.block_size

    # The sidecar is the authority: running weights under a different
    # architecture silently produces wrong scores rather than an error.
    sidecar = load_model_config(cli.model_path) or {}
    if not sidecar:
        logger.warning("No model_config.json next to %s -- architecture is "
                       "being guessed from defaults", cli.model_path)
    for key, value in sidecar.items():
        setattr(args, key, value)
    logger.info("checkpoint architecture: %s", sidecar or "<unknown>")
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=400)
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base")
    parser.add_argument("--device", default=None,
                        help="cpu / cuda / mps (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=42)
    cli = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                        level=logging.INFO)
    args = build_args(cli)

    device = torch.device(cli.device) if cli.device else resolve_device()
    args.device = device
    set_seed(cli.seed)

    tokenizer = RobertaTokenizer.from_pretrained(cli.tokenizer_name)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 1
    encoder = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config)

    model = GNNReGVD(encoder, config, tokenizer, args)
    load_checkpoint_weights(cli.model_path, model, device, args)
    model.to(device).eval()

    started = time.time()
    dataset = TextDataset(tokenizer, args, cli.input_file)
    logger.info("tokenised %d samples in %.1fs", len(dataset),
                time.time() - started)

    loader = DataLoader(dataset, sampler=SequentialSampler(dataset),
                        batch_size=cli.batch_size)
    probs, labels = [], []
    started = time.time()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            output = model(batch[0].to(device))
            logit = output[0] if isinstance(output, tuple) else output
            probs.append(logit.detach().cpu().numpy())
            labels.append(batch[1].numpy())
            if step and step % 20 == 0:
                done = (step + 1) * cli.batch_size
                logger.info("  %d/%d  %.1f samples/s", done, len(dataset),
                            done / (time.time() - started))
    elapsed = time.time() - started
    probs = np.concatenate(probs, 0)[:, 0]
    labels = np.concatenate(labels, 0)

    os.makedirs(os.path.dirname(os.path.abspath(cli.output_file)), exist_ok=True)
    meta = [json.loads(line) for line in open(cli.input_file)]
    # The untruncated length is worth carrying: it serves as a control
    # predictor -- whether a score tracks nothing more than how long the
    # function is -- and it separates a model that judged the code from one
    # that never saw the relevant lines, since anything past block_size - 2
    # tokens is cut off.
    window = args.block_size - 2
    with open(cli.output_file, "w") as out:
        for row, prob, label in zip(meta, probs, labels):
            code = row.pop("func", "")       # the score is the payload here
            n_tokens = len(tokenizer.tokenize(" ".join(code.split())))
            out.write(json.dumps({**row, "prob": float(prob),
                                  "label": int(label),
                                  "n_tokens": n_tokens,
                                  "truncated": n_tokens > window}) + "\n")

    logger.info("%d samples in %.1fs (%.1f samples/s) -> %s",
                len(dataset), elapsed, len(dataset) / elapsed, cli.output_file)


if __name__ == "__main__":
    main()
