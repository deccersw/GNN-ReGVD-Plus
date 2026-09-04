# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Extended with LoRA adapters and FAISS embedding head
# for hybrid vulnerability detection pipeline.

import logging
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
from modelGNN_updates import *
from utils import preprocess_features, preprocess_adj
from utils import *

logger = logging.getLogger(__name__)


def _to_device(array, device):
    """numpy -> float32 torch tensor on `device`.

    The cast to float32 must happen *before* the transfer: the graph helpers
    return float64 (numpy's default) and Apple MPS refuses float64 outright.
    """
    return torch.from_numpy(np.ascontiguousarray(array)).float().to(device)


def pool_positions_to_nodes(hidden, node_ids, n_nodes):
    """Mean-pool contextual token states onto graph nodes.

    Args:
        hidden:   (B, L, D) encoder hidden states -- differentiable
        node_ids: list of B arrays of length L, node index per token position
        n_nodes:  padded node count for the batch (from preprocess_adj)

    Returns:
        (B, n_nodes, D) node features, still attached to the autograd graph.

    In "uni" format several positions share a node (one node per distinct
    token), so their states are averaged; in "text" format the mapping is
    one-to-one and this reduces to a permutation.
    """
    batch, length, dim = hidden.shape
    device = hidden.device

    index = torch.zeros(batch, length, dtype=torch.long, device=device)
    for b, ids in enumerate(node_ids):
        index[b, :len(ids)] = torch.as_tensor(ids, dtype=torch.long,
                                              device=device)

    sums = hidden.new_zeros(batch, n_nodes, dim)
    counts = hidden.new_zeros(batch, n_nodes, 1)
    sums.scatter_add_(1, index.unsqueeze(-1).expand(batch, length, dim), hidden)
    counts.scatter_add_(1, index.unsqueeze(-1),
                        hidden.new_ones(batch, length, 1))
    return sums / counts.clamp(min=1.0)


def check_gradient_flow(model, input_ids, labels, verbose=True, loss_fn=None):
    """Report trainable parameters that receive no gradient.

    Exists because a whole adapter family (LoRA) silently sat outside the
    autograd graph: the parameters were created, counted and handed to the
    optimizer, but no gradient ever reached them, so training was a no-op for
    them. Cheap enough to run once at the start of every training job.
    """
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)

    output = model(input_ids, labels)
    # Use the caller's real objective when given. Checking only the
    # classification loss would flag the FAISS embedding head as detached,
    # since it is trained by the contrastive term instead.
    loss = loss_fn(output, labels) if loss_fn is not None else output[0]
    loss.backward()

    # Two outcomes must be told apart:
    #   grad is None  -> the tensor is outside the autograd graph. Always a bug.
    #   grad all zero -> it is connected but this step gives no signal. For
    #                    lora_A that is expected at initialisation, because
    #                    lora_B starts at zero and grad_A is proportional to B.
    connected, zero, detached = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            detached.append(name)
        elif not torch.any(param.grad != 0):
            zero.append(name)
        else:
            connected.append(name)

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()

    if verbose:
        total = len(connected) + len(zero) + len(detached)
        logger.info("Gradient flow check: %d/%d trainable tensors receive a "
                    "non-zero gradient (%d zero, %d detached)",
                    len(connected), total, len(zero), len(detached))
        if detached:
            logger.error("%d trainable tensor(s) are NOT in the autograd "
                         "graph and can never learn: %s%s",
                         len(detached), ", ".join(detached[:8]),
                         " ..." if len(detached) > 8 else "")
        if zero:
            logger.info("Zero-gradient this step (expected for lora_A while "
                        "lora_B is still zero): %s%s",
                        ", ".join(zero[:5]), " ..." if len(zero) > 5 else "")

    return {"with_grad": connected, "zero_grad": zero,
            "detached": detached,
            "without_grad": zero + detached}  # kept for older callers


#: Hyperparameters that change what a checkpoint *means*. If any of these
#: differ between training and inference, the weights are being run under a
#: different model and the scores are silently wrong.
ARCHITECTURE_KEYS = (
    "encoder_mode", "gnn", "format", "window_size", "block_size",
    "feature_dim_size", "hidden_size", "num_GNN_layers", "att_op",
    "remove_residual", "num_classes", "use_lora", "lora_rank", "lora_alpha",
    "use_faiss", "embed_dim",
)

MODEL_CONFIG_FILENAME = "model_config.json"


def save_model_config(checkpoint_dir, args, model=None):
    """Write the architecture sidecar next to a checkpoint.

    ``encoder_mode`` in particular cannot be recovered from the state dict --
    both modes have exactly the same parameter names and shapes -- so without
    this file a checkpoint trained in one mode loads silently into the other.
    """
    import json

    config = {}
    for key in ARCHITECTURE_KEYS:
        if hasattr(args, key):
            value = getattr(args, key)
            config[key] = value.item() if hasattr(value, "item") else value
    if model is not None and hasattr(model, "encoder_mode"):
        config["encoder_mode"] = model.encoder_mode  # resolved, never "auto"

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, MODEL_CONFIG_FILENAME)
    with open(path, "w") as fh:
        json.dump(config, fh, indent=2)
    logger.info("Saved architecture config to %s", path)
    return path


def load_model_config(checkpoint_path):
    """Read the sidecar for a checkpoint file or directory (None if absent)."""
    import json

    # Anything that is not an existing directory is treated as a file path,
    # so a checkpoint that has not been written yet still resolves correctly.
    directory = checkpoint_path if os.path.isdir(checkpoint_path) \
        else os.path.dirname(checkpoint_path)
    path = os.path.join(directory, MODEL_CONFIG_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:                              # pragma: no cover
        logger.warning("Could not read %s: %s", path, exc)
        return None


def apply_model_config(args, config, override=True):
    """Align ``args`` with a checkpoint's architecture, reporting conflicts."""
    if not config:
        return []

    conflicts = []
    for key, saved in config.items():
        current = getattr(args, key, None)
        if current is None or current == saved:
            continue
        # "auto" is a request to be resolved, not a genuine disagreement
        if key == "encoder_mode" and current == "auto":
            setattr(args, key, saved)
            continue
        conflicts.append((key, current, saved))
        if override:
            setattr(args, key, saved)

    for key, current, saved in conflicts:
        logger.warning(
            "Checkpoint was trained with %s=%r but %r was requested; %s",
            key, saved, current,
            "using the checkpoint's value" if override else "keeping the request",
        )
    return conflicts


SLIM_FORMAT = "slim-v1"


def save_checkpoint_weights(path, model, args=None, slim=True,
                            pretrained_prefix='encoder.roberta.'):
    """Save model weights, optionally storing only what training can change.

    With LoRA the state dict is ~125M parameters of which ~0.8M are trainable;
    the rest is a byte-identical copy of the pretrained encoder that is
    reconstructed from ``model_name_or_path`` on load anyway. Writing it every
    epoch costs ~500 MB per checkpoint, which matters on hosted runtimes where
    checkpoints go to Google Drive.

    Slim payloads are self-describing, so ``load_checkpoint_weights`` accepts
    both formats and old full checkpoints keep working.
    """
    model_to_save = model.module if hasattr(model, 'module') else model

    if not slim:
        torch.save(model_to_save.state_dict(), path)
        return {"format": "full",
                "tensors": len(model_to_save.state_dict())}

    # What may be omitted is exactly what `from_pretrained` restores
    # deterministically -- the contents of the pretrained file, i.e.
    # `encoder.roberta.*`. Everything else must be stored even when frozen:
    # RobertaForSequenceClassification's task head is randomly initialised on
    # every construction (it is reported MISSING when loading the pretrained
    # weights), so omitting it would silently change the model on reload.
    trainable = {}
    for name, param in model_to_save.named_parameters():
        if param.requires_grad or not name.startswith(pretrained_prefix):
            trainable[name] = param.detach().cpu()
    # Buffers are tiny and can change during training, so always keep them.
    buffers = {name: buf.detach().cpu()
               for name, buf in model_to_save.named_buffers()}

    payload = {
        "__format__": SLIM_FORMAT,
        "trainable": trainable,
        "buffers": buffers,
        # Recorded so a slim checkpoint cannot be silently loaded onto a model
        # built from different base weights.
        "base_model": getattr(args, "model_name_or_path", "") if args else "",
        "encoder_mode": getattr(model_to_save, "encoder_mode", ""),
        "pretrained_prefix": pretrained_prefix,
    }
    torch.save(payload, path)
    return {"format": SLIM_FORMAT, "tensors": len(trainable),
            "params": sum(t.numel() for t in trainable.values())}


def load_checkpoint_weights(path, model, device=None, args=None):
    """Load either checkpoint format onto ``model``.

    A slim checkpoint only carries the trainable tensors, so the model must
    already hold the pretrained base weights -- which is what constructing it
    from ``model_name_or_path`` does.
    """
    model_to_load = model.module if hasattr(model, 'module') else model
    obj = torch.load(path, map_location=device or "cpu", weights_only=False)

    if not (isinstance(obj, dict) and obj.get("__format__") == SLIM_FORMAT):
        missing, unexpected = model_to_load.load_state_dict(obj, strict=False)
        if missing or unexpected:
            logger.warning("Checkpoint mismatch: %d missing, %d unexpected "
                           "(e.g. %s / %s)", len(missing), len(unexpected),
                           list(missing)[:2], list(unexpected)[:2])
        logger.info("Loaded full checkpoint %s", path)
        return {"format": "full"}

    expected_base = obj.get("base_model", "")
    actual_base = getattr(args, "model_name_or_path", "") if args else ""
    if expected_base and actual_base and expected_base != actual_base:
        logger.warning(
            "Slim checkpoint was trained on base model %r but this run builds "
            "from %r; the frozen weights differ and results will be wrong.",
            expected_base, actual_base)

    saved = {**obj.get("trainable", {}), **obj.get("buffers", {})}
    own = dict(model_to_load.named_parameters())
    own.update(dict(model_to_load.named_buffers()))
    unknown = [n for n in saved if n not in own]
    if unknown:
        raise ValueError(
            f"Slim checkpoint {path} holds tensors this model does not have: "
            f"{unknown[:5]}. Architecture mismatch -- check model_config.json.")

    model_to_load.load_state_dict(saved, strict=False)
    logger.info("Loaded slim checkpoint %s (%d tensors, %s params; base "
                "weights come from the pretrained model)",
                path, len(obj.get("trainable", {})),
                f"{sum(t.numel() for t in obj.get('trainable', {}).values()):,}")
    return {"format": SLIM_FORMAT, "encoder_mode": obj.get("encoder_mode", "")}


def resolve_device(no_cuda=False, prefer=None):
    """Pick the best available accelerator: CUDA, then Apple MPS, then CPU.

    Without this, `torch.device("cuda" if is_available() else "cpu")` silently
    falls back to CPU on Apple silicon, leaving the GPU unused.
    """
    if prefer:
        return torch.device(prefer)
    if not no_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if not no_cuda and getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_device(tensor_or_module):
    """Infer device from a tensor or module instead of relying on a global variable."""
    if isinstance(tensor_or_module, torch.Tensor):
        return tensor_or_module.device
    if isinstance(tensor_or_module, nn.Module):
        try:
            return next(tensor_or_module.parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


# ─── LoRA layer ──────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation for a nn.Linear layer.
    Wraps an existing Linear and adds trainable low-rank matrices A, B.
    Output = original_linear(x) + x @ A @ B * scaling
    """

    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

        # Low-rank trainable matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out


def apply_lora_to_model(encoder, rank=8, alpha=16, target_modules=None):
    """
    Apply LoRA adapters to GraphCodeBERT/RoBERTa encoder.

    Replaces query and value projection layers in each attention head
    with LoRA-wrapped versions.

    Args:
        encoder: RobertaForSequenceClassification or RobertaModel
        rank: LoRA rank (lower = fewer params, higher = more capacity)
        alpha: LoRA scaling factor
        target_modules: list of substrings to match (default: query and value)

    Returns:
        encoder with LoRA applied, count of LoRA parameters
    """
    if target_modules is None:
        target_modules = ["query", "value"]

    lora_param_count = 0

    # First, freeze ALL encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False

    # Find and replace target linear layers with LoRA versions
    for name, module in encoder.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Navigate to parent module and replace
                parts = name.split(".")
                parent = encoder
                for part in parts[:-1]:
                    parent = getattr(parent, part)

                attr_name = parts[-1]
                original_linear = getattr(parent, attr_name)
                lora_layer = LoRALinear(original_linear, rank=rank, alpha=alpha)
                setattr(parent, attr_name, lora_layer)

                lora_param_count += rank * original_linear.in_features + rank * original_linear.out_features
                break

    return encoder, lora_param_count


# ─── Embedding Head (for FAISS) ─────────────────────────────────────────────

class EmbeddingHead(nn.Module):
    """
    Projects GNN output to a fixed-size L2-normalized embedding for FAISS.
    """

    def __init__(self, input_dim, embed_dim=512, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) — GNN graph-level output
        Returns:
            (batch_size, embed_dim) — L2-normalized embeddings
        """
        projected = self.projector(x.float())
        return F.normalize(projected, p=2, dim=1)


# ─── Classification Head (unchanged from original) ──────────────────────────

class PredictionClassification(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = args.hidden_size
        self.dense = nn.Linear(input_size, args.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(args.hidden_size, args.num_classes)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x.float())
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# ─── Original Model wrapper (unchanged) ─────────────────────────────────────

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = outputs
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


# ─── GNNReGVD with LoRA + Embedding Head ────────────────────────────────────

class GNNReGVD(nn.Module):
    """
    Graph Neural Network for ReGVD — extended with:
      - LoRA adapters on GraphCodeBERT (optional, controlled by args.use_lora)
      - Embedding head for FAISS (optional, controlled by args.use_faiss)

    Forward returns:
      - With labels: (total_loss, cls_prob, embeddings)
      - Without labels: (cls_prob, embeddings)

    Where embeddings is None if use_faiss=False.
    """

    def __init__(self, encoder, config, tokenizer, args):
        super(GNNReGVD, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # LoRA: apply to encoder if enabled
        self.use_lora = getattr(args, 'use_lora', False)
        self.lora_param_count = 0
        if self.use_lora:
            lora_rank = getattr(args, 'lora_rank', 8)
            lora_alpha = getattr(args, 'lora_alpha', 16)
            self.encoder, self.lora_param_count = apply_lora_to_model(
                self.encoder, rank=lora_rank, alpha=lora_alpha
            )

        # ── How node features are produced ──────────────────────────────
        # "static":     features are looked up in a detached copy of the
        #               embedding table. This is the original ReGVD design;
        #               the transformer is NEVER executed, so LoRA adapters
        #               sit outside the autograd graph and cannot train.
        # "contextual": the encoder runs and node features are pooled from
        #               its hidden states, so gradients reach the encoder
        #               (and the LoRA adapters).
        # "auto":       contextual when LoRA is on, static otherwise.
        self.encoder_mode = getattr(args, 'encoder_mode', 'auto')
        if self.encoder_mode == 'auto':
            self.encoder_mode = 'contextual' if self.use_lora else 'static'
        if self.encoder_mode not in ('static', 'contextual'):
            raise ValueError(
                f"encoder_mode must be 'static', 'contextual' or 'auto', "
                f"got {self.encoder_mode!r}"
            )

        if self.use_lora and self.encoder_mode == 'static':
            logger.warning(
                "use_lora=True with encoder_mode='static': the encoder is "
                "never executed in static mode, so the LoRA adapters receive "
                "no gradient and will stay at their initial values "
                "(lora_B is zeroed, so they are an exact no-op). "
                "Use --encoder_mode contextual to actually train them."
            )
        if not self.use_lora and self.encoder_mode == 'contextual':
            logger.warning(
                "encoder_mode='contextual' without LoRA: this is FULL "
                "fine-tuning -- all ~125M encoder weights are trainable and "
                "will be updated. Expect high memory use and a slow epoch. "
                "Add --use_lora to train ~0.3M adapter weights instead."
            )

        # In static mode the encoder is never executed, so every parameter in
        # it is dead weight: it would be handed to the optimizer, counted as
        # "trainable" and never receive a gradient. Freezing it here is what
        # makes the reported parameter counts mean something. (LoRA adapters
        # are frozen too -- the warning above already says they cannot train.)
        if self.encoder_mode == "static":
            frozen = 0
            for param in self.encoder.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen += param.numel()
            if frozen:
                logger.info("encoder_mode='static': froze %s encoder "
                            "parameters that the forward pass never touches",
                            f"{frozen:,}")

        # Contextual mode without LoRA means full fine-tuning of the encoder.
        # --freeze_encoder gives the third option the ablation needs: run the
        # encoder so node features are contextual, but train only the GNN and
        # the heads. Without it there is no arm to compare LoRA against that
        # differs from it in LoRA alone.
        if self.encoder_mode == "contextual" and getattr(args, "freeze_encoder", False):
            frozen = 0
            for name, param in self.encoder.named_parameters():
                if "lora_" in name or not param.requires_grad:
                    continue
                param.requires_grad = False
                frozen += param.numel()
            if frozen:
                logger.info("--freeze_encoder: froze %s encoder parameters; "
                            "the encoder still runs, only %s adapters train",
                            f"{frozen:,}",
                            "LoRA" if self.use_lora else "no")

        # Keep the adapters in the graph but out of the optimizer. This is the
        # control arm for "does adapting LoRA to a new domain help?": dropping
        # --use_lora instead would change the architecture, and a checkpoint
        # that carries LoRA tensors would not even load into it. Here the two
        # arms differ in exactly one thing -- whether the adapters move.
        if self.use_lora and getattr(args, "freeze_lora", False):
            frozen = 0
            for name, param in self.encoder.named_parameters():
                if "lora_" in name and param.requires_grad:
                    param.requires_grad = False
                    frozen += param.numel()
            logger.info("--freeze_lora: %s adapter parameters kept at their "
                        "loaded values", f"{frozen:,}")

        # Weight on the positive term of the classification loss. 1.0 keeps the
        # plain BCE every earlier run used.
        self.pos_weight = float(getattr(args, "pos_weight", 1.0) or 1.0)
        if self.pos_weight != 1.0:
            logger.info("Classification loss uses pos_weight=%.3f", self.pos_weight)

        # Word embeddings from encoder (static feature path only)
        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', 1) \
            if tokenizer is not None else 1
        self.tokenizer = tokenizer

        # GNN
        if args.gnn == "ReGGNN":
            self.gnn = ReGGNN(
                feature_dim_size=args.feature_dim_size,
                hidden_size=args.hidden_size,
                num_GNN_layers=args.num_GNN_layers,
                dropout=config.hidden_dropout_prob,
                residual=not args.remove_residual,
                att_op=args.att_op
            )
        else:
            self.gnn = ReGCN(
                feature_dim_size=args.feature_dim_size,
                hidden_size=args.hidden_size,
                num_GNN_layers=args.num_GNN_layers,
                dropout=config.hidden_dropout_prob,
                residual=not args.remove_residual,
                att_op=args.att_op
            )

        gnn_out_dim = self.gnn.out_dim

        # Classification head (binary: vuln/safe)
        self.classifier = PredictionClassification(config, args, input_size=gnn_out_dim)

        if getattr(args, 'num_classes', 1) > 1:
            logger.warning(
                "num_classes=%d but the head is a single-logit sigmoid "
                "classifier: the loss and all metrics use prob[:, 0], so the "
                "remaining output column(s) receive no gradient. Set "
                "--num_classes 1 for new runs (existing checkpoints keep "
                "their shape).", args.num_classes,
            )

        # FAISS embedding head (optional)
        self.use_faiss = getattr(args, 'use_faiss', False)
        self.embedding_head = None
        if self.use_faiss:
            embed_dim = getattr(args, 'embed_dim', 512)
            self.embedding_head = EmbeddingHead(
                input_dim=gnn_out_dim,
                embed_dim=embed_dim,
                dropout=config.hidden_dropout_prob
            )

    def _build_inputs(self, input_ids):
        """Produce (adj, adj_mask, node_features) for a batch.

        In contextual mode the returned node features carry gradient back to
        the encoder; in static mode they are a constant tensor.
        """
        _dev = input_ids.device
        ids_np = input_ids.cpu().detach().numpy()

        if self.encoder_mode == "static":
            if self.args.format == "uni":
                adj, x_feature = build_graph(
                    ids_np, self.w_embeddings,
                    window_size=self.args.window_size
                )
            else:
                adj, x_feature = build_graph_text(
                    ids_np, self.w_embeddings,
                    window_size=self.args.window_size
                )
            adj, adj_mask = preprocess_adj(adj)
            adj_feature = preprocess_features(x_feature)
            return (_to_device(adj, _dev), _to_device(adj_mask, _dev),
                    _to_device(adj_feature, _dev))

        # ── contextual ──
        if self.args.format == "uni":
            adj, node_ids = build_graph_index(
                ids_np, window_size=self.args.window_size)
        else:
            adj, node_ids = build_graph_text_index(
                ids_np, window_size=self.args.window_size)

        adj, adj_mask = preprocess_adj(adj)
        adj = _to_device(adj, _dev)
        adj_mask = _to_device(adj_mask, _dev)

        # This is the call that was missing: without it the encoder -- and
        # every LoRA adapter inside it -- is not part of the autograd graph.
        hidden = self.encoder.roberta(
            input_ids,
            attention_mask=input_ids.ne(self.pad_token_id),
        )[0]

        adj_feature = pool_positions_to_nodes(hidden, node_ids,
                                              n_nodes=adj.shape[1])
        return adj, adj_mask, adj_feature

    def forward(self, input_ids=None, labels=None):
        adj, adj_mask, adj_feature = self._build_inputs(input_ids)

        # ── GNN forward ──
        gnn_output = self.gnn(adj_feature, adj, adj_mask)

        # ── Classification head ──
        logits = self.classifier(gnn_output)
        prob = torch.sigmoid(logits)

        # ── Embedding head (for FAISS) ──
        embeddings = None
        if self.use_faiss and self.embedding_head is not None:
            embeddings = self.embedding_head(gnn_output)

        # ── Loss computation ──
        if labels is not None:
            labels = labels.float()
            # pos_weight scales the positive term, as in BCEWithLogitsLoss.
            # Devign is ~46% positive, so unweighted BCE is fine there; on a
            # corpus like MegaVul (4.7% positive) it drives the model straight
            # to the all-negative solution, which scores a high accuracy and
            # detects nothing.
            cls_loss = self.pos_weight * torch.log(prob[:, 0] + 1e-10) * labels + \
                       torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            cls_loss = -cls_loss.mean()
            return cls_loss, prob, embeddings
        else:
            return prob, embeddings

    def get_embedding(self, input_ids):
        """
        Get only the FAISS embedding for a batch (no classification).
        Used for building/updating the FAISS index.
        """
        with torch.no_grad():
            adj, adj_mask, adj_feature = self._build_inputs(input_ids)
            gnn_output = self.gnn(adj_feature, adj, adj_mask)

            if self.embedding_head is not None:
                return self.embedding_head(gnn_output)
            else:
                return gnn_output

    def get_trainable_params_info(self):
        """Print info about trainable vs frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        # LoRA params only count as effectively trainable when the encoder is
        # actually executed -- see the encoder_mode note in __init__.
        lora_effective = self.lora_param_count \
            if self.encoder_mode == "contextual" else 0
        effective = trainable - (self.lora_param_count - lora_effective)

        # The head is a single-logit sigmoid classifier: the loss and every
        # evaluation path read prob[:, 0] only. Any further output column is
        # allocated but never receives gradient.
        num_classes = getattr(self.args, "num_classes", 1)
        dead_head = 0
        if num_classes > 1:
            per_class = self.classifier.out_proj.weight.shape[1] + 1
            dead_head = (num_classes - 1) * per_class
            effective -= dead_head

        info = {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": frozen,
            "trainable_percent": round(100 * trainable / max(total, 1), 2),
            "lora_params": self.lora_param_count,
            "lora_params_effective": lora_effective,
            "effective_trainable_params": effective,
            "dead_classifier_params": dead_head,
            "encoder_mode": self.encoder_mode,
        }
        return info


# ─── DevignModel (unchanged from original) ──────────────────────────────────

class DevignModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DevignModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.tokenizer = tokenizer

        self.gnn = GGGNN(
            feature_dim_size=args.feature_dim_size, hidden_size=args.hidden_size,
            num_GNN_layers=args.num_GNN_layers, num_classes=args.num_classes,
            dropout=config.hidden_dropout_prob
        )

        self.conv_l1 = torch.nn.Conv1d(args.hidden_size, args.hidden_size, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(args.hidden_size, args.hidden_size, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = args.feature_dim_size + args.hidden_size
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=args.num_classes)
        self.mlp_y = nn.Linear(in_features=args.hidden_size, out_features=args.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, labels=None):
        _dev = input_ids.device

        if self.args.format == "uni":
            adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings)
        else:
            adj, x_feature = build_graph_text(input_ids.cpu().detach().numpy(), self.w_embeddings)

        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature).to(_dev).float()

        outputs = self.gnn(adj_feature.to(_dev).float(), adj.to(_dev).float(),
                           adj_mask.to(_dev).float())

        c_i = torch.cat((outputs, adj_feature), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(nn.functional.relu(self.conv_l1(outputs.transpose(1, 2))))
        Y_2 = self.maxpool2(nn.functional.relu(self.conv_l2(Y_1))).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(nn.functional.relu(self.conv_l1_for_concat(c_i.transpose(1, 2))))
        Z_2 = self.maxpool2_for_concat(nn.functional.relu(self.conv_l2_for_concat(Z_1))).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        prob = self.sigmoid(avg)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
