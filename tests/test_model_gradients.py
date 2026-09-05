"""
Regression tests for gradient flow through GNNReGVD.

These exist because of a silent architectural bug: LoRA adapters were created,
counted, reported in the logs and handed to the optimizer, yet the encoder they
live in was never executed, so no gradient ever reached them and training was a
no-op for every LoRA weight. Nothing crashed and nothing warned.

The tests use a stand-in encoder with the same module layout as
RobertaForSequenceClassification so they run without transformers installed.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

torch = pytest.importorskip("torch")
import torch.nn as nn                                          # noqa: E402

from model import (GNNReGVD, check_gradient_flow,               # noqa: E402
                   pool_positions_to_nodes)
from modelGNN_updates import (build_graph, build_graph_index,    # noqa: E402
                              build_graph_text, build_graph_text_index)

VOCAB, DIM, HID, SEQ = 120, 64, 32, 24


# ----------------------------------------------------------------------
# Stand-in encoder mirroring RobertaForSequenceClassification's layout
# ----------------------------------------------------------------------

class _SelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(DIM, DIM)
        self.key = nn.Linear(DIM, DIM)
        self.value = nn.Linear(DIM, DIM)

    def forward(self, x):
        return self.value(x) + self.query(x) + self.key(x)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = _SelfAttn()

    def forward(self, x):
        return self.self(x)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attention()

    def forward(self, x):
        return self.attention(x)


class _Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(VOCAB, DIM)

    def forward(self, ids):
        return self.word_embeddings(ids)


class _Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([_Layer() for _ in range(2)])

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class _Roberta(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = _Embeddings()
        self.encoder = _Enc()

    def forward(self, ids, attention_mask=None):
        return (self.encoder(self.embeddings(ids)),)


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = _Roberta()

    def forward(self, ids, attention_mask=None):
        return self.roberta(ids, attention_mask)


class FakeConfig:
    hidden_dropout_prob = 0.1


def make_args(**overrides):
    base = dict(
        gnn="ReGCN", format="uni", window_size=5,
        feature_dim_size=DIM, hidden_size=HID, num_GNN_layers=2,
        remove_residual=False, att_op="mul", num_classes=1,
        use_lora=True, lora_rank=4, lora_alpha=8,
        use_faiss=True, embed_dim=16, encoder_mode="auto",
    )
    base.update(overrides)
    return type("Args", (), base)()


def make_model(**overrides):
    torch.manual_seed(0)
    return GNNReGVD(FakeEncoder(), FakeConfig(), None, make_args(**overrides))


def batch(size=4):
    torch.manual_seed(1)
    ids = torch.randint(5, VOCAB, (size, SEQ))
    labels = torch.tensor([1.0, 0.0] * (size // 2))
    return ids, labels


def lora_names(model):
    return [n for n, _ in model.named_parameters() if "lora_" in n]


# ======================================================================
# The bug itself
# ======================================================================

class TestLoRAGradientFlow:
    def test_static_mode_freezes_the_whole_encoder(self):
        """The encoder is never executed in static mode, so nothing in it may
        be reported as trainable -- that is what hid the original LoRA bug."""
        model = make_model(encoder_mode="static")
        live = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
        assert live == [], f"static mode left {len(live)} encoder params trainable"
        assert lora_names(model), "sanity: the model does have LoRA tensors"

    def test_static_mode_never_trains_lora(self):
        model = make_model(encoder_mode="static")
        ids, labels = batch()
        report = check_gradient_flow(model, ids, labels, verbose=False)
        assert not any("lora_" in n for n in
                       report["with_grad"] + report["zero_grad"])

    def test_contextual_mode_connects_lora(self):
        model = make_model(encoder_mode="contextual")
        ids, labels = batch()
        report = check_gradient_flow(model, ids, labels, verbose=False)

        assert [n for n in report["detached"] if "lora_" in n] == []

    def test_lora_b_gets_gradient_immediately(self):
        """lora_B is the one that can move on step 1; lora_A cannot (B is zero)."""
        model = make_model(encoder_mode="contextual")
        ids, labels = batch()
        report = check_gradient_flow(model, ids, labels, verbose=False)

        assert all("lora_B" in n or "lora_" not in n
                   for n in report["with_grad"] if "lora_" in n)
        assert all("lora_A" in n for n in report["zero_grad"] if "lora_" in n)

    def test_lora_weights_actually_move_during_training(self):
        """The end-to-end property that matters: do the weights change?"""
        model = make_model(encoder_mode="contextual")
        ids, labels = batch()
        before = {n: p.detach().clone()
                  for n, p in model.named_parameters() if "lora_" in n}

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2)
        for _ in range(3):
            opt.zero_grad()
            model(ids, labels)[0].backward()
            opt.step()

        moved = [n for n, p in model.named_parameters()
                 if "lora_" in n and not torch.equal(p, before[n])]
        assert len(moved) == len(before) > 0

    def test_lora_weights_frozen_in_static_mode(self):
        model = make_model(encoder_mode="static")
        ids, labels = batch()
        before = {n: p.detach().clone()
                  for n, p in model.named_parameters() if "lora_" in n}

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2)
        for _ in range(3):
            opt.zero_grad()
            model(ids, labels)[0].backward()
            opt.step()

        moved = [n for n, p in model.named_parameters()
                 if "lora_" in n and not torch.equal(p, before[n])]
        assert moved == []

    def test_encoder_is_invoked_only_in_contextual_mode(self):
        for mode, expected in (("static", False), ("contextual", True)):
            model = make_model(encoder_mode=mode)
            seen = {"called": False}
            original = model.encoder.roberta.forward

            def spy(*a, _orig=original, **kw):
                seen["called"] = True
                return _orig(*a, **kw)

            model.encoder.roberta.forward = spy
            model(*batch())
            assert seen["called"] is expected, f"mode={mode}"


class TestEncoderModeResolution:
    def test_auto_follows_use_lora(self):
        assert make_model(use_lora=True).encoder_mode == "contextual"
        assert make_model(use_lora=False).encoder_mode == "static"

    def test_explicit_mode_wins(self):
        assert make_model(use_lora=True,
                          encoder_mode="static").encoder_mode == "static"

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="encoder_mode"):
            make_model(encoder_mode="nonsense")

    def test_reported_effective_params_exclude_dead_lora(self):
        static = make_model(encoder_mode="static").get_trainable_params_info()
        ctx = make_model(encoder_mode="contextual").get_trainable_params_info()

        assert static["lora_params"] == ctx["lora_params"] > 0
        assert static["lora_params_effective"] == 0
        assert ctx["lora_params_effective"] == ctx["lora_params"]
        assert static["effective_trainable_params"] < \
               ctx["effective_trainable_params"]


# ======================================================================
# The two feature paths must describe the same graph
# ======================================================================

class TestGraphEquivalence:
    def test_uni_adjacency_identical(self):
        ids = torch.randint(5, VOCAB, (3, SEQ)).numpy()
        embeddings = torch.randn(VOCAB, DIM).numpy()

        adj_static, _ = build_graph(ids, embeddings, window_size=5)
        adj_ctx, node_ids = build_graph_index(ids, window_size=5)

        assert len(adj_static) == len(adj_ctx)
        for a, b in zip(adj_static, adj_ctx):
            assert a.shape == b.shape
            assert (a != b).nnz == 0

    def test_text_adjacency_identical(self):
        ids = torch.randint(5, VOCAB, (3, SEQ)).numpy()
        embeddings = torch.randn(VOCAB, DIM).numpy()

        adj_static, _ = build_graph_text(ids, embeddings, window_size=5)
        adj_ctx, _ = build_graph_text_index(ids, window_size=5)

        for a, b in zip(adj_static, adj_ctx):
            assert (a != b).nnz == 0

    def test_node_ids_map_every_position(self):
        ids = torch.randint(5, VOCAB, (2, SEQ)).numpy()
        adj, node_ids = build_graph_index(ids, window_size=5)
        for i, mapping in enumerate(node_ids):
            assert len(mapping) == SEQ
            assert mapping.max() == adj[i].shape[0] - 1

    def test_same_token_shares_a_node(self):
        ids = torch.tensor([[7, 9, 7, 9, 7, 9]]).numpy()
        adj, node_ids = build_graph_index(ids, window_size=3)
        assert adj[0].shape == (2, 2)
        assert node_ids[0][0] == node_ids[0][2] == node_ids[0][4]


class TestPooling:
    def test_averages_positions_of_the_same_node(self):
        hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]])
        node_ids = [torch.tensor([0, 0, 1]).numpy()]
        out = pool_positions_to_nodes(hidden, node_ids, n_nodes=2)

        assert torch.allclose(out[0, 0], torch.tensor([2.0, 2.0]))
        assert torch.allclose(out[0, 1], torch.tensor([5.0, 5.0]))

    def test_padding_nodes_are_zero(self):
        hidden = torch.randn(1, 3, 4)
        out = pool_positions_to_nodes(hidden, [torch.tensor([0, 0, 1]).numpy()],
                                      n_nodes=5)
        assert torch.count_nonzero(out[0, 2:]) == 0

    def test_gradient_reaches_the_input(self):
        hidden = torch.randn(2, 6, 4, requires_grad=True)
        node_ids = [torch.tensor([0, 1, 0, 1, 2, 2]).numpy()] * 2
        pool_positions_to_nodes(hidden, node_ids, n_nodes=3).sum().backward()
        assert hidden.grad is not None and torch.any(hidden.grad != 0)


class TestDeadClassifierColumn:
    def test_extra_output_column_gets_no_gradient(self):
        """num_classes>1 allocates weights the loss can never reach."""
        model = make_model(num_classes=2, encoder_mode="contextual")
        ids, labels = batch()
        model.zero_grad()
        model(ids, labels)[0].backward()

        grad = model.classifier.out_proj.weight.grad
        assert torch.any(grad[0] != 0)
        assert torch.all(grad[1] == 0)

    def test_dead_params_are_reported(self):
        info = make_model(num_classes=2).get_trainable_params_info()
        assert info["dead_classifier_params"] == HID + 1

    def test_single_class_head_has_no_dead_params(self):
        info = make_model(num_classes=1).get_trainable_params_info()
        assert info["dead_classifier_params"] == 0


class TestCheckpointConfigSidecar:
    def test_roundtrip(self, tmp_path):
        from model import load_model_config, save_model_config

        model = make_model(encoder_mode="contextual")
        save_model_config(str(tmp_path), make_args(encoder_mode="auto"), model)

        config = load_model_config(str(tmp_path / "model.bin"))
        assert config is not None
        # "auto" must be stored resolved, otherwise it tells us nothing
        assert config["encoder_mode"] == "contextual"
        assert config["hidden_size"] == HID

    def test_missing_sidecar_returns_none(self, tmp_path):
        from model import load_model_config
        assert load_model_config(str(tmp_path / "model.bin")) is None

    def test_conflicting_request_is_overridden(self, tmp_path):
        from model import apply_model_config, load_model_config, save_model_config

        save_model_config(str(tmp_path), make_args(),
                          make_model(encoder_mode="contextual"))
        config = load_model_config(str(tmp_path))

        args = make_args(encoder_mode="static")
        conflicts = apply_model_config(args, config, override=True)

        assert ("encoder_mode", "static", "contextual") in conflicts
        assert args.encoder_mode == "contextual"

    def test_auto_is_resolved_without_being_a_conflict(self, tmp_path):
        from model import apply_model_config, load_model_config, save_model_config

        save_model_config(str(tmp_path), make_args(),
                          make_model(encoder_mode="contextual"))
        config = load_model_config(str(tmp_path))

        args = make_args(encoder_mode="auto")
        assert apply_model_config(args, config) == []
        assert args.encoder_mode == "contextual"


class TestForwardContract:
    @pytest.mark.parametrize("mode", ["static", "contextual"])
    @pytest.mark.parametrize("fmt", ["uni", "text"])
    def test_shapes_and_finiteness(self, mode, fmt):
        model = make_model(encoder_mode=mode, format=fmt)
        ids, labels = batch()

        loss, prob, emb = model(ids, labels)
        assert loss.shape == () and torch.isfinite(loss)
        assert prob.shape == (4, 1)
        assert emb.shape == (4, 16)
        assert torch.allclose(emb.norm(dim=1), torch.ones(4), atol=1e-5)

    @pytest.mark.parametrize("mode", ["static", "contextual"])
    def test_get_embedding_matches_forward(self, mode):
        model = make_model(encoder_mode=mode)
        model.eval()
        ids, _ = batch()
        with torch.no_grad():
            _, emb_forward = model(ids)
            emb_direct = model.get_embedding(ids)
        assert torch.allclose(emb_forward, emb_direct, atol=1e-5)

    def test_get_embedding_does_not_leak_gradients(self):
        model = make_model(encoder_mode="contextual")
        emb = model.get_embedding(batch()[0])
        assert not emb.requires_grad


# ======================================================================
# Device selection (Apple silicon was silently falling back to CPU)
# ======================================================================

class TestDeviceSelection:
    def test_resolve_device_prefers_accelerator(self):
        from model import resolve_device
        device = resolve_device()
        has_accel = torch.cuda.is_available() or (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available())
        assert (device.type != "cpu") == has_accel

    def test_no_cuda_forces_cpu(self):
        from model import resolve_device
        assert resolve_device(no_cuda=True).type == "cpu"

    def test_explicit_override_wins(self):
        from model import resolve_device
        assert resolve_device(prefer="cpu").type == "cpu"

    def test_graph_tensors_are_float32(self):
        """float64 reaches the device otherwise, and MPS rejects it outright."""
        model = make_model(encoder_mode="static")
        adj, mask, features = model._build_inputs(batch()[0])
        assert adj.dtype == torch.float32
        assert mask.dtype == torch.float32
        assert features.dtype == torch.float32


@pytest.mark.skipif(
    not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    reason="MPS not available")
class TestMPS:
    @pytest.mark.parametrize("mode", ["static", "contextual"])
    def test_forward_backward_runs_on_mps(self, mode):
        model = make_model(encoder_mode=mode).to("mps")
        ids, labels = batch()
        loss = model(ids.to("mps"), labels.to("mps"))[0]
        loss.backward()
        assert torch.isfinite(loss)

    def test_lora_receives_gradient_on_mps(self):
        model = make_model(encoder_mode="contextual").to("mps")
        ids, labels = batch()
        report = check_gradient_flow(model, ids.to("mps"), labels.to("mps"),
                                     verbose=False)
        assert [n for n in report["detached"] if "lora_" in n] == []


class TestGradientCheckObjective:
    def test_custom_loss_covers_the_embedding_head(self):
        """With only the classification loss the FAISS head looks detached."""
        from losses import SupervisedContrastiveLoss
        contrastive = SupervisedContrastiveLoss(0.07)
        model = make_model(encoder_mode="contextual")
        ids, labels = batch()

        cls_only = check_gradient_flow(model, ids, labels, verbose=False)
        combined = check_gradient_flow(
            model, ids, labels, verbose=False,
            loss_fn=lambda out, lab: 0.7 * out[0] + 0.3 * contrastive(out[2], lab))

        assert any("embedding_head" in n for n in cls_only["detached"])
        assert not any("embedding_head" in n for n in combined["detached"])


class TestParameterFreezing:
    def test_static_mode_reports_only_real_trainables(self):
        info = make_model(encoder_mode="static").get_trainable_params_info()
        # GNN + heads only; the 125M encoder must not be counted
        assert info["trainable_percent"] < 50.0
        assert info["lora_params_effective"] == 0

    def test_contextual_mode_keeps_lora_trainable(self):
        model = make_model(encoder_mode="contextual")
        live = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
        assert live and all("lora_" in n for n in live), \
            "only the LoRA tensors should be trainable inside the encoder"

    def test_no_dead_parameters_reach_the_optimizer(self):
        """Every parameter we hand to AdamW must be able to receive gradient."""
        from losses import SupervisedContrastiveLoss
        contrastive = SupervisedContrastiveLoss(0.07)
        for mode in ("static", "contextual"):
            model = make_model(encoder_mode=mode)
            ids, labels = batch()
            report = check_gradient_flow(
                model, ids, labels, verbose=False,
                loss_fn=lambda o, l: 0.7 * o[0] + 0.3 * contrastive(o[2], l))
            assert report["detached"] == [], \
                f"mode={mode}: dead params in optimizer: {report['detached'][:5]}"


# ======================================================================
# Slim checkpoints: store only what training can change
# ======================================================================

class TestSlimCheckpoints:
    def _trained(self, **kw):
        model = make_model(**kw)
        ids, labels = batch()
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-1)
        for _ in range(3):
            opt.zero_grad(); model(ids, labels)[0].backward(); opt.step()
        return model

    def test_slim_holds_trainable_plus_unreproducible_tensors(self, tmp_path):
        from model import save_checkpoint_weights
        model = make_model(encoder_mode="contextual")
        path = tmp_path / "m.bin"
        save_checkpoint_weights(str(path), model, slim=True)

        payload = torch.load(str(path), weights_only=False)
        saved = set(payload["trainable"])
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert trainable <= saved
        # frozen weights under the pretrained prefix are the only omissions
        omitted = {n for n, _ in model.named_parameters()} - saved
        assert omitted and all(n.startswith("encoder.roberta.") for n in omitted)

    def test_slim_roundtrip_is_lossless_for_every_tensor(self, tmp_path):
        """Frozen-but-randomly-initialised weights must survive too."""
        from model import load_checkpoint_weights, save_checkpoint_weights
        trained = self._trained(encoder_mode="contextual")
        path = tmp_path / "m.bin"
        save_checkpoint_weights(str(path), trained, slim=True)

        fresh = make_model(encoder_mode="contextual")
        # perturb a frozen, non-pretrained tensor so a silent omission shows up
        with torch.no_grad():
            for name, p in fresh.named_parameters():
                if not p.requires_grad and not name.startswith("encoder.roberta."):
                    p.add_(1.0)

        load_checkpoint_weights(str(path), fresh)
        ref = dict(trained.named_parameters())
        differing = [n for n, p in fresh.named_parameters()
                     if not torch.equal(p, ref[n])]
        assert differing == [], f"slim lost {len(differing)} tensors: {differing[:5]}"

    def test_slim_is_much_smaller_than_full(self, tmp_path):
        from model import save_checkpoint_weights
        model = make_model(encoder_mode="contextual")
        slim, full = tmp_path / "s.bin", tmp_path / "f.bin"
        save_checkpoint_weights(str(slim), model, slim=True)
        save_checkpoint_weights(str(full), model, slim=False)
        assert slim.stat().st_size < full.stat().st_size / 2

    def test_slim_roundtrip_restores_trained_weights(self, tmp_path):
        from model import load_checkpoint_weights, save_checkpoint_weights
        trained = self._trained(encoder_mode="contextual")
        path = tmp_path / "m.bin"
        save_checkpoint_weights(str(path), trained, slim=True)

        fresh = make_model(encoder_mode="contextual")
        differing = [n for n, p in fresh.named_parameters()
                     if p.requires_grad and not torch.equal(
                         p, dict(trained.named_parameters())[n])]
        assert differing, "sanity: training must have changed something"

        load_checkpoint_weights(str(path), fresh)
        for name, p in fresh.named_parameters():
            assert torch.equal(p, dict(trained.named_parameters())[name]), name

    def test_full_checkpoints_still_load(self, tmp_path):
        """Old checkpoints predate the slim format and must keep working."""
        from model import load_checkpoint_weights
        trained = self._trained(encoder_mode="contextual")
        path = tmp_path / "legacy.bin"
        torch.save(trained.state_dict(), str(path))   # the old way

        fresh = make_model(encoder_mode="contextual")
        assert load_checkpoint_weights(str(path), fresh)["format"] == "full"
        for name, p in fresh.named_parameters():
            assert torch.equal(p, dict(trained.named_parameters())[name]), name

    def test_slim_records_and_reports_its_format(self, tmp_path):
        from model import SLIM_FORMAT, load_checkpoint_weights, save_checkpoint_weights
        model = make_model(encoder_mode="contextual")
        path = tmp_path / "m.bin"
        save_checkpoint_weights(str(path), model, slim=True)
        assert load_checkpoint_weights(str(path),
                                       make_model(encoder_mode="contextual")
                                       )["format"] == SLIM_FORMAT

    def test_architecture_mismatch_is_rejected(self, tmp_path):
        """A slim payload naming unknown tensors must fail loudly, not silently."""
        from model import SLIM_FORMAT, load_checkpoint_weights
        path = tmp_path / "bogus.bin"
        torch.save({"__format__": SLIM_FORMAT,
                    "trainable": {"no.such.param": torch.zeros(3)},
                    "buffers": {}}, str(path))
        with pytest.raises(ValueError, match="does not have"):
            load_checkpoint_weights(str(path), make_model())

    def test_static_mode_slim_excludes_the_frozen_encoder(self, tmp_path):
        """None of the pretrained bulk is written.

        LoRA tensors are the exception and are kept: they live under the
        encoder prefix but are not in the pretrained file, so nothing would
        restore them on load.
        """
        from model import save_checkpoint_weights
        model = make_model(encoder_mode="static")
        path = tmp_path / "m.bin"
        save_checkpoint_weights(str(path), model, slim=True)
        payload = torch.load(str(path), weights_only=False)
        assert not any(n.startswith("encoder.") and "lora_" not in n
                       for n in payload["trainable"])


class TestNoRawCheckpointLoads:
    """Every model checkpoint must go through load_checkpoint_weights().

    Slim payloads are dicts with a __format__ key, not state dicts, so a raw
    `load_state_dict(torch.load(path))` blows up on them. Four such call sites
    survived the switch to the slim format and only surfaced when a real Colab
    run crashed after training, while building the FAISS index.
    """

    def test_no_module_loads_a_model_checkpoint_directly(self):
        import re
        root = os.path.join(os.path.dirname(__file__), "..")
        offenders = []
        for rel in ("code/run.py", "code/run_finetune.py", "code/inference.py",
                    "scanner/pipeline.py"):
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                continue
            for i, line in enumerate(open(path), 1):
                if re.search(r"load_state_dict\(\s*torch\.load", line):
                    # optimizer/scheduler state is a plain dict and is fine
                    if re.search(r"\b(optimizer|scheduler)\.load_state_dict", line):
                        continue
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        assert offenders == [], (
            "these load a checkpoint without going through "
            "load_checkpoint_weights(), so slim checkpoints will crash:\n"
            + "\n".join(offenders))


# ----------------------------------------------------------------------
# pos_weight on the classification loss
# ----------------------------------------------------------------------

class TestPosWeight:
    """Unweighted BCE collapses to all-negative on an imbalanced corpus.

    MegaVul is 4.7% positive: there, predicting "safe" for everything scores
    0.953 accuracy, and plain BCE walks straight to it. pos_weight is what
    makes the positive class cost enough to matter.
    """

    def test_default_is_plain_bce(self):
        model = make_model()
        assert model.pos_weight == 1.0

    def test_absent_attribute_defaults_to_one(self):
        args = make_args()
        assert not hasattr(args, "pos_weight")
        model = GNNReGVD(FakeEncoder(), FakeConfig(), None, args)
        assert model.pos_weight == 1.0

    def test_weight_reaches_the_loss(self):
        ids, labels = batch()
        base = make_model(pos_weight=1.0)
        loss_plain = base(ids, labels)[0]

        weighted = make_model(pos_weight=10.0)
        loss_weighted = weighted(ids, labels)[0]

        # Same weights, same batch: only the positive term is rescaled, and
        # the batch has positives, so the loss has to move.
        assert not torch.isclose(loss_plain, loss_weighted)

    def test_weight_only_touches_the_positive_term(self):
        ids = torch.randint(5, VOCAB, (4, SEQ))
        all_negative = torch.zeros(4)
        plain = make_model(pos_weight=1.0)(ids, all_negative)[0]
        heavy = make_model(pos_weight=50.0)(ids, all_negative)[0]
        # With no positives in the batch the weight has nothing to scale.
        assert torch.isclose(plain, heavy)

    def test_gradient_still_flows_with_a_weight(self):
        ids, labels = batch()
        model = make_model(pos_weight=8.0, encoder_mode="contextual")
        report = check_gradient_flow(model, ids, labels, verbose=False)
        # The embedding head is trained by the contrastive term, so the
        # classification loss alone legitimately leaves it out.
        assert [n for n in report["detached"]
                if "embedding_head" not in n] == []


# ----------------------------------------------------------------------
# --freeze_encoder
# ----------------------------------------------------------------------

class TestFreezeEncoder:
    """The ablation arm that isolates LoRA.

    Comparing "contextual + LoRA" against "static" conflates LoRA with the
    change of node features. The only comparison that attributes a gain to
    LoRA alone is against a contextual run whose encoder is frozen.
    """

    @staticmethod
    def _trainable(model, predicate):
        return sum(p.numel() for n, p in model.encoder.named_parameters()
                   if p.requires_grad and predicate(n))

    def test_freezes_the_encoder_but_keeps_lora_trainable(self):
        model = make_model(encoder_mode="contextual", use_lora=True,
                           freeze_encoder=True)
        assert self._trainable(model, lambda n: "lora_" not in n) == 0
        assert self._trainable(model, lambda n: "lora_" in n) > 0

    def test_without_lora_nothing_in_the_encoder_trains(self):
        model = make_model(encoder_mode="contextual", use_lora=False,
                           freeze_encoder=True)
        assert self._trainable(model, lambda n: True) == 0
        # The GNN and the heads still have to train, or the arm is useless.
        assert sum(p.numel() for p in model.gnn.parameters()
                   if p.requires_grad) > 0

    def test_off_by_default_contextual_without_lora_is_a_full_finetune(self):
        model = make_model(encoder_mode="contextual", use_lora=False)
        assert self._trainable(model, lambda n: True) > 0

    def test_encoder_still_runs_when_frozen(self):
        ids, labels = batch()
        model = make_model(encoder_mode="contextual", use_lora=True,
                           freeze_encoder=True)
        assert model.encoder_mode == "contextual"
        report = check_gradient_flow(model, ids, labels, verbose=False)
        assert [n for n in report["detached"]
                if "embedding_head" not in n] == []
        # Frozen weights are excluded from the report, not silently detached.
        assert any("lora_" in n for n in
                   report["with_grad"] + report["zero_grad"])

    def test_is_a_no_op_in_static_mode(self):
        frozen = make_model(encoder_mode="static", use_lora=False,
                            freeze_encoder=True)
        plain = make_model(encoder_mode="static", use_lora=False)
        assert self._trainable(frozen, lambda n: True) == \
               self._trainable(plain, lambda n: True) == 0


# ----------------------------------------------------------------------
# --freeze_lora
# ----------------------------------------------------------------------

class TestFreezeLora:
    """The control arm for "what does adapting LoRA to a new domain buy?".

    Dropping --use_lora is not that control: it changes the architecture, and a
    checkpoint carrying lora_A/lora_B tensors cannot be loaded into a model
    without them at all. Freezing the adapters keeps everything identical
    except whether they move.
    """

    @staticmethod
    def _lora_params(model):
        return [(n, p) for n, p in model.encoder.named_parameters()
                if "lora_" in n]

    def test_off_by_default(self):
        model = make_model(encoder_mode="contextual")
        assert any(p.requires_grad for _, p in self._lora_params(model))

    def test_freezes_every_adapter(self):
        model = make_model(encoder_mode="contextual", freeze_lora=True)
        params = self._lora_params(model)
        assert params, "sanity: the model does have adapters"
        assert not any(p.requires_grad for _, p in params)

    def test_adapters_stay_in_the_forward_pass(self):
        """Frozen is not removed: the loaded adapter values must still apply."""
        ids, _ = batch()
        trained = make_model(encoder_mode="contextual", freeze_lora=True)
        with torch.no_grad():
            for name, param in trained.encoder.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.05)      # make the adapters do something
        model_off = make_model(encoder_mode="contextual", freeze_lora=True)
        trained.eval(); model_off.eval()      # dropout would mask the effect
        with torch.no_grad():
            a = trained(ids)[0]
            b = model_off(ids)[0]
        assert not torch.allclose(a, b)

    def test_gnn_and_heads_still_train(self):
        model = make_model(encoder_mode="contextual", freeze_lora=True)
        assert sum(p.numel() for p in model.gnn.parameters()
                   if p.requires_grad) > 0
        assert sum(p.numel() for p in model.classifier.parameters()
                   if p.requires_grad) > 0

    def test_combines_with_freeze_encoder(self):
        """The encoder runs, yet nothing inside it can move."""
        model = make_model(encoder_mode="contextual", freeze_encoder=True,
                           freeze_lora=True)
        assert sum(p.numel() for p in model.encoder.parameters()
                   if p.requires_grad) == 0
        assert model.encoder_mode == "contextual"

    def test_is_a_no_op_without_lora(self):
        model = make_model(encoder_mode="contextual", use_lora=False,
                           freeze_lora=True)
        assert self._lora_params(model) == []


# ----------------------------------------------------------------------
# Slim checkpoints must survive a reload in a fresh process
# ----------------------------------------------------------------------

class TestSlimKeepsAdapters:
    """A frozen adapter is still part of the model.

    The slim format omits what `from_pretrained` restores deterministically.
    LoRA tensors live under the encoder prefix but are absent from the
    pretrained file: reloading recreates them with lora_B zeroed, which is an
    exact no-op. Dropping a frozen adapter therefore does not save space, it
    changes the model -- and only on the next load, in a different process,
    long after the run that produced the numbers.
    """

    @staticmethod
    def _saved_names(path, model):
        from model import save_checkpoint_weights
        save_checkpoint_weights(str(path), model, slim=True)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        return set(payload["trainable"])

    def test_trainable_adapters_are_stored(self, tmp_path):
        model = make_model(encoder_mode="contextual")
        names = self._saved_names(tmp_path / "hot.bin", model)
        assert [n for n in names if "lora_" in n]

    def test_frozen_adapters_are_stored_too(self, tmp_path):
        model = make_model(encoder_mode="contextual", freeze_lora=True)
        names = self._saved_names(tmp_path / "cold.bin", model)
        lora = [n for n in names if "lora_" in n]
        assert lora, "a frozen adapter was dropped from the slim checkpoint"

    def test_frozen_and_trainable_store_the_same_tensors(self, tmp_path):
        """The two ablation arms must round-trip through the same shape."""
        hot = self._saved_names(tmp_path / "a.bin",
                                make_model(encoder_mode="contextual"))
        cold = self._saved_names(tmp_path / "b.bin",
                                 make_model(encoder_mode="contextual",
                                            freeze_lora=True))
        assert hot == cold

    def test_frozen_adapter_values_survive_a_reload(self, tmp_path):
        """The failure this guards: same input, different answer after reload."""
        from model import load_checkpoint_weights, save_checkpoint_weights

        trained = make_model(encoder_mode="contextual", freeze_lora=True)
        with torch.no_grad():
            for name, param in trained.encoder.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.05)          # a non-trivial adapter
        ids, _ = batch()
        trained.eval()                        # dropout would mask the effect
        with torch.no_grad():
            before = trained(ids)[0].clone()

        path = str(tmp_path / "slim.bin")
        save_checkpoint_weights(path, trained, slim=True)

        # A fresh model, as a new process would build it: adapters back at
        # their initialisation, lora_B zeroed.
        fresh = make_model(encoder_mode="contextual", freeze_lora=True)
        load_checkpoint_weights(path, fresh, torch.device("cpu"))
        fresh.eval()
        with torch.no_grad():
            after = fresh(ids)[0]

        assert torch.allclose(before, after, atol=1e-6)

    def test_encoder_weights_are_still_omitted(self, tmp_path):
        """The space saving must survive the fix."""
        model = make_model(encoder_mode="contextual", freeze_lora=True)
        names = self._saved_names(tmp_path / "bulk.bin", model)
        bulk = [n for n in names
                if n.startswith("encoder.roberta.") and "lora_" not in n]
        assert bulk == []
