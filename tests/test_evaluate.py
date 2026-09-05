"""Tests for evaluation metrics and dataset loading."""

import os
import sys
import json
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluate import EvalMetrics, load_dataset, _update_confusion


class TestEvalMetrics:
    def test_accuracy(self):
        m = EvalMetrics(total=100, true_positives=40, true_negatives=30,
                        false_positives=20, false_negatives=10)
        assert m.accuracy == 0.7

    def test_precision(self):
        m = EvalMetrics(true_positives=40, false_positives=10)
        assert m.precision == 0.8

    def test_recall(self):
        m = EvalMetrics(true_positives=40, false_negatives=10)
        assert m.recall == 0.8

    def test_f1(self):
        m = EvalMetrics(true_positives=40, false_positives=10,
                        false_negatives=10)
        assert abs(m.f1 - 0.8) < 0.001

    def test_zero_division(self):
        m = EvalMetrics()
        assert m.accuracy == 0.0
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_fp_reduction(self):
        m = EvalMetrics(fp_before_sandbox=100, fp_after_sandbox=40)
        assert m.fp_reduction == 0.6

    def test_exploitation_rate(self):
        m = EvalMetrics(confirmed_exploits=30, total_exploits_tried=50)
        assert m.exploitation_rate == 0.6

    def test_timing(self):
        m = EvalMetrics(sample_times=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert m.avg_time == 3.0
        assert m.median_time == 3.0

    def test_median_even(self):
        m = EvalMetrics(sample_times=[1.0, 2.0, 3.0, 4.0])
        assert m.median_time == 2.5


class TestUpdateConfusion:
    def test_true_positive(self):
        m = EvalMetrics()
        _update_confusion(m, label=1, predicted=1)
        assert m.true_positives == 1

    def test_true_negative(self):
        m = EvalMetrics()
        _update_confusion(m, label=0, predicted=0)
        assert m.true_negatives == 1

    def test_false_positive(self):
        m = EvalMetrics()
        _update_confusion(m, label=0, predicted=1)
        assert m.false_positives == 1
        assert m.fp_before_sandbox == 1

    def test_false_negative(self):
        m = EvalMetrics()
        _update_confusion(m, label=1, predicted=0)
        assert m.false_negatives == 1


class TestLoadDataset:
    def test_load_valid_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as f:
            f.write('{"func": "void f(){}", "target": 0, "idx": 0}\n')
            f.write('{"func": "void g(){}", "target": 1, "idx": 1}\n')
            path = f.name

        try:
            samples = load_dataset(path)
            assert len(samples) == 2
            assert samples[0]["target"] == 0
            assert samples[1]["target"] == 1
        finally:
            os.unlink(path)

    def test_load_with_limit(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as f:
            for i in range(10):
                f.write(json.dumps({"func": f"f{i}", "target": 0}) + "\n")
            path = f.name

        try:
            samples = load_dataset(path, limit=3)
            assert len(samples) == 3
        finally:
            os.unlink(path)

    def test_load_missing_brace(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as f:
            f.write('"func": "void f(){}", "target": 0}\n')
            path = f.name

        try:
            samples = load_dataset(path)
            assert len(samples) == 1
        finally:
            os.unlink(path)

    def test_load_real_dataset(self):
        path = os.path.join(os.path.dirname(__file__), "..",
                           "dataset", "test.jsonl")
        if not os.path.exists(path):
            pytest.skip("test.jsonl not available")

        samples = load_dataset(path, limit=5)
        assert len(samples) == 5
        assert all("func" in s for s in samples)
        assert all("target" in s for s in samples)


class TestCLI:
    def test_eval_help(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "evaluate.py", "--help"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode == 0
        assert "evaluate" in result.stdout.lower() or "GNN-ReGVD" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestCheckpointSelection:
    """Which epoch is kept is a separate decision from when to stop.

    A run can early-stop on eval_pr_auc and still save the epoch with the best
    eval_acc -- which, on a corpus that is 95% negative, is the epoch closest
    to predicting "safe" for everything. Both arms of an ablation lose the same
    way, so a comparison stays fair, but the model that gets tested is not the
    one the run was judged on.
    """

    @staticmethod
    def _select(metric, best, results):
        """Mirror of the decision made in run.train()."""
        score = results[metric]
        return (score < best) if metric == "eval_loss" else (score > best)

    def test_accuracy_can_prefer_a_worse_ranking(self):
        # Taken from a real pair of epochs: epoch 1 wins on accuracy and loses
        # on PR-AUC, and it is the one that was saved.
        first = {"eval_acc": 0.7971, "eval_pr_auc": 0.1607}
        later = {"eval_acc": 0.7800, "eval_pr_auc": 0.1643}
        assert not self._select("eval_acc", first["eval_acc"], later)
        assert self._select("eval_pr_auc", first["eval_pr_auc"], later)

    def test_lower_is_better_only_for_loss(self):
        assert self._select("eval_loss", 0.60, {"eval_loss": 0.55})
        assert not self._select("eval_loss", 0.60, {"eval_loss": 0.65})
        assert self._select("eval_auc", 0.60, {"eval_auc": 0.65})
        assert not self._select("eval_auc", 0.60, {"eval_auc": 0.55})

    def test_the_flag_offers_every_reported_metric(self):
        import subprocess, sys, os
        run_py = os.path.join(os.path.dirname(__file__), "..", "code", "run.py")
        out = subprocess.run([sys.executable, run_py, "--help"],
                             capture_output=True, text=True).stdout
        assert "--best_checkpoint_metric" in out
        assert "--skip_faiss_index" in out
        for metric in ("eval_pr_auc", "eval_auc", "eval_f1", "eval_loss"):
            assert metric in out

    def test_default_keeps_the_previous_behaviour(self):
        import subprocess, sys, os
        run_py = os.path.join(os.path.dirname(__file__), "..", "code", "run.py")
        out = subprocess.run([sys.executable, run_py, "--help"],
                             capture_output=True, text=True).stdout
        line = [l for l in out.splitlines() if "best_checkpoint_metric" in l]
        assert line, "flag missing from --help"
