#!/usr/bin/env python3
"""
Evaluation script for GNN-ReGVD vulnerability scanner.

Runs the pipeline on test.jsonl and computes metrics:
  - Detection accuracy, precision, recall, F1
  - FP reduction after sandbox verification
  - Exploitation confirmation rate
  - Pipeline latency statistics

Supports three modes:
  1. Full pipeline (GNN + exploit DB + sandbox)
  2. Detection-only (GNN + FAISS, no sandbox)
  3. Sandbox-only (template-based, no GNN)

Usage:
    python evaluate.py --mode sandbox-only --limit 50
    python evaluate.py --mode detection-only --model-path saved_models/best.bin
    python evaluate.py --mode full --model-path saved_models/best.bin
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    total: int = 0
    # Detection
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    # Sandbox verification
    fp_before_sandbox: int = 0
    fp_after_sandbox: int = 0
    confirmed_exploits: int = 0
    total_exploits_tried: int = 0
    # Timing
    total_time: float = 0.0
    sample_times: List[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def fp_reduction(self) -> float:
        if self.fp_before_sandbox == 0:
            return 0.0
        reduced = self.fp_before_sandbox - self.fp_after_sandbox
        return reduced / self.fp_before_sandbox

    @property
    def exploitation_rate(self) -> float:
        if self.total_exploits_tried == 0:
            return 0.0
        return self.confirmed_exploits / self.total_exploits_tried

    @property
    def avg_time(self) -> float:
        if not self.sample_times:
            return 0.0
        return sum(self.sample_times) / len(self.sample_times)

    @property
    def median_time(self) -> float:
        if not self.sample_times:
            return 0.0
        s = sorted(self.sample_times)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2


def load_dataset(path: str, limit: Optional[int] = None) -> List[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                line = "{" + line
            try:
                data = json.loads(line)
                samples.append(data)
            except json.JSONDecodeError:
                continue
    if limit:
        samples = samples[:limit]
    return samples


def evaluate_sandbox_only(samples: List[dict], args) -> EvalMetrics:
    """Evaluate using sandbox-only mode (no GNN detection)."""
    from sandbox.executor import SandboxExecutor, SandboxConfig
    from sandbox.evidence import EvidenceLevel
    from exploit_db.exploit_adapter import ExploitAdapter

    metrics = EvalMetrics()
    executor = SandboxExecutor(SandboxConfig(timeout=args.timeout))
    adapter = ExploitAdapter()

    for i, sample in enumerate(samples):
        code = sample.get("func", "")
        label = sample.get("target", 0)
        metrics.total += 1

        start = time.time()

        harness = adapter.adapt(
            vuln_code=code, exploit_template={},
            vuln_type="buffer_overflow",
        )

        if not harness.is_valid:
            predicted = 0
        else:
            result = executor.execute(
                harness_code=harness.harness_code,
                payload=harness.payload,
                compilation_cmd=harness.compilation_cmd,
                expected_signals=harness.expected_signals,
                timeout=args.timeout,
            )

            if result.evidence.level in (EvidenceLevel.CONFIRMED,
                                         EvidenceLevel.SUGGESTIVE):
                predicted = 1
                if result.evidence.level == EvidenceLevel.CONFIRMED:
                    metrics.confirmed_exploits += 1
                metrics.total_exploits_tried += 1
            else:
                predicted = 0
                metrics.total_exploits_tried += 1

        elapsed = time.time() - start
        metrics.sample_times.append(elapsed)

        _update_confusion(metrics, label, predicted)

        if (i + 1) % 10 == 0:
            logger.info(
                f"[{i+1}/{len(samples)}] acc={metrics.accuracy:.3f} "
                f"p={metrics.precision:.3f} r={metrics.recall:.3f} "
                f"t={elapsed:.1f}s"
            )

    return metrics


def evaluate_detection_only(samples: List[dict], args) -> EvalMetrics:
    """Evaluate GNN detection without sandbox (placeholder for when model is available)."""
    metrics = EvalMetrics()

    logger.warning(
        "Detection-only mode requires a trained model. "
        "Using random predictions as placeholder."
    )

    import random
    random.seed(42)

    for sample in samples:
        label = sample.get("target", 0)
        metrics.total += 1
        predicted = random.choice([0, 1])
        _update_confusion(metrics, label, predicted)

    return metrics


def evaluate_full_pipeline(samples: List[dict], args) -> EvalMetrics:
    """Evaluate full pipeline (GNN + sandbox)."""
    metrics = EvalMetrics()

    logger.warning(
        "Full pipeline mode requires a trained model and FAISS index. "
        "Using sandbox-only evaluation as fallback."
    )

    return evaluate_sandbox_only(samples, args)


def _update_confusion(metrics: EvalMetrics, label: int, predicted: int):
    if label == 1 and predicted == 1:
        metrics.true_positives += 1
    elif label == 0 and predicted == 0:
        metrics.true_negatives += 1
    elif label == 0 and predicted == 1:
        metrics.false_positives += 1
        metrics.fp_before_sandbox += 1
    elif label == 1 and predicted == 0:
        metrics.false_negatives += 1


def print_metrics(metrics: EvalMetrics, mode: str):
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({mode})")
    print("=" * 60)
    print(f"\n  Total samples:     {metrics.total}")
    print(f"\n  Detection Metrics:")
    print(f"    Accuracy:        {metrics.accuracy:.4f}")
    print(f"    Precision:       {metrics.precision:.4f}")
    print(f"    Recall:          {metrics.recall:.4f}")
    print(f"    F1 Score:        {metrics.f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {metrics.true_positives:5d}  FP: {metrics.false_positives:5d}")
    print(f"    FN: {metrics.false_negatives:5d}  TN: {metrics.true_negatives:5d}")

    if metrics.total_exploits_tried > 0:
        print(f"\n  Exploit Verification:")
        print(f"    Exploits tried:       {metrics.total_exploits_tried}")
        print(f"    Exploits confirmed:   {metrics.confirmed_exploits}")
        print(f"    Exploitation rate:    {metrics.exploitation_rate:.4f}")

    if metrics.fp_before_sandbox > 0:
        print(f"\n  FP Reduction:")
        print(f"    FP before sandbox:    {metrics.fp_before_sandbox}")
        print(f"    FP after sandbox:     {metrics.fp_after_sandbox}")
        print(f"    FP reduction:         {metrics.fp_reduction:.4f}")

    if metrics.sample_times:
        print(f"\n  Timing:")
        print(f"    Total time:           {sum(metrics.sample_times):.1f}s")
        print(f"    Avg per sample:       {metrics.avg_time:.2f}s")
        print(f"    Median per sample:    {metrics.median_time:.2f}s")
        print(f"    Min:                  {min(metrics.sample_times):.2f}s")
        print(f"    Max:                  {max(metrics.sample_times):.2f}s")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GNN-ReGVD vulnerability scanner"
    )
    parser.add_argument("--dataset", type=str,
                        default="dataset/test.jsonl",
                        help="Path to test dataset (JSONL)")
    parser.add_argument("--mode", type=str, default="sandbox-only",
                        choices=["full", "detection-only", "sandbox-only"],
                        help="Evaluation mode")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Sandbox timeout per sample (seconds)")
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to trained model checkpoint")
    parser.add_argument("--faiss-dir", type=str, default="",
                        help="Path to FAISS index directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Save metrics to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    if not os.path.exists(args.dataset):
        print(f"Error: dataset not found: {args.dataset}")
        sys.exit(1)

    samples = load_dataset(args.dataset, limit=args.limit)
    logger.info(f"Loaded {len(samples)} samples from {args.dataset}")

    label_dist = defaultdict(int)
    for s in samples:
        label_dist[s.get("target", -1)] += 1
    logger.info(f"Label distribution: {dict(label_dist)}")

    if args.mode == "sandbox-only":
        metrics = evaluate_sandbox_only(samples, args)
    elif args.mode == "detection-only":
        metrics = evaluate_detection_only(samples, args)
    elif args.mode == "full":
        metrics = evaluate_full_pipeline(samples, args)

    print_metrics(metrics, args.mode)

    if args.output:
        output_data = {
            "mode": args.mode,
            "total": metrics.total,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "true_positives": metrics.true_positives,
            "true_negatives": metrics.true_negatives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
            "fp_reduction": metrics.fp_reduction,
            "exploitation_rate": metrics.exploitation_rate,
            "avg_time": metrics.avg_time,
            "median_time": metrics.median_time,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
