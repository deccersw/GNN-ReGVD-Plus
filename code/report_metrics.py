"""Turn a `predict_dump.py` file into the metric set an imbalanced corpus needs.

Three things this reports that a bare accuracy/F1/ROC-AUC line does not:

* **PR-AUC against its own random baseline.** At a 4.7% positive rate ROC-AUC
  stays comfortably above 0.5 while the model is useless; PR-AUC divided by the
  positive rate says how much better than chance the ranking actually is.
* **The all-negative accuracy.** Predicting "safe" for everything scores
  1 - positive_rate. Any accuracy below that line is worse than silence.
* **A length baseline.** "Longer functions are likelier to be vulnerable" is a
  one-line predictor that needs no model at all. If it wins, the reported
  numbers are about function length, not about code.

With `--paired`, rows carrying a `pair` id and a `role` of before/after are also
compared head to head: the two texts differ by the patch alone, so the fraction
of pairs where the vulnerable version scores higher has a 0.5 chance baseline
and is immune to both class imbalance and label leakage.

Usage:
    python code/report_metrics.py runs/megavul_test.static.jsonl --paired
"""

import argparse
import collections
import json

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

SWEEP = (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7)
BUDGETS = (0.01, 0.05, 0.10, 0.20)


def load(path):
    return [json.loads(line) for line in open(path)]


def headline(y, p):
    n, pos = len(y), int(y.sum())
    rate = pos / n
    print(f"n={n}  positive={pos} ({rate:.3%})")
    print(f"all-negative accuracy = {1 - rate:.4f}   "
          f"PR-AUC random baseline = {rate:.4f}")
    print()
    ap = average_precision_score(y, p)
    print(f"ROC-AUC = {roc_auc_score(y, p):.4f}")
    print(f"PR-AUC  = {ap:.4f}   (lift over random x{ap / rate:.2f})")
    return rate


def at_threshold(y, p, threshold=0.5):
    print(f"\n--- at threshold {threshold} ---")
    pred = p > threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"accuracy  = {(y == pred).mean():.4f}")
    print(f"precision = {precision_score(y, pred, zero_division=0):.4f}")
    print(f"recall    = {recall_score(y, pred, zero_division=0):.4f}")
    print(f"F1        = {f1_score(y, pred, zero_division=0):.4f}")
    print(f"flag rate = {pred.mean():.4f}   TP={tp} FP={fp} FN={fn} TN={tn}")


def sweep(y, p):
    print("\n--- threshold sweep ---")
    print(f"{'thr':>6} {'flag%':>7} {'prec':>8} {'rec':>8} {'F1':>8}")
    for t in SWEEP:
        q = p > t
        print(f"{t:6.2f} {q.mean() * 100:6.2f}% "
              f"{precision_score(y, q, zero_division=0):8.4f} "
              f"{recall_score(y, q, zero_division=0):8.4f} "
              f"{f1_score(y, q, zero_division=0):8.4f}")


def alert_budget(y, p):
    print("\n--- recall at a fixed alert budget ---")
    n, pos = len(y), int(y.sum())
    order = np.argsort(-p)
    for frac in BUDGETS:
        k = int(n * frac)
        hit = int(y[order[:k]].sum())
        print(f"top {frac * 100:4.0f}% ({k:6d} alerts): "
              f"recall={hit / pos:.4f}  precision={hit / k:.4f}")


def length_baseline(rows, y, p):
    lengths = np.array([r.get("n_tokens", 0) for r in rows], dtype=float)
    if not lengths.any():
        return
    print("\n--- control: does raw function length predict better? ---")
    rate = y.mean()
    print(f"{'predictor':<24} {'ROC-AUC':>9} {'PR-AUC':>9} {'lift':>7}")
    for name, score in (("detector score", p), ("token length", lengths)):
        ap = average_precision_score(y, score)
        print(f"{name:<24} {roc_auc_score(y, score):9.4f} {ap:9.4f} "
              f"{ap / rate:7.2f}")


def by_truncation(rows, y, p):
    """Split the metrics by whether the detector saw the whole function.

    Anything past block_size - 2 tokens never reaches the model, so a function
    over that length is judged on a prefix. Reporting the two groups apart
    separates "the model is wrong" from "the model never saw the relevant code",
    and exposes the case where the overall number is carried by the difference
    between the groups rather than by discrimination inside either.
    """
    flags = np.array([bool(r.get("truncated", False)) for r in rows])
    if not flags.any() or flags.all():
        return
    print("\n--- split by whether the function fits the token window ---")
    for name, mask in (("fits", ~flags), ("truncated", flags)):
        yy, pp = y[mask], p[mask]
        pos = int(yy.sum())
        if pos == 0 or pos == len(yy):
            continue
        rate = pos / len(yy)
        ap = average_precision_score(yy, pp)
        print(f"{name:<10} n={mask.sum():6d}  positive={rate:.2%}  "
              f"ROC-AUC={roc_auc_score(yy, pp):.4f}  "
              f"PR-AUC={ap:.4f} (lift x{ap / rate:.2f})")


#: Score gap below which two predictions count as the same answer. A pair whose
#: two functions differ only past the token window reaches the model as one
#: input and must tie, but floating-point reduction order varies with batch
#: size, so those ties come back as differences around 1e-8. Without a
#: tolerance the tie count -- and every rate derived from it -- changes with
#: --batch_size.
PAIR_TOL = 1e-6


def paired(rows, tol=PAIR_TOL):
    by_pair = collections.defaultdict(dict)
    for row in rows:
        if row.get("role") in ("before", "after"):
            by_pair[row["pair"]][row["role"]] = row["prob"]
    pairs = [(v["before"], v["after"]) for v in by_pair.values() if len(v) == 2]
    if not pairs:
        return
    before = np.array([x[0] for x in pairs])
    after = np.array([x[1] for x in pairs])
    ties = float((np.abs(before - after) <= tol).mean())
    wins = float((before > after + tol).mean())
    print(f"\n--- paired: vulnerable vs its own fix ({len(pairs)} pairs) ---")
    print(f"score(before) > score(after) = {wins:.4f}   (chance 0.5)")
    print(f"tied within {tol:g}          = {ties:.4f}")
    if ties < 1:
        print(f"among non-tied pairs         = {wins / (1 - ties):.4f}")
    print(f"median delta                 = {np.median(before - after):+.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="output of predict_dump.py")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--paired", action="store_true",
                        help="also run the before/after ranking check")
    args = parser.parse_args()

    rows = load(args.predictions)
    y = np.array([r["label"] for r in rows])
    p = np.array([r["prob"] for r in rows])

    headline(y, p)
    at_threshold(y, p, args.threshold)
    sweep(y, p)
    alert_budget(y, p)
    length_baseline(rows, y, p)
    by_truncation(rows, y, p)
    if args.paired:
        paired(rows)


if __name__ == "__main__":
    main()
