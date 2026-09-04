"""MegaVul C/C++ -> JSONL splits the detector can read.

MegaVul ships as one 1.2 GB JSON array (`megavul_simple.json`); `json.load` on
it needs several GB of RAM, so it is streamed object by object instead.

Labelling follows the dataset's own construction. A record with `is_vul=true`
describes one function touched by a CVE fix, and carries both sides of it:

    func_before -> target 1   (the vulnerable version)
    func        -> target 0   (the same function after the fix)

Both get the same `pair` id, which makes a leakage-free ranking check possible:
the two texts differ by the patch alone, so "does the model score the vulnerable
one higher?" has a known answer and a 0.5 chance baseline. A record with
`is_vul=false` is a function that the fix commit touched incidentally; it
contributes its post-fix body as a negative.

Splitting is **by CVE**, never at random. `func_before` and `func` differ by a
line or two, so a random split would put near-identical texts on both sides of
it and inflate every metric.

Usage:
    python dataset/megavul.py extract --src <path>/megavul_simple.json --out-dir data/megavul
    python dataset/megavul.py split   --out-dir data/megavul
"""

import argparse
import collections
import hashlib
import json
import os
import random
import time


def iter_json_array(path, chunk=8 << 20):
    """Yield objects from a top-level JSON array without loading the file.

    Running out of buffer is not the end of the array: it only means the next
    read has not happened yet. Treating the two as the same thing truncates the
    corpus silently -- no error, just fewer functions than the file holds --
    which is why the refill below is separate from the end-of-array test.
    """
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as fh:
        buf = fh.read(chunk)
        buf = buf[buf.find("[") + 1:]
        while True:
            # Skip separators, pulling more data whenever the buffer runs dry.
            while True:
                buf = buf.lstrip()
                while buf[:1] == ",":
                    buf = buf[1:].lstrip()
                if buf:
                    break
                more = fh.read(chunk)
                if not more:                           # end of file
                    return
                buf += more
            if buf[:1] == "]":                         # end of array
                return
            while True:
                try:
                    obj, end = decoder.raw_decode(buf)
                    break
                except ValueError:
                    more = fh.read(chunk)
                    if not more:                       # truncated file
                        return
                    buf += more
            yield obj
            buf = buf[end:]


def _sha(text):
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def extract(src, out_path):
    """Flatten MegaVul into one JSONL row per unique function."""
    started = time.time()
    stats = collections.Counter()
    seen = set()

    with open(out_path, "w") as out:
        for n, rec in enumerate(iter_json_array(src), 1):
            stats["records"] += 1
            base = {
                "cve": rec.get("cve_id"),
                "cwe": rec.get("cwe_ids") or [],
                "repo": rec.get("repo_name"),
                "file": rec.get("file_path"),
                "fn": rec.get("func_name"),
                "pair": _sha(rec["commit_hash"] + rec["file_path"]
                             + rec["func_name"]),
            }
            if rec["is_vul"]:
                rows = [(rec["func_before"], 1, "before"), (rec["func"], 0, "after")]
            else:
                rows = [(rec["func"], 0, "nonvul")]

            for code, target, role in rows:
                if not code or not code.strip():
                    stats["empty"] += 1
                    continue
                # Exact duplicates are common across records and would leak
                # across splits as well as double-count in the metrics.
                key = _sha(code)
                if key in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(key)
                stats["emit_" + role] += 1
                out.write(json.dumps({**base, "func": code, "target": target,
                                      "role": role, "sha": key}) + "\n")

            if n % 50000 == 0:
                print(f"  {n} records, {time.time() - started:.0f}s", flush=True)

    print(f"extract: {stats['records']} records -> {out_path} "
          f"in {time.time() - started:.0f}s")
    print("  " + json.dumps(dict(stats)))
    return stats


def split(all_path, out_dir, seed=42, train=0.70, valid=0.15):
    """Partition by CVE so a fix and its vulnerable original never separate."""
    rows = [json.loads(line) for line in open(all_path)]
    cves = sorted({r["cve"] or "<none>" for r in rows})
    random.Random(seed).shuffle(cves)

    n = len(cves)
    assignment = {}
    for cve in cves[:int(train * n)]:
        assignment[cve] = "train"
    for cve in cves[int(train * n):int((train + valid) * n)]:
        assignment[cve] = "valid"
    for cve in cves[int((train + valid) * n):]:
        assignment[cve] = "test"

    handles = {s: open(os.path.join(out_dir, f"megavul_{s}.jsonl"), "w")
               for s in ("train", "valid", "test")}
    counts = collections.Counter()
    for i, row in enumerate(rows):
        name = assignment[row["cve"] or "<none>"]
        row["idx"] = str(i)                  # TextDataset requires the field
        counts[name] += 1
        counts[name + "_pos"] += row["target"]
        handles[name].write(json.dumps(row) + "\n")
    for handle in handles.values():
        handle.close()

    for name in ("train", "valid", "test"):
        total, pos = counts[name], counts[name + "_pos"]
        print(f"{name:6s} {total:7d} functions, {pos:6d} positive "
              f"({pos / total:.2%})")
    return counts


def subsample(src_path, out_path, neg_ratio=None, size=None, seed=42):
    """Shrink a split, either by rebalancing it or by sampling it uniformly.

    `neg_ratio` keeps every positive and that many negatives per positive. It
    is a **training** device: at a 4.7% positive rate an epoch is almost all
    negatives, and most of the compute buys nothing. It changes the class
    prior, so a file built this way must never be used to evaluate -- precision
    and PR-AUC would be read against a prior the deployment never sees.

    `size` samples uniformly and leaves the prior alone, which is what a
    validation subset for early stopping needs: full validation passes cost
    minutes per epoch in contextual mode.
    """
    if (neg_ratio is None) == (size is None):
        raise ValueError("pass exactly one of neg_ratio / size")

    rows = [json.loads(line) for line in open(src_path)]
    rng = random.Random(seed)

    if neg_ratio is not None:
        positives = [r for r in rows if r["target"] == 1]
        negatives = [r for r in rows if r["target"] == 0]
        keep = min(len(negatives), int(round(neg_ratio * len(positives))))
        chosen = positives + rng.sample(negatives, keep)
        note = (f"rebalanced to 1:{neg_ratio} -- FOR TRAINING ONLY, "
                f"the class prior no longer matches the corpus")
    else:
        chosen = rng.sample(rows, min(size, len(rows)))
        note = "uniform sample, class prior preserved"

    rng.shuffle(chosen)
    with open(out_path, "w") as out:
        for row in chosen:
            out.write(json.dumps(row) + "\n")

    pos = sum(r["target"] for r in chosen)
    print(f"{src_path} -> {out_path}")
    print(f"  {len(chosen)} functions, {pos} positive ({pos / len(chosen):.2%})")
    print(f"  source was {len(rows)} functions, "
          f"{sum(r['target'] for r in rows) / len(rows):.2%} positive")
    print(f"  {note}")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="megavul_simple.json -> JSONL")
    p_extract.add_argument("--src", required=True)
    p_extract.add_argument("--out-dir", default="data/megavul")

    p_split = sub.add_parser("split", help="JSONL -> train/valid/test by CVE")
    p_split.add_argument("--out-dir", default="data/megavul")
    p_split.add_argument("--seed", type=int, default=42)

    p_sub = sub.add_parser("subsample", help="shrink one split")
    p_sub.add_argument("--split", required=True,
                       choices=["train", "valid", "test"])
    p_sub.add_argument("--out", required=True)
    p_sub.add_argument("--neg-ratio", type=float, default=None,
                       help="keep all positives and this many negatives each "
                            "(training only: it changes the class prior)")
    p_sub.add_argument("--size", type=int, default=None,
                       help="uniform sample of this many rows (prior kept)")
    p_sub.add_argument("--out-dir", default="data/megavul")
    p_sub.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    all_path = os.path.join(args.out_dir, "megavul_all.jsonl")

    if args.command == "extract":
        extract(args.src, all_path)
    elif args.command == "split":
        split(all_path, args.out_dir, seed=args.seed)
    else:
        if args.neg_ratio is not None and args.split != "train":
            raise SystemExit(
                f"--neg-ratio rebalances the classes, so the result cannot be "
                f"used to measure anything; refusing to build it from "
                f"'{args.split}'. Use --size to shrink an evaluation split.")
        subsample(os.path.join(args.out_dir, f"megavul_{args.split}.jsonl"),
                  args.out, neg_ratio=args.neg_ratio, size=args.size,
                  seed=args.seed)


if __name__ == "__main__":
    main()
