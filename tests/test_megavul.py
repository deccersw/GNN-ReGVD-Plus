"""Tests for the MegaVul preparation pipeline.

The split is the part worth guarding. MegaVul stores a vulnerable function and
its patched counterpart as two texts that differ by a line or two; if a random
split separates them, the model meets a near-copy of every test sample during
training and every metric comes out inflated. Splitting by CVE is what prevents
that, and nothing about the file format makes the mistake visible afterwards.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dataset"))

from megavul import extract, iter_json_array, split                # noqa: E402


def _record(cve, commit, fn, is_vul=True, before="int bad(){int x;}",
            after="int bad(){int x=0;}"):
    return {
        "cve_id": cve, "cwe_ids": ["CWE-457"], "repo_name": "demo",
        "commit_hash": commit, "file_path": "src/a.c", "func_name": fn,
        "is_vul": is_vul, "func_before": before, "func": after,
    }


@pytest.fixture
def corpus(tmp_path):
    records = []
    for i in range(40):
        records.append(_record(f"CVE-2020-{1000 + i}", f"c{i}", f"fn{i}",
                               before=f"int fn{i}(){{int x;}}",
                               after=f"int fn{i}(){{int x=0;}}"))
    for i in range(10):
        records.append(_record(f"CVE-2021-{i}", f"d{i}", f"g{i}", is_vul=False,
                               after=f"int g{i}(){{return {i};}}"))
    src = tmp_path / "megavul_simple.json"
    src.write_text(json.dumps(records))
    return src, tmp_path


class TestStreaming:
    def test_reads_every_object(self, corpus):
        src, _ = corpus
        assert len(list(iter_json_array(str(src)))) == 50

    def test_small_chunks_do_not_split_objects(self, corpus):
        src, _ = corpus
        # A chunk far smaller than one record forces raw_decode to fail and
        # refill repeatedly -- the path that actually matters on a 1.2 GB file.
        assert len(list(iter_json_array(str(src), chunk=16))) == 50

    def test_empty_array(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("[]")
        assert list(iter_json_array(str(path))) == []


class TestExtract:
    def test_vulnerable_record_yields_both_sides(self, corpus):
        src, tmp = corpus
        out = tmp / "all.jsonl"
        extract(str(src), str(out))
        rows = [json.loads(l) for l in open(out)]
        vuln = [r for r in rows if r["role"] == "before"]
        fixed = [r for r in rows if r["role"] == "after"]
        assert len(vuln) == len(fixed) == 40
        assert all(r["target"] == 1 for r in vuln)
        assert all(r["target"] == 0 for r in fixed)

    def test_both_sides_share_a_pair_id(self, corpus):
        src, tmp = corpus
        out = tmp / "all.jsonl"
        extract(str(src), str(out))
        rows = [json.loads(l) for l in open(out)]
        pairs = {}
        for row in rows:
            if row["role"] in ("before", "after"):
                pairs.setdefault(row["pair"], set()).add(row["role"])
        assert all(sides == {"before", "after"} for sides in pairs.values())

    def test_non_vulnerable_record_is_a_single_negative(self, corpus):
        src, tmp = corpus
        out = tmp / "all.jsonl"
        extract(str(src), str(out))
        rows = [json.loads(l) for l in open(out)]
        plain = [r for r in rows if r["role"] == "nonvul"]
        assert len(plain) == 10
        assert all(r["target"] == 0 for r in plain)

    def test_exact_duplicates_are_dropped(self, tmp_path):
        # The same body under two CVEs: keeping both would double-count it and
        # let it straddle the split.
        records = [_record("CVE-1", "a", "f", before="int f(){int x;}"),
                   _record("CVE-2", "b", "f", before="int f(){int x;}")]
        src = tmp_path / "dup.json"
        src.write_text(json.dumps(records))
        out = tmp_path / "all.jsonl"
        stats = extract(str(src), str(out))
        assert stats["duplicate"] == 2         # both before and after repeat
        bodies = [json.loads(l)["func"] for l in open(out)]
        assert len(bodies) == len(set(bodies))

    def test_empty_bodies_are_skipped(self, tmp_path):
        src = tmp_path / "empty_fn.json"
        src.write_text(json.dumps([_record("CVE-3", "c", "f", before="   ")]))
        out = tmp_path / "all.jsonl"
        stats = extract(str(src), str(out))
        assert stats["empty"] == 1
        assert all(json.loads(l)["role"] != "before" for l in open(out))


class TestSplit:
    @staticmethod
    def _read(tmp, name):
        return [json.loads(l) for l in
                open(os.path.join(str(tmp), f"megavul_{name}.jsonl"))]

    def _prepare(self, corpus):
        src, tmp = corpus
        all_path = tmp / "all.jsonl"
        extract(str(src), str(all_path))
        split(str(all_path), str(tmp))
        return tmp

    def test_no_cve_appears_in_two_splits(self, corpus):
        tmp = self._prepare(corpus)
        sets = {name: {r["cve"] for r in self._read(tmp, name)}
                for name in ("train", "valid", "test")}
        assert not sets["train"] & sets["valid"]
        assert not sets["train"] & sets["test"]
        assert not sets["valid"] & sets["test"]

    def test_a_fix_never_leaves_its_vulnerable_original(self, corpus):
        """The failure the CVE split exists to prevent."""
        tmp = self._prepare(corpus)
        home = {}
        for name in ("train", "valid", "test"):
            for row in self._read(tmp, name):
                if row["role"] in ("before", "after"):
                    home.setdefault(row["pair"], set()).add(name)
        assert all(len(where) == 1 for where in home.values())

    def test_every_row_survives_exactly_once(self, corpus):
        tmp = self._prepare(corpus)
        total = sum(len(self._read(tmp, n)) for n in ("train", "valid", "test"))
        assert total == len([json.loads(l) for l in open(tmp / "all.jsonl")])

    def test_rows_carry_the_idx_TextDataset_requires(self, corpus):
        tmp = self._prepare(corpus)
        rows = self._read(tmp, "train")
        assert rows and all("idx" in r for r in rows)

    def test_split_is_deterministic_for_a_seed(self, corpus):
        tmp = self._prepare(corpus)
        first = [r["idx"] for r in self._read(tmp, "test")]
        split(str(tmp / "all.jsonl"), str(tmp), seed=42)
        assert [r["idx"] for r in self._read(tmp, "test")] == first

    def test_a_different_seed_moves_cves(self, corpus):
        tmp = self._prepare(corpus)
        first = {r["cve"] for r in self._read(tmp, "test")}
        split(str(tmp / "all.jsonl"), str(tmp), seed=7)
        assert {r["cve"] for r in self._read(tmp, "test")} != first


class TestSubsample:
    """Shrinking a split is where a prior gets destroyed by accident.

    Rebalancing is legitimate for training and ruinous for evaluation: metrics
    read against a prior the corpus does not have are not wrong by a little,
    they are meaningless. The two modes are kept apart deliberately.
    """

    def _prepare(self, corpus):
        src, tmp = corpus
        all_path = tmp / "all.jsonl"
        extract(str(src), str(all_path))
        split(str(all_path), str(tmp))
        return tmp

    def test_neg_ratio_keeps_every_positive(self, corpus):
        from megavul import subsample
        tmp = self._prepare(corpus)
        src = os.path.join(str(tmp), "megavul_train.jsonl")
        rows = [json.loads(l) for l in open(src)]
        want = sum(r["target"] for r in rows)
        out = subsample(src, str(tmp / "sub.jsonl"), neg_ratio=1)
        assert sum(r["target"] for r in out) == want

    def test_neg_ratio_sets_the_negative_count(self, corpus):
        from megavul import subsample
        tmp = self._prepare(corpus)
        src = os.path.join(str(tmp), "megavul_train.jsonl")
        out = subsample(src, str(tmp / "sub.jsonl"), neg_ratio=2)
        pos = sum(r["target"] for r in out)
        neg = len(out) - pos
        assert neg == min(2 * pos,
                          sum(1 for l in open(src)
                              if json.loads(l)["target"] == 0))

    def test_size_preserves_rows_verbatim(self, corpus):
        from megavul import subsample
        tmp = self._prepare(corpus)
        src = os.path.join(str(tmp), "megavul_valid.jsonl")
        total = sum(1 for _ in open(src))
        out = subsample(src, str(tmp / "sub.jsonl"), size=max(1, total // 2))
        originals = {json.dumps(json.loads(l), sort_keys=True) for l in open(src)}
        assert all(json.dumps(r, sort_keys=True) in originals for r in out)

    def test_size_larger_than_the_split_is_not_an_error(self, corpus):
        from megavul import subsample
        tmp = self._prepare(corpus)
        src = os.path.join(str(tmp), "megavul_valid.jsonl")
        total = sum(1 for _ in open(src))
        out = subsample(src, str(tmp / "sub.jsonl"), size=total * 10)
        assert len(out) == total

    def test_exactly_one_mode_is_required(self, corpus):
        from megavul import subsample
        tmp = self._prepare(corpus)
        src = os.path.join(str(tmp), "megavul_train.jsonl")
        for kwargs in ({}, {"neg_ratio": 2, "size": 5}):
            with pytest.raises(ValueError):
                subsample(src, str(tmp / "sub.jsonl"), **kwargs)

    def test_is_deterministic_for_a_seed(self, corpus):
        from megavul import subsample
        tmp = self._prepare(corpus)
        src = os.path.join(str(tmp), "megavul_train.jsonl")
        first = subsample(src, str(tmp / "a.jsonl"), neg_ratio=1, seed=5)
        second = subsample(src, str(tmp / "b.jsonl"), neg_ratio=1, seed=5)
        assert [r["idx"] for r in first] == [r["idx"] for r in second]
