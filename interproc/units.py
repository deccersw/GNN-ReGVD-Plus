"""
Stages 5 and 8: pick roots, build units, report statistics.

This is the module's public entry point: ``InterprocPipeline(cfg).build(path)``
turns a project directory into a list of AnalysisUnits.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional, Set

from .budget import TokenCounter
from .bundler import Bundler
from .callgraph import CallGraph
from .config import InliningConfig
from .discovery import ProjectDiscovery, is_test_file
from .inliner import Inliner
from .models import AnalysisUnit, FuncId, FunctionDef, ProjectIndex
from .scoring import CallSiteScorer, has_memory_op, has_sink
from .symbols import build_index

logger = logging.getLogger(__name__)


@dataclass
class BuildStats:
    project: str = ""
    files_parsed: int = 0
    functions_found: int = 0
    roots_selected: int = 0
    roots_skipped_trivial: int = 0
    roots_skipped_test: int = 0
    roots_skipped_duplicate: int = 0
    units_built: int = 0
    units_with_inlining: int = 0
    units_truncated: int = 0
    total_inlined_calls: int = 0
    call_resolution: Dict[str, int] = field(default_factory=dict)
    skip_reasons: Dict[str, int] = field(default_factory=dict)
    avg_tokens_before: float = 0.0
    avg_tokens_after: float = 0.0
    avg_growth: float = 0.0
    tokenizer_exact: bool = False
    parser_backend: str = ""
    elapsed_sec: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class InterprocPipeline:
    """Project directory -> list of AnalysisUnit."""

    def __init__(self, config: Optional[InliningConfig] = None):
        self.config = config or InliningConfig()
        self.counter = TokenCounter(self.config.tokenizer_name,
                                    self.config.use_real_tokenizer)
        self.stats = BuildStats()
        self.index: Optional[ProjectIndex] = None
        self.graph: Optional[CallGraph] = None

    # ------------------------------------------------------------------

    def analyze(self, project_path: str) -> ProjectIndex:
        """Stages 1-4: parse the project and resolve its call graph."""
        started = time.time()
        discovery = ProjectDiscovery(self.config)
        facts = discovery.collect(project_path)

        self.index = build_index(project_path, facts, self.config)
        self.graph = CallGraph(self.index, self.config).build()

        self.stats.project = project_path
        self.stats.files_parsed = discovery.stats.files_parsed
        self.stats.functions_found = len(self.index.functions)
        self.stats.call_resolution = self.graph.summary()
        self.stats.parser_backend = discovery.backend.name
        self.stats.elapsed_sec = round(time.time() - started, 3)
        return self.index

    def build(self, project_path: str) -> List[AnalysisUnit]:
        return list(self.iter_units(project_path))

    def iter_units(self, project_path: str) -> Iterator[AnalysisUnit]:
        started = time.time()
        if self.index is None:
            self.analyze(project_path)

        inliner = Inliner(self.index, self.graph, self.config, self.counter)
        bundler = Bundler(self.index, self.graph, self.config, self.counter)

        before_total = 0
        after_total = 0
        roots = self.select_roots()
        self.stats.roots_selected = len(roots)

        for idx, fid in enumerate(roots):
            fn = self.index.functions[fid]
            expansion = inliner.expand_root(fid)
            bundle = bundler.build(fid)

            base_tokens = self.counter.count(fn.func_text, key=fn.body_hash)
            before_total += base_tokens
            after_total += expansion.tokens

            for rec in expansion.skipped:
                self.stats.skip_reasons[rec.reason] = \
                    self.stats.skip_reasons.get(rec.reason, 0) + 1

            unit = AnalysisUnit(
                unit_id=self._unit_id(fid),
                root=fid,
                code_for_gnn=expansion.code,
                code_for_sandbox=bundle.code,
                segments=expansion.segments,
                inlined=expansion.inlined,
                skipped=expansion.skipped,
                depth_used=expansion.depth_used,
                tokens_gnn=expansion.tokens,
                tokens_sandbox=bundle.tokens,
                truncated=expansion.truncated,
                provenance_files=sorted(
                    set(expansion.provenance) | set(bundle.files)),
                sandbox_functions=bundle.functions,
                stats={
                    "tokens_before_inlining": base_tokens,
                    "inlined_count": len(expansion.inlined),
                    "skipped_count": len(expansion.skipped),
                    "bundle_dropped": len(bundle.dropped),
                    "max_depth": self.config.max_depth,
                    "strategy": self.config.strategy,
                },
            )

            self.stats.units_built += 1
            if expansion.inlined:
                self.stats.units_with_inlining += 1
                self.stats.total_inlined_calls += len(expansion.inlined)
            if expansion.truncated:
                self.stats.units_truncated += 1

            yield unit

        n = max(1, self.stats.units_built)
        self.stats.avg_tokens_before = round(before_total / n, 1)
        self.stats.avg_tokens_after = round(after_total / n, 1)
        self.stats.avg_growth = round(after_total / before_total, 3) \
            if before_total else 0.0
        self.stats.tokenizer_exact = self.counter.exact
        self.stats.elapsed_sec = round(time.time() - started, 3)

        logger.info("[Interproc] %d units built from %d functions in %.2fs "
                    "(inlining applied in %d, truncated %d, growth x%.2f)",
                    self.stats.units_built, self.stats.functions_found,
                    self.stats.elapsed_sec, self.stats.units_with_inlining,
                    self.stats.units_truncated, self.stats.avg_growth)

    # ------------------------------------------------------------------
    # Stage 5: root selection
    # ------------------------------------------------------------------

    def select_roots(self) -> List[FuncId]:
        cfg = self.config
        roots: List[FuncId] = []
        seen_bodies: Set[str] = set()

        for fid in sorted(self.index.functions):
            fn = self.index.functions[fid]

            if cfg.root_skip_test_files and is_test_file(fid.file):
                self.stats.roots_skipped_test += 1
                continue

            if cfg.dedupe_identical_bodies:
                digest = fn.body_hash
                if digest in seen_bodies:
                    self.stats.roots_skipped_duplicate += 1
                    continue
                seen_bodies.add(digest)

            if self._is_trivial_root(fn):
                self.stats.roots_skipped_trivial += 1
                continue

            roots.append(fid)
            if cfg.max_roots and len(roots) >= cfg.max_roots:
                break

        return roots

    def _is_trivial_root(self, fn: FunctionDef) -> bool:
        """Trivial means *uninteresting*, not merely short (decision D-005)."""
        if fn.calls:
            return False
        if has_sink(fn) or has_memory_op(fn):
            return False
        tokens = self.counter.count(fn.func_text, key=fn.body_hash)
        return tokens < self.config.root_min_tokens

    # ------------------------------------------------------------------

    def _unit_id(self, fid: FuncId) -> str:
        raw = f"{fid}|{self.config.config_hash()}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]


def write_units_jsonl(units: List[AnalysisUnit], path: str) -> int:
    with open(path, "w", encoding="utf-8") as fh:
        for i, unit in enumerate(units):
            fh.write(json.dumps(unit.to_jsonl_record(idx=i),
                                ensure_ascii=False) + "\n")
    return len(units)


def read_units_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
