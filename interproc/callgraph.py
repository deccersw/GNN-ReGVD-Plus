"""
Stage 4: resolve call sites to definitions.

The rules are applied in a fixed order and every outcome is recorded, including
the refusals. "I could not tell which function this is" is a first-class result
here: guessing wrong would splice the body of the wrong function into the text
we hand to the detector, which is worse than not inlining at all.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .config import InliningConfig
from .models import CallEdge, CallSite, FuncId, FunctionDef, ProjectIndex
from .symbols import visible_files

logger = logging.getLogger(__name__)


class CallGraph:
    def __init__(self, index: ProjectIndex, config: InliningConfig):
        self.index = index
        self.config = config
        self._never = set(config.never_inline)
        self.edges: Dict[FuncId, List[CallEdge]] = defaultdict(list)
        self.callers: Dict[FuncId, Set[FuncId]] = defaultdict(set)
        self.resolution_counts: Dict[str, int] = defaultdict(int)
        self._scc_of: Dict[FuncId, int] = {}

    # ------------------------------------------------------------------

    def build(self) -> "CallGraph":
        for fid, fn in self.index.functions.items():
            for site in fn.calls:
                edge = self._resolve(fn, site)
                self.edges[fid].append(edge)
                self.resolution_counts[edge.resolution] += 1
                if edge.callee is not None:
                    self.callers[edge.callee].add(fid)

        self._compute_scc()
        total = sum(self.resolution_counts.values())
        resolved = sum(c for r, c in self.resolution_counts.items()
                       if r.startswith("exact"))
        logger.info("[CallGraph] %d call sites, %d resolved (%.1f%%): %s",
                    total, resolved,
                    100.0 * resolved / total if total else 0.0,
                    dict(self.resolution_counts))
        return self

    def edges_of(self, fid: FuncId) -> List[CallEdge]:
        return self.edges.get(fid, [])

    def same_scc(self, a: FuncId, b: FuncId) -> bool:
        return (a in self._scc_of and b in self._scc_of
                and self._scc_of[a] == self._scc_of[b])

    # ------------------------------------------------------------------

    def _resolve(self, caller: FunctionDef, site: CallSite) -> CallEdge:
        name = site.callee_name

        def edge(callee, resolution, confidence=1.0):
            return CallEdge(caller.fid, site, callee, resolution, confidence)

        if name in self._never:
            return edge(None, "never_inline")
        if name in self.index.macros:
            return edge(None, "macro")

        candidates = list(self.index.by_name.get(name, ()))
        if not candidates:
            return edge(None, "external" if name in self.index.prototypes
                        else "unresolved")

        # Rule 1: a definition in the caller's own file always wins.
        same_file = [c for c in candidates if c.file == caller.fid.file]
        if len(same_file) == 1:
            return edge(same_file[0], "exact_same_file")
        if len(same_file) > 1:
            picked = self._by_arity(same_file, site)
            if picked:
                return edge(picked, "exact_same_file")
            return edge(None, "ambiguous")

        # Rule 2: a static definition in another translation unit is invisible.
        candidates = [c for c in candidates
                      if not self._is_static(c) or self.config.allow_cross_tu_static]
        if not candidates:
            return edge(None, "unresolved")

        # Rule 3: prefer definitions the caller's file can actually see.
        visible = visible_files(self.index, caller.fid.file)
        in_scope = [c for c in candidates if c.file in visible]

        # Rule 3b: the usual C layout is "declared in a shared header, defined
        # in a .c the caller never includes". Treat such a definition as
        # visible when the two files share a header and the name is declared
        # there -- that is exactly what the linker will do (decision D-017).
        if not in_scope and name in self.index.prototypes:
            caller_headers = self.index.include_closure.get(caller.fid.file, set())
            in_scope = [
                c for c in candidates
                if caller_headers & self.index.include_closure.get(c.file, set())
            ]

        pool, resolution = (in_scope, "exact_include") if in_scope \
            else (candidates, "exact_global")

        if len(pool) == 1:
            return edge(pool[0], resolution)

        picked = self._by_arity(pool, site)
        if picked:
            return edge(picked, resolution, 0.9)
        return edge(None, "ambiguous")

    def _by_arity(self, candidates: List[FuncId],
                  site: CallSite) -> Optional[FuncId]:
        matching = []
        for fid in candidates:
            fn = self.index.functions.get(fid)
            if fn is None:
                continue
            if fn.is_vararg:
                if site.arg_count >= len(fn.params):
                    matching.append(fid)
            elif len(fn.params) == site.arg_count:
                matching.append(fid)
        if len(matching) == 1:
            return matching[0]
        return None

    def _is_static(self, fid: FuncId) -> bool:
        fn = self.index.functions.get(fid)
        return bool(fn and fn.is_static)

    # ------------------------------------------------------------------

    def _compute_scc(self) -> None:
        """Strongly connected components, used to keep cycles from expanding."""
        try:
            import networkx as nx
        except ImportError:                               # pragma: no cover
            return
        g = nx.DiGraph()
        g.add_nodes_from(self.index.functions)
        for caller, edges in self.edges.items():
            for e in edges:
                if e.callee is not None:
                    g.add_edge(caller, e.callee)
        for idx, comp in enumerate(nx.strongly_connected_components(g)):
            if len(comp) > 1:
                for node in comp:
                    self._scc_of[node] = idx

    # ------------------------------------------------------------------

    def to_dot(self) -> str:
        lines = ["digraph callgraph {", '  node [shape=box, fontsize=10];']
        for caller, edges in sorted(self.edges.items()):
            for e in edges:
                if e.callee is None:
                    continue
                style = "" if e.resolution.startswith("exact") else " [style=dashed]"
                lines.append(f'  "{caller.name}" -> "{e.callee.name}"{style};')
        lines.append("}")
        return "\n".join(lines)

    def summary(self) -> Dict[str, int]:
        return dict(self.resolution_counts)
