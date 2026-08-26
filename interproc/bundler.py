"""
Stage 7: the compilable view of a unit (decision D-001).

No inlining happens here. The callees are emitted as whole definitions,
leaves first, preceded by the includes, types and globals they need. That is
what Module 3 (LLM harness generation) and Module 4 (sandbox) consume, and it
is why early `return`, goto and varargs are non-problems for them.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from . import clex
from .budget import TokenCounter
from .callgraph import CallGraph
from .config import InliningConfig
from .models import FuncId, FunctionDef, ProjectIndex

logger = logging.getLogger(__name__)


@dataclass
class BundleResult:
    code: str
    functions: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    tokens: int = 0
    dropped: List[str] = field(default_factory=list)


class Bundler:
    def __init__(self, index: ProjectIndex, graph: CallGraph,
                 config: InliningConfig, counter: TokenCounter):
        self.index = index
        self.graph = graph
        self.config = config
        self.counter = counter

    def build(self, root_fid: FuncId) -> BundleResult:
        root = self.index.functions[root_fid]
        order = self._collect(root_fid)

        chosen, dropped = self._fit_budget(order, root)
        text = self._emit(root, chosen)

        return BundleResult(
            code=text,
            functions=[str(f.fid) for f in chosen] + [str(root.fid)],
            files=sorted({f.fid.file for f in chosen} | {root.fid.file}),
            tokens=self.counter.count(text),
            dropped=dropped,
        )

    # ------------------------------------------------------------------

    def _collect(self, root_fid: FuncId) -> List[FunctionDef]:
        """Reachable callees in leaves-first order, cycles broken by visit set."""
        ordered: List[FunctionDef] = []
        seen: Set[FuncId] = {root_fid}

        def walk(fid: FuncId, depth: int) -> None:
            if depth > self.config.bundle_max_depth:
                return
            for edge in self.graph.edges_of(fid):
                callee = edge.callee
                if callee is None or callee in seen:
                    continue
                fn = self.index.functions.get(callee)
                if fn is None:
                    continue
                seen.add(callee)
                walk(callee, depth + 1)
                ordered.append(fn)

        walk(root_fid, 1)
        return ordered

    def _fit_budget(self, callees: List[FunctionDef],
                    root: FunctionDef):
        limit = self.config.bundle_max_tokens
        used = self.counter.count(root.func_text, key=root.body_hash)
        chosen: List[FunctionDef] = []
        dropped: List[str] = []

        for fn in callees:
            cost = self.counter.count(fn.func_text, key=fn.body_hash)
            if used + cost > limit:
                dropped.append(str(fn.fid))
                continue
            used += cost
            chosen.append(fn)
        return chosen, dropped

    # ------------------------------------------------------------------

    def _emit(self, root: FunctionDef, callees: List[FunctionDef]) -> str:
        files = [root.fid.file] + [f.fid.file for f in callees]
        parts: List[str] = []

        includes = self._system_includes(files)
        if includes:
            parts.append("\n".join(f"#include <{h}>" for h in includes))

        bodies = [root] + callees
        used_names = self._identifiers_in(bodies)

        types = self._needed_types(used_names)
        if types:
            parts.append("\n".join(types))

        globals_ = self._needed_globals(used_names,
                                        {f.name for f in bodies})
        if globals_:
            parts.append("\n".join(globals_))

        for fn in callees:
            parts.append(fn.func_text.rstrip())
        parts.append(root.func_text.rstrip())

        return "\n\n".join(p for p in parts if p.strip()) + "\n"

    def _system_includes(self, files: List[str]) -> List[str]:
        headers: Set[str] = set()
        for rel in dict.fromkeys(files):
            ff = self.index.files.get(rel)
            if ff is None:
                continue
            for inc in ff.includes:
                if inc.is_system:
                    headers.add(inc.path)
            for dep in self.index.include_closure.get(rel, ()):  # local headers
                dep_ff = self.index.files.get(dep)
                if dep_ff is None:
                    continue
                for inc in dep_ff.includes:
                    if inc.is_system:
                        headers.add(inc.path)
        return sorted(headers)

    @staticmethod
    def _identifiers_in(functions: List[FunctionDef]) -> Set[str]:
        names: Set[str] = set()
        for fn in functions:
            text = fn.func_text
            mask = clex.code_mask(text)
            for _s, _e, name in clex.iter_identifiers(text, mask):
                names.add(name)
        return names

    def _needed_types(self, used: Set[str]) -> List[str]:
        """Transitive closure of referenced project types, dependency ordered."""
        wanted: List[str] = []
        seen: Set[str] = set()
        frontier = [n for n in sorted(used) if n in self.index.types]

        while frontier:
            name = frontier.pop(0)
            if name in seen:
                continue
            seen.add(name)
            tdef = self.index.types[name]
            mask = clex.code_mask(tdef.text)
            for _s, _e, ident in clex.iter_identifiers(tdef.text, mask):
                if ident in self.index.types and ident not in seen:
                    frontier.append(ident)
            wanted.append(name)

        # dependants were appended after their dependencies were queued, so
        # emitting in reverse discovery order puts the leaves first
        return [self.index.types[n].text.rstrip() for n in reversed(wanted)]

    def _needed_globals(self, used: Set[str],
                        function_names: Set[str]) -> List[str]:
        out: List[str] = []
        for name in sorted(used):
            if name in function_names:
                continue
            g = self.index.globals.get(name)
            if g is not None:
                out.append(g.text.rstrip())
        return out
