"""
Stage 6: depth-limited, budget-aware expansion of a root function.

Three text transformations happen here -- splicing a callee body in place of a
call, rewriting `return` inside that body, and alpha-renaming colliding locals.
All three are expressed as "replace byte range [a,b) with these segments" and
applied in a single left-to-right pass (decision D-008), so no offset is ever
recomputed after an edit.

Each function body is expanded in its own coordinate space (D-010): a callee is
rendered to a finished segment list before it is inserted into its caller.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import clex
from .budget import Budget, TokenCounter
from .callgraph import CallGraph
from .config import InliningConfig
from .models import (CallSite, FuncId, FunctionDef, InlineRecord, ProjectIndex,
                     Segment, SkipRecord)
from .scoring import CallSiteScorer

logger = logging.getLogger(__name__)


@dataclass
class Repl:
    """One pending edit in a function's coordinate space."""
    start: int
    end: int
    segments: List[Segment]
    priority: float = 0.0
    kind: str = "inline"

    @property
    def is_insert(self) -> bool:
        return self.start == self.end


@dataclass
class ExpansionResult:
    code: str
    segments: List[Segment]
    inlined: List[InlineRecord] = field(default_factory=list)
    skipped: List[SkipRecord] = field(default_factory=list)
    tokens: int = 0
    depth_used: int = 0
    truncated: bool = False
    provenance: List[str] = field(default_factory=list)


@dataclass
class _Ctx:
    """Mutable state shared across one root expansion."""
    root_param_names: Set[str]
    live_names: Set[str]
    occurrences: Dict[FuncId, int] = field(default_factory=dict)
    inlined: List[InlineRecord] = field(default_factory=list)
    skipped: List[SkipRecord] = field(default_factory=list)
    provenance: Set[str] = field(default_factory=set)
    rename_counter: int = 0
    max_depth_reached: int = 0
    hit_budget: bool = False


class Inliner:
    def __init__(self, index: ProjectIndex, graph: CallGraph,
                 config: InliningConfig, counter: TokenCounter):
        self.index = index
        self.graph = graph
        self.config = config
        self.counter = counter
        self.scorer = CallSiteScorer(index, config)

    # ------------------------------------------------------------------

    def expand_root(self, fid: FuncId) -> ExpansionResult:
        fn = self.index.functions[fid]
        cfg = self.config

        budget = Budget(cfg.max_tokens)
        root_tokens = self.counter.count(fn.func_text, key=fn.body_hash)
        budget.spend(root_tokens)

        ctx = _Ctx(
            root_param_names={p.name for p in fn.params},
            live_names=set(fn.local_names) | {fn.name},
        )
        ctx.provenance.add(fn.fid.file)

        if root_tokens >= cfg.max_tokens:
            logger.debug("[Inliner] %s already fills the window (%d tokens); "
                         "no room to inline", fid, root_tokens)
            ctx.hit_budget = True

        segments = self._render(fn, 0, len(fn.func_text), depth=0,
                                path=[fid], budget=budget, ctx=ctx,
                                rename_map={}, return_target=None)

        code = "".join(s.text for s in segments)
        return ExpansionResult(
            code=code,
            segments=segments,
            inlined=ctx.inlined,
            skipped=ctx.skipped,
            tokens=self.counter.count(code),
            depth_used=ctx.max_depth_reached,
            truncated=ctx.hit_budget or root_tokens >= cfg.max_tokens,
            provenance=sorted(ctx.provenance),
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, fn: FunctionDef, start: int, end: int, depth: int,
                path: List[FuncId], budget: Budget, ctx: _Ctx,
                rename_map: Dict[str, str],
                return_target: Optional[str]) -> List[Segment]:
        text = fn.func_text
        mask = clex.code_mask(text)
        repls: List[Repl] = []

        # depth is the nesting level of *this* body; its callees land at
        # depth + 1, so expansion stops once depth reaches max_depth.
        if depth < self.config.max_depth:
            repls.extend(self._inline_repls(fn, text, mask, start, end, depth,
                                            path, budget, ctx, rename_map))
        else:
            self._record_depth_skips(fn, start, end, ctx)

        if return_target is not None or depth > 0:
            repls.extend(self._return_repls(fn, text, mask, start, end,
                                            return_target, depth))

        if rename_map:
            repls.extend(self._rename_repls(fn, text, mask, start, end,
                                            rename_map, depth))

        return self._apply(fn, text, start, end, repls, depth)

    def _inline_repls(self, fn: FunctionDef, text: str, mask: bytearray,
                      start: int, end: int, depth: int, path: List[FuncId],
                      budget: Budget, ctx: _Ctx,
                      rename_map: Dict[str, str]) -> List[Repl]:
        cfg = self.config
        self._record_unresolved_skips(fn, start, end, ctx)
        candidates = self._candidates(fn, start, end, depth, ctx)
        repls: List[Repl] = []
        inlined_here = 0

        for score, site, callee in candidates:
            if inlined_here >= cfg.max_inline_sites:
                ctx.skipped.append(SkipRecord(site.callee_name, "max_sites",
                                              site.line))
                continue

            reason = self._veto(callee, path, ctx)
            if reason:
                ctx.skipped.append(SkipRecord(site.callee_name, reason,
                                              site.line, str(callee.fid)))
                continue

            built = self._try_inline(fn, text, mask, site, callee, score,
                                     depth, path, budget, ctx, rename_map)
            if built is None:
                continue
            repls.extend(built)
            inlined_here += 1

        return repls

    def _record_unresolved_skips(self, fn: FunctionDef, start: int, end: int,
                                 ctx: _Ctx) -> None:
        """Surface graph-level refusals (never_inline, ambiguous, external).

        Without this the unit's ``skipped`` list would only explain the
        decisions the inliner itself made, and a reader could not tell whether
        a call was left alone deliberately or simply never considered.
        """
        for edge in self.graph.edges_of(fn.fid):
            if edge.callee is not None:
                continue
            site = edge.site
            if start <= site.start and site.end <= end:
                ctx.skipped.append(SkipRecord(site.callee_name,
                                              edge.resolution, site.line))

    def _record_depth_skips(self, fn: FunctionDef, start: int, end: int,
                            ctx: _Ctx) -> None:
        for edge in self.graph.edges_of(fn.fid):
            if edge.callee is None:
                continue
            site = edge.site
            if start <= site.start and site.end <= end:
                ctx.skipped.append(SkipRecord(site.callee_name, "depth",
                                              site.line, str(edge.callee)))

    def _candidates(self, fn: FunctionDef, start: int, end: int, depth: int,
                    ctx: _Ctx) -> List[Tuple[float, CallSite, FunctionDef]]:
        out: List[Tuple[float, CallSite, FunctionDef]] = []
        for edge in self.graph.edges_of(fn.fid):
            site = edge.site
            if not (start <= site.start and site.end <= end):
                continue
            if edge.callee is None:
                continue
            callee = self.index.functions.get(edge.callee)
            if callee is None:
                continue
            score = self.scorer.score(site, callee, depth, ctx.root_param_names)
            out.append((score, site, callee))

        strategy = self.config.strategy
        if strategy == "dfs":
            out.sort(key=lambda t: (t[1].start, str(t[2].fid)))
        elif strategy == "bfs":
            # shallow-first: same ranking as priority with depth penalised hard
            out.sort(key=lambda t: (-(t[0] - 2.0 * self.config.w_depth * depth),
                                    t[1].start, str(t[2].fid)))
        else:
            out.sort(key=lambda t: (-t[0], t[1].start, str(t[2].fid)))
        return out

    def _veto(self, callee: FunctionDef, path: Sequence[FuncId],
              ctx: _Ctx) -> Optional[str]:
        """Structural reasons never to inline this callee here."""
        cfg = self.config
        if callee.fid in path:
            return "recursion"
        if self.graph.same_scc(path[-1], callee.fid):
            return "recursion_scc"
        if callee.is_vararg:
            return "vararg"
        if callee.is_virtual:
            return "virtual"
        if "has_asm" in callee.flags or "has_setjmp" in callee.flags:
            return "unsafe_construct"
        if cfg.skip_labeled_callees and (
                "has_label" in callee.flags or "has_goto" in callee.flags):
            return "labels"
        if "unbalanced_preproc" in callee.flags:
            return "preproc"
        if "truncated_body" in callee.flags:
            return "truncated_body"
        if ctx.occurrences.get(callee.fid, 0) >= cfg.max_expansions_per_callee:
            return "repeat"
        callee_tokens = self.counter.count(callee.func_text, key=callee.body_hash)
        if callee_tokens > cfg.max_callee_tokens:
            return "callee_too_big"
        return None

    def _try_inline(self, fn: FunctionDef, text: str, mask: bytearray,
                    site: CallSite, callee: FunctionDef, score: float,
                    depth: int, path: List[FuncId], budget: Budget,
                    ctx: _Ctx, caller_renames: Dict[str, str]) -> Optional[List[Repl]]:
        """Render the callee, then commit only if it fits in the budget."""
        snap_occ = dict(ctx.occurrences)
        snap_live = set(ctx.live_names)
        snap_prov = set(ctx.provenance)
        snap_in, snap_skip = len(ctx.inlined), len(ctx.skipped)
        snap_counter, snap_depth = ctx.rename_counter, ctx.max_depth_reached

        rename_map = self._rename_map(callee, ctx)
        ret_name = ""
        if site.result_used:
            ret_name = self._unique_name(f"__r_{callee.name}", ctx)

        ctx.occurrences[callee.fid] = ctx.occurrences.get(callee.fid, 0) + 1
        ctx.provenance.add(callee.fid.file)
        ctx.max_depth_reached = max(ctx.max_depth_reached, depth + 1)

        child = Budget(budget.remaining)
        body_segments = self._render(
            callee,
            callee.body_start + 1, callee.body_end - 1,
            depth=depth + 1,
            path=path + [callee.fid],
            budget=child,
            ctx=ctx,
            rename_map=rename_map,
            return_target=ret_name or None,
        )

        block = self._build_block(fn, site, callee, body_segments, rename_map,
                                  ret_name, depth, caller_renames)
        cost = self.counter.count("".join(s.text for s in block))

        if not budget.can_afford(cost):
            ctx.occurrences = snap_occ
            ctx.live_names = snap_live
            ctx.provenance = snap_prov
            ctx.rename_counter = snap_counter
            ctx.max_depth_reached = snap_depth
            del ctx.inlined[snap_in:]
            del ctx.skipped[snap_skip:]
            ctx.hit_budget = True
            ctx.skipped.append(SkipRecord(site.callee_name, "budget", site.line,
                                          f"needs {cost}, left {budget.remaining}"))
            return None

        budget.spend(cost)
        ctx.inlined.append(InlineRecord(
            callee=str(callee.fid), callee_name=callee.name, depth=depth + 1,
            tokens_added=cost, score=round(score, 3), line=site.line))

        if not site.result_used:
            return [Repl(site.start, site.end, block, score, "inline")]

        stmt_start = self._statement_start(text, mask, site.start)
        return [
            Repl(stmt_start, stmt_start, block, score, "inline"),
            Repl(site.start, site.end,
                 [Segment(ret_name, fn.fid.file, site.line, fn.name, depth,
                          "ret_use")],
                 score, "inline"),
        ]

    # ------------------------------------------------------------------

    def _build_block(self, caller: FunctionDef, site: CallSite,
                     callee: FunctionDef, body_segments: List[Segment],
                     rename_map: Dict[str, str], ret_name: str, depth: int,
                     caller_renames: Dict[str, str]) -> List[Segment]:
        """Assemble `T ret; { bindings; body }` for one inlined call."""
        origin = callee.fid.file
        out: List[Segment] = []

        def seg(text: str, role: str, line: int = 0) -> Segment:
            return Segment(text, origin, line or callee.start_line,
                           callee.name, depth + 1, role)

        if self.config.debug_markers:
            out.append(seg(f"/* inline {callee.name} d{depth + 1} */\n", "marker"))

        if ret_name:
            ret_type = callee.return_type_for_temp()
            out.append(seg(f"{ret_type} {ret_name};\n", "ret_decl"))

        out.append(seg("{\n", "body"))

        for i, param in enumerate(callee.params):
            if i >= len(site.arg_texts):
                break
            arg = site.arg_texts[i]
            if caller_renames:
                arg = clex.substitute_identifiers(arg, caller_renames)
            pname = rename_map.get(param.name, param.name)
            decl = _declarator(param.type, pname)
            out.append(seg(f"{decl} = {arg};\n", "param_bind", site.line))

        out.extend(body_segments)
        out.append(seg("\n}\n", "body"))
        return out

    def _rename_map(self, callee: FunctionDef, ctx: _Ctx) -> Dict[str, str]:
        """Rename only names that actually collide (decision D-011)."""
        if not self.config.rename_on_collision:
            return {}
        collisions = sorted(callee.local_names & ctx.live_names)
        mapping: Dict[str, str] = {}
        for name in collisions:
            ctx.rename_counter += 1
            new = f"{name}_i{ctx.rename_counter}"
            while new in ctx.live_names:
                ctx.rename_counter += 1
                new = f"{name}_i{ctx.rename_counter}"
            mapping[name] = new
            ctx.live_names.add(new)
        ctx.live_names |= (callee.local_names - set(mapping))
        return mapping

    def _unique_name(self, base: str, ctx: _Ctx) -> str:
        name = base
        n = 1
        while name in ctx.live_names:
            n += 1
            name = f"{base}{n}"
        ctx.live_names.add(name)
        return name

    def _return_repls(self, fn: FunctionDef, text: str, mask: bytearray,
                      start: int, end: int, target: Optional[str],
                      depth: int) -> List[Repl]:
        """`return expr;` -> `tmp = expr;` (or just `expr;` when unused)."""
        out: List[Repl] = []
        for kw_start, kw_end, has_expr in clex.find_return_statements(text, mask):
            if not (start <= kw_start and kw_end <= end):
                continue
            replacement = f"{target} =" if (target and has_expr) else ""
            out.append(Repl(
                kw_start, kw_end,
                [Segment(replacement, fn.fid.file,
                         fn.start_line + text.count('\n', 0, kw_start),
                         fn.name, depth, "ret_assign")],
                priority=1e6, kind="return"))
        return out

    def _rename_repls(self, fn: FunctionDef, text: str, mask: bytearray,
                      start: int, end: int, mapping: Dict[str, str],
                      depth: int) -> List[Repl]:
        out: List[Repl] = []
        for s, e, name in clex.iter_identifiers(text, mask, start, end):
            new = mapping.get(name)
            if new is None:
                continue
            out.append(Repl(s, e,
                            [Segment(new, fn.fid.file,
                                     fn.start_line + text.count('\n', 0, s),
                                     fn.name, depth, "rename")],
                            priority=5e5, kind="rename"))
        return out

    @staticmethod
    def _statement_start(text: str, mask: bytearray, pos: int) -> int:
        j = pos - 1
        while j >= 0:
            if mask[j] == clex.CODE and text[j] in ';{}':
                break
            j -= 1
        j += 1
        while j < pos and text[j].isspace():
            j += 1
        return j

    # ------------------------------------------------------------------

    def _apply(self, fn: FunctionDef, text: str, start: int, end: int,
               repls: List[Repl], depth: int) -> List[Segment]:
        """Single left-to-right pass; nested ranges resolved by priority (D-009)."""
        accepted: List[Repl] = []
        for repl in sorted(repls, key=lambda r: (-r.priority, r.start)):
            if any(_overlaps(repl, other) for other in accepted):
                continue
            accepted.append(repl)
        accepted.sort(key=lambda r: (r.start, 0 if r.is_insert else 1))

        segments: List[Segment] = []
        pos = start
        for repl in accepted:
            if repl.start > pos:
                segments.append(self._verbatim(fn, text, pos, repl.start, depth))
            segments.extend(repl.segments)
            pos = max(pos, repl.end)
        if pos < end:
            segments.append(self._verbatim(fn, text, pos, end, depth))
        return [s for s in segments if s.text]

    @staticmethod
    def _verbatim(fn: FunctionDef, text: str, a: int, b: int,
                  depth: int) -> Segment:
        return Segment(
            text=text[a:b],
            origin_file=fn.fid.file,
            origin_line=fn.start_line + text.count('\n', 0, a),
            origin_func=fn.name,
            depth=depth,
            role="root" if depth == 0 else "body",
        )


def _overlaps(a: Repl, b: Repl) -> bool:
    if a.is_insert or b.is_insert:
        return False
    return a.start < b.end and b.start < a.end


def _declarator(type_text: str, name: str) -> str:
    """Build `T name` handling array and pointer spellings sensibly."""
    t = " ".join((type_text or "").split()) or "int"
    suffix = ""
    while t.endswith(']'):
        open_idx = t.rfind('[')
        if open_idx < 0:
            break
        suffix = t[open_idx:] + suffix
        t = t[:open_idx].strip()
    if suffix:
        # an array parameter decays to a pointer; keep it simple and valid
        return f"{t} *{name}"
    if t.endswith('*'):
        return f"{t}{name}"
    return f"{t} {name}"
