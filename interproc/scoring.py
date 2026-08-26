"""
Call-site prioritisation (decision D-003).

With a 398-token window, the order in which call sites are inlined decides
what the detector actually gets to see. Ranking by security relevance rather
than by source order is the difference between spending the budget on a
logging helper and spending it on the function that calls strcpy.
"""

import logging
import re
from typing import Dict, Set

from . import clex
from .config import InliningConfig
from .models import CallSite, FunctionDef, ProjectIndex

logger = logging.getLogger(__name__)

try:
    from analysis.taint import DANGEROUS_SINKS, INPUT_SOURCES
    SINK_NAMES: Set[str] = {n for names in DANGEROUS_SINKS.values() for n in names}
    SOURCE_NAMES: Set[str] = set(INPUT_SOURCES)
except Exception:                                          # pragma: no cover
    SINK_NAMES = {"strcpy", "strcat", "sprintf", "gets", "memcpy", "memmove",
                  "system", "popen", "printf", "fprintf"}
    SOURCE_NAMES = {"argv", "getenv", "fgets", "scanf", "read", "recv"}

MEMORY_NAMES = {"malloc", "calloc", "realloc", "free", "alloca", "memset",
                "memcpy", "memmove", "new", "delete"}

_TRIVIAL_RETURN = re.compile(r"^\s*\{\s*return\s+[^;{}]*;\s*\}\s*$", re.S)


class CallSiteScorer:
    def __init__(self, index: ProjectIndex, config: InliningConfig):
        self.index = index
        self.config = config
        self._sink_cache: Dict[str, bool] = {}
        self._mem_cache: Dict[str, bool] = {}

    def score(self, site: CallSite, callee: FunctionDef, depth: int,
              root_names: Set[str]) -> float:
        cfg = self.config
        value = 0.0

        if self._has_names(callee, SINK_NAMES, self._sink_cache):
            value += cfg.w_sink
        if self._has_names(callee, MEMORY_NAMES, self._mem_cache):
            value += cfg.w_mem
        if self._args_tainted(site, root_names):
            value += cfg.w_taint

        tokens = max(1, len(callee.func_text.split()))
        value += cfg.w_small * (1.0 - min(1.0, tokens / max(1, cfg.max_callee_tokens)))
        value -= cfg.w_depth * depth

        if self.is_trivial(callee):
            value -= cfg.w_trivial

        return value

    # ------------------------------------------------------------------

    def _has_names(self, callee: FunctionDef, wanted: Set[str],
                   cache: Dict[str, bool]) -> bool:
        key = str(callee.fid)
        hit = cache.get(key)
        if hit is None:
            body = callee.body_inner
            mask = clex.code_mask(body)
            hit = clex.contains_word(body, mask, wanted)
            cache[key] = hit
        return hit

    @staticmethod
    def _args_tainted(site: CallSite, root_names: Set[str]) -> bool:
        """Does any argument mention a caller parameter or an input source?"""
        for arg in site.arg_texts:
            mask = clex.code_mask(arg)
            for _s, _e, name in clex.iter_identifiers(arg, mask):
                if name in root_names or name in SOURCE_NAMES:
                    return True
        return False

    @staticmethod
    def is_trivial(fn: FunctionDef) -> bool:
        """Accessor-shaped: a single return of an expression, no calls."""
        if fn.calls:
            return False
        return bool(_TRIVIAL_RETURN.match(fn.func_text[fn.body_start:fn.body_end]))


def has_sink(fn: FunctionDef) -> bool:
    body = fn.body_inner
    return clex.contains_word(body, clex.code_mask(body), SINK_NAMES)


def has_memory_op(fn: FunctionDef) -> bool:
    body = fn.body_inner
    if clex.contains_word(body, clex.code_mask(body), MEMORY_NAMES):
        return True
    return '[' in body and '=' in body
