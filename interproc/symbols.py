"""
Stage 3: fold per-file facts into a project-wide index.

The include closure computed here is what makes cross-file resolution more
than "same name wins": a definition is only a candidate for a call site if the
caller's file can actually see it.
"""

import logging
import os
from collections import deque
from typing import Dict, List, Set

from .config import InliningConfig
from .models import FileFacts, FuncId, ProjectIndex

logger = logging.getLogger(__name__)


def build_index(root: str, facts: List[FileFacts],
                config: InliningConfig) -> ProjectIndex:
    index = ProjectIndex(root=root)

    for ff in facts:
        index.files[ff.rel_path] = ff
        for fn in ff.functions:
            if fn.fid in index.functions:
                continue
            index.functions[fn.fid] = fn
            index.by_name.setdefault(fn.name, []).append(fn.fid)
        for t in ff.types:
            index.types.setdefault(t.name, t)
        for g in ff.globals:
            index.globals.setdefault(g.name, g)
        for m in ff.macros:
            index.macros.add(m.name)
        index.prototypes |= ff.prototypes

    index.include_closure = _include_closure(index, config)

    logger.info("[Index] %d files, %d functions, %d distinct names, "
                "%d macros, %d types",
                len(index.files), len(index.functions), len(index.by_name),
                len(index.macros), len(index.types))
    return index


def _include_closure(index: ProjectIndex,
                     config: InliningConfig) -> Dict[str, Set[str]]:
    """Transitive set of project files each file can see through #include."""
    direct: Dict[str, Set[str]] = {}
    by_basename: Dict[str, List[str]] = {}
    for rel in index.files:
        by_basename.setdefault(os.path.basename(rel), []).append(rel)

    for rel, ff in index.files.items():
        targets: Set[str] = set()
        base_dir = os.path.dirname(rel)
        for inc in ff.includes:
            if inc.is_system:
                continue
            resolved = _resolve_include(inc.path, base_dir, index,
                                        by_basename, config)
            if resolved:
                targets.add(resolved)
        direct[rel] = targets

    closure: Dict[str, Set[str]] = {}
    for rel in index.files:
        seen: Set[str] = set()
        queue = deque(direct.get(rel, ()))
        while queue:
            nxt = queue.popleft()
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.extend(direct.get(nxt, ()))
        closure[rel] = seen
    return closure


def _resolve_include(inc_path: str, base_dir: str, index: ProjectIndex,
                     by_basename: Dict[str, List[str]],
                     config: InliningConfig):
    candidate = os.path.normpath(os.path.join(base_dir, inc_path))
    if candidate in index.files:
        return candidate

    for inc_dir in config.include_dirs:
        candidate = os.path.normpath(os.path.join(inc_dir, inc_path))
        if candidate in index.files:
            return candidate

    if inc_path in index.files:
        return inc_path

    matches = by_basename.get(os.path.basename(inc_path), [])
    if len(matches) == 1:
        return matches[0]
    # Several files share the basename: prefer one whose tail matches the
    # spelled include path, otherwise give up rather than guess.
    tail_matches = [m for m in matches if m.endswith(inc_path)]
    if len(tail_matches) == 1:
        return tail_matches[0]
    return None


def visible_files(index: ProjectIndex, rel_path: str) -> Set[str]:
    return {rel_path} | index.include_closure.get(rel_path, set())


def function_ids_named(index: ProjectIndex, name: str) -> List[FuncId]:
    return index.by_name.get(name, [])
