"""
Stage 1-2: walk a project, read sources, parse them into FileFacts.

Everything here is defensive: a project is arbitrary third-party code, so a
symlink loop, a 400 MB generated file or an invalid encoding must degrade into
a skipped file with a log line, never into an exception.
"""

import fnmatch
import hashlib
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import InliningConfig
from .models import FileFacts
from .parsers.base import select_backend

logger = logging.getLogger(__name__)

CPP_EXT = {".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx"}
TEST_MARKERS = ("test", "tests", "testing", "gtest", "unittest", "benchmark")


@dataclass
class DiscoveryStats:
    files_seen: int = 0
    files_parsed: int = 0
    skipped_size: int = 0
    skipped_binary: int = 0
    skipped_glob: int = 0
    read_errors: int = 0
    cache_hits: int = 0
    functions: int = 0


class ProjectDiscovery:
    def __init__(self, config: InliningConfig):
        self.config = config
        self.backend = select_backend(config.parser_backend)
        self.stats = DiscoveryStats()
        logger.info("[Discovery] parser backend: %s v%s",
                    self.backend.name, self.backend.version)

    # ------------------------------------------------------------------

    def collect(self, root: str) -> List[FileFacts]:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise NotADirectoryError(f"not a directory: {root}")

        cache = _Cache(root, self.config, self.backend)
        facts: List[FileFacts] = []

        for path in self._walk(root):
            self.stats.files_seen += 1
            rel = os.path.relpath(path, root)
            read = self._read(path)
            if read is None:
                continue
            text, digest = read

            cached = cache.get(rel, digest)
            if cached is not None:
                self.stats.cache_hits += 1
                facts.append(cached)
                self.stats.files_parsed += 1
                self.stats.functions += len(cached.functions)
                continue

            language = "cpp" if os.path.splitext(path)[1].lower() in CPP_EXT else "c"
            try:
                ff = self.backend.parse(path, rel, text, language)
            except Exception as exc:                     # pragma: no cover
                logger.warning("[Discovery] parse failed for %s: %s", rel, exc)
                self.stats.read_errors += 1
                continue

            cache.put(rel, digest, ff)
            facts.append(ff)
            self.stats.files_parsed += 1
            self.stats.functions += len(ff.functions)

            if len(facts) >= self.config.max_files:
                logger.warning("[Discovery] max_files=%d reached, stopping",
                               self.config.max_files)
                break

        cache.flush()
        logger.info("[Discovery] %d files parsed, %d functions found "
                    "(%d cache hits)", self.stats.files_parsed,
                    self.stats.functions, self.stats.cache_hits)
        return facts

    # ------------------------------------------------------------------

    def _walk(self, root: str) -> List[str]:
        exts = {e.lower() for e in self.config.include_ext}
        excluded = set(self.config.exclude_dirs)
        seen_dirs = set()
        out: List[str] = []

        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            real = os.path.realpath(dirpath)
            if real in seen_dirs:
                continue
            seen_dirs.add(real)
            dirnames[:] = sorted(d for d in dirnames if d not in excluded)

            for fname in sorted(filenames):
                if os.path.splitext(fname)[1].lower() not in exts:
                    continue
                path = os.path.join(dirpath, fname)
                rel = os.path.relpath(path, root)
                if any(fnmatch.fnmatch(rel, g) for g in self.config.exclude_globs):
                    self.stats.skipped_glob += 1
                    continue
                out.append(path)
        return out

    def _read(self, path: str) -> Optional[Tuple[str, str]]:
        try:
            size = os.path.getsize(path)
            if size > self.config.max_file_bytes:
                self.stats.skipped_size += 1
                logger.debug("[Discovery] skip (size %d): %s", size, path)
                return None
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError as exc:
            self.stats.read_errors += 1
            logger.debug("[Discovery] unreadable %s: %s", path, exc)
            return None

        if b"\x00" in raw[:8192]:
            self.stats.skipped_binary += 1
            return None

        # CRLF must die here: every offset downstream is a byte offset into
        # this string, and a stray \r would shift them all.
        text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
        digest = hashlib.sha1(raw).hexdigest()[:16]
        return text, digest


def is_test_file(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").lower().split("/")
    stem = os.path.splitext(parts[-1])[0]
    if any(p in TEST_MARKERS for p in parts[:-1]):
        return True
    return stem.startswith("test_") or stem.endswith("_test") \
        or stem.endswith("_tests") or stem.startswith("bench")


class _Cache:
    """On-disk parse cache. Key includes the backend version (decision D-016)."""

    def __init__(self, root: str, config: InliningConfig, backend):
        self.enabled = config.use_cache
        self.path = os.path.join(root, config.cache_dir, "facts.pkl")
        self.key_suffix = f"{backend.name}:{backend.version}:{config.config_hash()}"
        self.data: Dict[str, Tuple[str, FileFacts]] = {}
        self.dirty = False
        if not self.enabled:
            return
        try:
            with open(self.path, "rb") as fh:
                stored = pickle.load(fh)
            if stored.get("suffix") == self.key_suffix:
                self.data = stored.get("data", {})
        except Exception:
            self.data = {}

    def get(self, rel: str, digest: str) -> Optional[FileFacts]:
        if not self.enabled:
            return None
        hit = self.data.get(rel)
        if hit and hit[0] == digest:
            return hit[1]
        return None

    def put(self, rel: str, digest: str, facts: FileFacts) -> None:
        if not self.enabled:
            return
        self.data[rel] = (digest, facts)
        self.dirty = True

    def flush(self) -> None:
        if not (self.enabled and self.dirty):
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "wb") as fh:
                pickle.dump({"suffix": self.key_suffix, "data": self.data}, fh)
        except Exception as exc:                          # pragma: no cover
            logger.debug("[Discovery] cache write failed: %s", exc)
