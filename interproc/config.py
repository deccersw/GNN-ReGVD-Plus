"""
Configuration for the interprocedural inlining module.

``max_depth`` is the hyperparameter the whole design turns on: 0 reproduces the
current single-function behaviour byte for byte (regression invariant, D-013),
1 inlines direct callees, 2 also their callees, and so on.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import List

# Names that must never be inlined: they are the very tokens the detector was
# trained on (decision D-004). Sourced from the existing taint analyser when
# importable so the two modules cannot drift apart.
try:  # pragma: no cover - exercised implicitly by the test suite
    from analysis.taint import DANGEROUS_SINKS, INPUT_SOURCES

    _TAINT_NAMES: List[str] = sorted(
        {n for names in DANGEROUS_SINKS.values() for n in names}
        | set(INPUT_SOURCES)
    )
except Exception:  # pragma: no cover - standalone use without the repo
    _TAINT_NAMES = [
        "strcpy", "strcat", "sprintf", "gets", "memcpy", "memmove",
        "printf", "fprintf", "snprintf", "system", "popen", "execl",
        "malloc", "calloc", "realloc", "free", "alloca",
        "scanf", "fscanf", "sscanf", "read", "recv", "fread", "fgets",
    ]

_LIBC_EXTRA = [
    "strncpy", "strncat", "strlen", "strnlen", "strcmp", "strncmp", "strdup",
    "memset", "memcmp", "puts", "putchar", "fputs", "fwrite", "fclose",
    "fflush", "exit", "abort", "assert", "atoi", "atol", "strtol", "strtoul",
    "qsort", "bsearch", "time", "rand", "srand", "close", "write", "send",
    "socket", "bind", "listen", "accept", "mmap", "munmap", "pthread_create",
    "pthread_mutex_lock", "pthread_mutex_unlock", "usleep", "sleep",
]

DEFAULT_NEVER_INLINE: List[str] = sorted(set(_TAINT_NAMES) | set(_LIBC_EXTRA))

DEFAULT_INCLUDE_EXT = [".c", ".h", ".cc", ".cpp", ".cxx", ".c++", ".hpp",
                       ".hh", ".hxx", ".inl"]

DEFAULT_EXCLUDE_DIRS = [
    ".git", ".svn", ".hg", "build", "_build", "cmake-build-debug",
    "cmake-build-release", "out", "dist", "node_modules", "third_party",
    "thirdparty", "vendor", "external", "extern", ".venv", "venv",
    "__pycache__", ".interproc_cache",
]


@dataclass
class InliningConfig:
    # --- the two knobs the user asked for -------------------------------
    enabled: bool = True
    max_depth: int = 2                    # --inline-depth

    # --- strategy -------------------------------------------------------
    strategy: str = "priority"            # priority | dfs | bfs
    parser_backend: str = "auto"          # auto | treesitter | lexer

    # --- budgets --------------------------------------------------------
    max_tokens: int = 398                 # ScannerConfig.block_size - 2
    max_callee_tokens: int = 200
    max_inline_sites: int = 32
    max_expansions_per_callee: int = 2
    bundle_max_tokens: int = 12000
    bundle_max_depth: int = 3

    # --- inlining semantics --------------------------------------------
    rename_on_collision: bool = True
    skip_labeled_callees: bool = True
    allow_cross_tu_static: bool = False
    debug_markers: bool = False
    never_inline: List[str] = field(default_factory=lambda: list(DEFAULT_NEVER_INLINE))

    # --- scoring weights ------------------------------------------------
    w_sink: float = 3.0
    w_mem: float = 2.0
    w_taint: float = 2.5
    w_small: float = 1.0
    w_depth: float = 1.0
    w_trivial: float = 2.0

    # --- discovery ------------------------------------------------------
    include_ext: List[str] = field(default_factory=lambda: list(DEFAULT_INCLUDE_EXT))
    exclude_dirs: List[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_DIRS))
    exclude_globs: List[str] = field(default_factory=list)
    include_dirs: List[str] = field(default_factory=list)
    max_file_bytes: int = 2_000_000
    max_files: int = 20000
    cache_dir: str = ".interproc_cache"
    use_cache: bool = True

    # --- root selection -------------------------------------------------
    root_min_tokens: int = 16
    root_skip_test_files: bool = True
    dedupe_identical_bodies: bool = True
    max_roots: int = 0                    # 0 = unlimited

    # --- tokenizer ------------------------------------------------------
    tokenizer_name: str = "microsoft/graphcodebert-base"
    use_real_tokenizer: bool = True

    def config_hash(self) -> str:
        """Stable digest of the settings that change unit content."""
        relevant = {
            k: v for k, v in asdict(self).items()
            if k not in ("cache_dir", "use_cache", "max_roots", "max_files")
        }
        blob = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, data: dict) -> "InliningConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_scanner_config(cls, sc) -> "InliningConfig":
        """Build from a ScannerConfig, mapping the flat ``inline_*`` fields."""
        data = {}
        if hasattr(sc, "use_inlining"):
            data["enabled"] = getattr(sc, "use_inlining")
        for name in cls.__dataclass_fields__:
            for candidate in (f"inline_{name}", name):
                if hasattr(sc, candidate):
                    data[name] = getattr(sc, candidate)
                    break
        block_size = getattr(sc, "block_size", None)
        if block_size and "max_tokens" not in data:
            data["max_tokens"] = block_size - 2
        return cls.from_dict(data)
