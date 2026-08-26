"""
Data model for the interprocedural inlining module.

Coordinate convention, relied upon by the whole module: every byte offset
stored on a CallSite or a FunctionDef sub-range is **relative to
FunctionDef.func_text**, not to the file. Each function body is therefore its
own coordinate space (decision D-010), which is what keeps recursive expansion
from shifting offsets across nesting levels.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass(frozen=True, order=True)
class FuncId:
    """Stable identity of a function within a project."""
    file: str          # project-relative path
    name: str
    start_line: int

    def __str__(self) -> str:
        return f"{self.file}:{self.name}:{self.start_line}"


@dataclass
class Param:
    type: str
    name: str


@dataclass
class CallSite:
    callee_name: str
    arg_texts: List[str]
    start: int                 # relative to owning FunctionDef.func_text
    end: int
    line: int                  # absolute line in the file
    kind: str = "direct"       # direct | method | macro | unknown
    receiver: str = ""
    result_used: bool = True

    @property
    def arg_count(self) -> int:
        return len(self.arg_texts)


@dataclass
class FunctionDef:
    fid: FuncId
    name: str
    qualified_name: str
    return_type: str
    params: List[Param]
    func_text: str             # signature + body, verbatim from the file
    body_start: int            # offset of '{' within func_text
    body_end: int              # offset just past '}' within func_text
    start_line: int
    end_line: int
    is_static: bool = False
    is_vararg: bool = False
    is_virtual: bool = False
    is_template: bool = False
    calls: List[CallSite] = field(default_factory=list)
    flags: Set[str] = field(default_factory=set)
    local_names: Set[str] = field(default_factory=set)
    _token_count: Optional[int] = None

    @property
    def body_inner(self) -> str:
        """Body without the enclosing braces."""
        return self.func_text[self.body_start + 1:self.body_end - 1]

    @property
    def body_inner_offset(self) -> int:
        return self.body_start + 1

    @property
    def signature(self) -> str:
        return self.func_text[:self.body_start].strip()

    @property
    def body_hash(self) -> str:
        return hashlib.sha1(
            " ".join(self.func_text.split()).encode("utf-8", "replace")
        ).hexdigest()[:16]

    def return_type_for_temp(self) -> str:
        """Return type stripped of storage/linkage words, for a temp variable."""
        drop = {"static", "inline", "extern", "virtual", "constexpr",
                "__inline", "__inline__", "_Noreturn", "friend", "explicit"}
        words = [w for w in self.return_type.split() if w not in drop]
        return " ".join(words).strip() or "int"

    def is_inlinable(self) -> bool:
        """Structural veto, independent of budget and configuration."""
        return not (self.is_vararg or self.is_virtual
                    or {"has_asm", "has_setjmp"} & self.flags)


@dataclass
class TypeDef:
    name: str
    kind: str                  # struct | union | enum | typedef
    text: str
    line: int


@dataclass
class GlobalDef:
    name: str
    text: str
    line: int


@dataclass
class MacroDef:
    name: str
    is_function_like: bool


@dataclass
class Include:
    path: str
    is_system: bool


@dataclass
class FileFacts:
    path: str                  # absolute
    rel_path: str              # project-relative
    language: str              # c | cpp
    text: str
    sha1: str
    functions: List[FunctionDef] = field(default_factory=list)
    types: List[TypeDef] = field(default_factory=list)
    globals: List[GlobalDef] = field(default_factory=list)
    macros: List[MacroDef] = field(default_factory=list)
    includes: List[Include] = field(default_factory=list)
    prototypes: Set[str] = field(default_factory=set)
    parse_errors: List[str] = field(default_factory=list)


@dataclass
class CallEdge:
    caller: FuncId
    site: CallSite
    callee: Optional[FuncId]
    resolution: str            # exact_same_file | exact_include | exact_global |
                               # ambiguous | external | macro | never_inline |
                               # unresolved
    confidence: float = 1.0

    @property
    def resolved(self) -> bool:
        return self.callee is not None


@dataclass
class Segment:
    """A piece of generated text together with where it came from."""
    text: str
    origin_file: str = ""
    origin_line: int = 0
    origin_func: str = ""
    depth: int = 0
    role: str = "root"         # root | param_bind | body | ret_decl | marker


@dataclass
class InlineRecord:
    callee: str                # str(FuncId)
    callee_name: str
    depth: int
    tokens_added: int
    score: float
    line: int                  # call-site line in the caller


@dataclass
class SkipRecord:
    callee_name: str
    reason: str
    line: int = 0
    detail: str = ""


@dataclass
class AnalysisUnit:
    unit_id: str
    root: FuncId
    code_for_gnn: str
    code_for_sandbox: str
    segments: List[Segment] = field(default_factory=list)
    inlined: List[InlineRecord] = field(default_factory=list)
    skipped: List[SkipRecord] = field(default_factory=list)
    depth_used: int = 0
    tokens_gnn: int = 0
    tokens_sandbox: int = 0
    truncated: bool = False
    provenance_files: List[str] = field(default_factory=list)
    sandbox_functions: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def source_map(self) -> List[Dict]:
        """Expanded-line -> origin mapping, derived from the segment list."""
        out: List[Dict] = []
        line = 1
        for seg in self.segments:
            span = seg.text.count('\n')
            out.append({
                "gnn_line_start": line,
                "gnn_line_end": line + span,
                "origin_file": seg.origin_file,
                "origin_line": seg.origin_line,
                "origin_func": seg.origin_func,
                "depth": seg.depth,
                "role": seg.role,
            })
            line += span
        return out

    def to_jsonl_record(self, idx: int = 0, label: int = -1) -> Dict:
        """Serialise in a shape the existing dataset tooling already accepts."""
        return {
            "idx": idx,
            "func": self.code_for_gnn,
            "target": label,
            "unit": {
                "unit_id": self.unit_id,
                "root": str(self.root),
                "file": self.root.file,
                "function": self.root.name,
                "start_line": self.root.start_line,
                "code_for_sandbox": self.code_for_sandbox,
                "depth_used": self.depth_used,
                "tokens_gnn": self.tokens_gnn,
                "tokens_sandbox": self.tokens_sandbox,
                "truncated": self.truncated,
                "provenance_files": self.provenance_files,
                "sandbox_functions": self.sandbox_functions,
                "inlined": [vars(r) for r in self.inlined],
                "skipped": [vars(r) for r in self.skipped],
                "stats": self.stats,
            },
        }


@dataclass
class ProjectIndex:
    """Everything known about a project after parsing (stages 1-3)."""
    root: str
    files: Dict[str, FileFacts] = field(default_factory=dict)      # rel_path -> facts
    functions: Dict[FuncId, FunctionDef] = field(default_factory=dict)
    by_name: Dict[str, List[FuncId]] = field(default_factory=dict)
    include_closure: Dict[str, Set[str]] = field(default_factory=dict)
    macros: Set[str] = field(default_factory=set)
    types: Dict[str, TypeDef] = field(default_factory=dict)
    globals: Dict[str, GlobalDef] = field(default_factory=dict)
    prototypes: Set[str] = field(default_factory=set)

    def get(self, fid: FuncId) -> Optional[FunctionDef]:
        return self.functions.get(fid)

    @property
    def external_names(self) -> Set[str]:
        """Declared somewhere but never defined in this project."""
        return {n for n in self.prototypes if n not in self.by_name}
