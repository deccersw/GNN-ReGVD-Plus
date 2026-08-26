"""
Module 0: depth-limited interprocedural inlining.

Turns a C/C++ project directory into a list of :class:`AnalysisUnit` objects
that the existing GNN-ReGVD+ pipeline can consume unchanged. See
PLAN_INTERPROC_INLINING.md for the design and INTERPROC_DECISIONS.md for the
rationale behind each choice.

    from interproc import InterprocPipeline, InliningConfig

    units = InterprocPipeline(InliningConfig(max_depth=2)).build("./myproject")
    units[0].code_for_gnn       # inlined, fits the detector's token window
    units[0].code_for_sandbox   # self-contained, compiles
"""

from .config import InliningConfig
from .models import AnalysisUnit, CallEdge, FuncId, FunctionDef, ProjectIndex
from .units import InterprocPipeline, read_units_jsonl, write_units_jsonl

__all__ = [
    "InliningConfig",
    "InterprocPipeline",
    "AnalysisUnit",
    "CallEdge",
    "FuncId",
    "FunctionDef",
    "ProjectIndex",
    "read_units_jsonl",
    "write_units_jsonl",
]

__version__ = "0.1.0"
