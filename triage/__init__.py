"""
Triage Head — LLM-based vulnerability triage for C/C++ code.

Second verification head alongside sandbox: uses LLM reasoning
to classify GNN detections as true_positive / false_positive / uncertain.

Uses Joern CPG taint analysis (or regex heuristic fallback) for graph context,
matching the existing taint infrastructure in analysis/taint.py.
"""

from .triage_head import TriageHead
from .config import TriageConfig

__all__ = ["TriageHead", "TriageConfig"]
