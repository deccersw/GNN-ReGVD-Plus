"""
Report generator — formats ScanResult into human-readable and machine-readable outputs.
"""

import json
from typing import List
from .pipeline import ScanResult


def format_report(result: ScanResult, code_snippet: str = "",
                  verbose: bool = False) -> str:
    """Format a ScanResult into a readable text report."""
    lines = []
    lines.append("=" * 60)
    lines.append("VULNERABILITY SCAN REPORT")
    lines.append("=" * 60)

    verdict_icons = {
        "CONFIRMED": "[!!]",
        "SUGGESTIVE": "[!]",
        "NEUTRAL": "[?]",
        "SAFE": "[OK]",
    }
    icon = verdict_icons.get(result.verdict, "[?]")
    lines.append(f"\n  Verdict:      {icon} {result.verdict}")
    lines.append(f"  Confidence:   {result.confidence:.2%}")
    lines.append(f"  Vuln Type:    {result.vuln_type} ({result.vuln_type_confidence:.2%})")
    lines.append(f"  Total Time:   {result.total_time:.2f}s")

    lines.append(f"\n  Detection Scores:")
    lines.append(f"    Classifier:   {result.cls_prob:.4f}")
    lines.append(f"    FAISS:        {result.faiss_score:.4f}")
    lines.append(f"    Hybrid:       {result.hybrid_score:.4f}")

    if result.exploits_tried > 0:
        lines.append(f"\n  Exploit Verification:")
        lines.append(f"    Tried:      {result.exploits_tried}")
        lines.append(f"    Confirmed:  {result.exploits_confirmed}")

        if verbose:
            for i, detail in enumerate(result.evidence_details):
                lines.append(f"\n    --- Exploit #{i+1}: {detail.get('exploit_id', 'N/A')} ---")
                lines.append(f"    Evidence:   {detail.get('evidence_level', 'N/A')}")
                lines.append(f"    Reason:     {detail.get('reason', 'N/A')}")
                lines.append(f"    Exit code:  {detail.get('exit_code', 'N/A')}")
                lines.append(f"    Compiled:   {detail.get('compiled', 'N/A')}")
                lines.append(f"    Time:       {detail.get('execution_time', 0):.2f}s")

    if result.triage_verdict:
        lines.append(f"\n  Triage Head (LLM):")
        lines.append(f"    Verdict:    {result.triage_verdict}")
        lines.append(f"    Confidence: {result.triage_confidence:.2%}")
        if verbose and result.triage_reasoning:
            # Truncate long reasoning to first 300 chars
            reasoning = result.triage_reasoning[:300]
            if len(result.triage_reasoning) > 300:
                reasoning += "..."
            lines.append(f"    Reasoning:  {reasoning}")

    if result.file:
        location = f"{result.file}:{result.function}:{result.start_line}"
        lines.append(f"\n  Location:     {location}")
        if result.inlined_functions:
            lines.append(f"    Inlined:    {', '.join(result.inlined_functions)}")
        if result.inline_truncated:
            lines.append(f"    Truncated:  yes (unit exceeded the token window)")

    if code_snippet:
        lines.append(f"\n  Code (first 200 chars):")
        lines.append(f"    {code_snippet[:200]}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def to_json(result: ScanResult) -> str:
    """Convert ScanResult to JSON string."""
    data = {
        "verdict": result.verdict,
        "confidence": result.confidence,
        "cls_prob": result.cls_prob,
        "faiss_score": result.faiss_score,
        "hybrid_score": result.hybrid_score,
        "vuln_type": result.vuln_type,
        "vuln_type_confidence": result.vuln_type_confidence,
        "exploits_tried": result.exploits_tried,
        "exploits_confirmed": result.exploits_confirmed,
        "evidence_details": result.evidence_details,
        "total_time": result.total_time,
    }
    if result.triage_verdict:
        data["triage_verdict"] = result.triage_verdict
        data["triage_confidence"] = result.triage_confidence
        data["triage_reasoning"] = result.triage_reasoning

    # Module 0 provenance. A project-level report without it cannot say which
    # function a verdict belongs to, which makes the whole result unusable.
    if result.unit_id or result.file:
        data.update({
            "unit_id": result.unit_id,
            "file": result.file,
            "function": result.function,
            "start_line": result.start_line,
            "end_line": result.end_line,
            "inline_depth_used": result.inline_depth_used,
            "inlined_functions": result.inlined_functions,
            "provenance_files": result.provenance_files,
            "inline_truncated": result.inline_truncated,
        })
    return json.dumps(data, indent=2)


def batch_summary(results: List[ScanResult]) -> str:
    """Generate a summary of batch scan results."""
    total = len(results)
    if total == 0:
        return "No results."

    confirmed = sum(1 for r in results if r.verdict == "CONFIRMED")
    suggestive = sum(1 for r in results if r.verdict == "SUGGESTIVE")
    neutral = sum(1 for r in results if r.verdict == "NEUTRAL")
    safe = sum(1 for r in results if r.verdict == "SAFE")
    avg_time = sum(r.total_time for r in results) / total

    lines = [
        f"Batch Summary ({total} samples):",
        f"  CONFIRMED:  {confirmed} ({confirmed/total:.1%})",
        f"  SUGGESTIVE: {suggestive} ({suggestive/total:.1%})",
        f"  NEUTRAL:    {neutral} ({neutral/total:.1%})",
        f"  SAFE:       {safe} ({safe/total:.1%})",
        f"  Avg time:   {avg_time:.2f}s",
    ]
    return "\n".join(lines)
