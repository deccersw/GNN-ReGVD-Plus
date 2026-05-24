"""
Evidence Classifier — analyzes sandbox execution output to determine
whether a vulnerability was successfully exploited.

Classification levels (from VBF paper, extended):
  CONFIRMED  — direct proof of exploitation (ASan report, crash with known pattern)
  SUGGESTIVE — indirect signs (abnormal exit code, suspicious output)
  NEUTRAL    — no signal (timeout, clean exit with no output)
  SAFE       — vulnerability not confirmed (sanitizers clean, expected output)
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


class EvidenceLevel(str, Enum):
    CONFIRMED = "CONFIRMED"
    SUGGESTIVE = "SUGGESTIVE"
    NEUTRAL = "NEUTRAL"
    SAFE = "SAFE"


ASAN_PATTERNS = [
    r"AddressSanitizer",
    r"heap-buffer-overflow",
    r"stack-buffer-overflow",
    r"global-buffer-overflow",
    r"heap-use-after-free",
    r"double-free",
    r"attempting double-free",
    r"stack-use-after-return",
    r"stack-use-after-scope",
    r"use-after-poison",
    r"alloc-dealloc-mismatch",
    r"SEGV on unknown address",
    r"READ of size \d+",
    r"WRITE of size \d+",
    r"container-overflow",
    r"dynamic-stack-buffer-overflow",
    r"initialization-order-fiasco",
    r"new-delete-type-mismatch",
]

UBSAN_PATTERNS = [
    r"UndefinedBehaviorSanitizer",
    r"runtime error:",
    r"signed integer overflow",
    r"unsigned integer overflow",
    r"division by zero",
    r"null pointer",
    r"misaligned address",
    r"shift exponent",
    r"load of value .* is not a valid value",
    r"member access within .* address",
    r"load of misaligned address",
    r"index \d+ out of bounds",
    r"implicit conversion from type",
    r"negation of \d+ cannot be represented",
]

VALGRIND_PATTERNS = [
    r"Invalid read of size \d+",
    r"Invalid write of size \d+",
    r"Invalid free\(\)",
    r"Mismatched free\(\) / delete",
    r"Conditional jump or move depends on uninitialised",
    r"Use of uninitialised value",
    r"Source and destination overlap",
    r"definitely lost: [1-9]\d* bytes",
    r"indirectly lost: [1-9]\d* bytes",
    r"possibly lost: [1-9]\d* bytes",
    r"HEAP SUMMARY",
    r"ERROR SUMMARY: [1-9]\d* errors",
]

CRASH_PATTERNS = [
    r"Segmentation fault",
    r"SIGSEGV",
    r"SIGABRT",
    r"SIGBUS",
    r"SIGFPE",
    r"SIGILL",
    r"Aborted",
    r"core dumped",
    r"\*\*\* stack smashing detected \*\*\*",
    r"munmap_chunk\(\): invalid pointer",
    r"free\(\): invalid pointer",
    r"free\(\): double free detected",
    r"free\(\): invalid next size",
    r"malloc\(\): corrupted",
    r"malloc\(\): memory corruption",
    r"double free or corruption",
    r"corrupted size vs\. prev_size",
    r"Trace/breakpoint trap",
]

MSAN_PATTERNS = [
    r"MemorySanitizer",
    r"use-of-uninitialized-value",
    r"Uninitialized value was created by a heap allocation",
    r"Uninitialized value was created by an alloca",
    r"Uninitialized value was stored to memory",
]

TSAN_PATTERNS = [
    r"ThreadSanitizer",
    r"data race",
    r"lock-order-inversion",
    r"thread leak",
    r"signal-unsafe call inside of a signal",
    r"destroy of a locked mutex",
    r"mutex .* is already locked",
]

SUGGESTIVE_PATTERNS = [
    r"FATAL|Fatal|fatal",
    r"PANIC|panic",
    r"assertion failed",
    r"ASSERT",
    r"out of bounds",
    r"buffer overrun",
]

SANITIZER_TO_CWE = {
    "heap-buffer-overflow": "CWE-122",
    "stack-buffer-overflow": "CWE-121",
    "global-buffer-overflow": "CWE-120",
    "heap-use-after-free": "CWE-416",
    "double-free": "CWE-415",
    "stack-use-after-return": "CWE-562",
    "stack-use-after-scope": "CWE-562",
    "alloc-dealloc-mismatch": "CWE-762",
    "container-overflow": "CWE-787",
    "dynamic-stack-buffer-overflow": "CWE-121",
    "initialization-order-fiasco": "CWE-696",
    "new-delete-type-mismatch": "CWE-843",
    "use-after-poison": "CWE-416",
    "signed integer overflow": "CWE-190",
    "unsigned integer overflow": "CWE-190",
    "division by zero": "CWE-369",
    "null pointer": "CWE-476",
    "shift exponent": "CWE-682",
    "misaligned address": "CWE-188",
    "use-of-uninitialized-value": "CWE-457",
    "data race": "CWE-362",
    "lock-order-inversion": "CWE-833",
    "thread leak": "CWE-404",
}

SIGNAL_EXIT_CODES = {
    -11: "SIGSEGV", 139: "SIGSEGV",
    -6: "SIGABRT", 134: "SIGABRT",
    -8: "SIGFPE", 136: "SIGFPE",
    -4: "SIGILL", 132: "SIGILL",
    -7: "SIGBUS", 135: "SIGBUS",
    -5: "SIGTRAP", 133: "SIGTRAP",
}


@dataclass
class AsanReport:
    """Parsed AddressSanitizer report."""
    error_type: str = ""
    access_type: str = ""
    access_size: int = 0
    address: str = ""
    description: str = ""
    stack_frames: List[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    level: EvidenceLevel
    reason: str
    matched_patterns: list
    confidence: float = 1.0
    asan_report: Optional[AsanReport] = None
    valgrind_errors: int = 0
    cwe_id: str = ""
    sanitizer_source: str = ""


class EvidenceClassifier:
    """
    Classifies sandbox execution results into evidence levels.
    """

    def classify(self, exit_code: int, stdout: str, stderr: str,
                 timed_out: bool = False,
                 expected_signals: list = None) -> ClassificationResult:
        combined = (stdout or "") + "\n" + (stderr or "")
        matched = []

        for pattern in ASAN_PATTERNS:
            if re.search(pattern, combined):
                matched.append(("ASAN", pattern))

        for pattern in UBSAN_PATTERNS:
            if re.search(pattern, combined):
                matched.append(("UBSAN", pattern))

        for pattern in MSAN_PATTERNS:
            if re.search(pattern, combined):
                matched.append(("MSAN", pattern))

        for pattern in TSAN_PATTERNS:
            if re.search(pattern, combined):
                matched.append(("TSAN", pattern))

        for pattern in CRASH_PATTERNS:
            if re.search(pattern, combined):
                matched.append(("CRASH", pattern))

        valgrind_matched = []
        for pattern in VALGRIND_PATTERNS:
            if re.search(pattern, combined):
                valgrind_matched.append(("VALGRIND", pattern))

        valgrind_errors = self._count_valgrind_errors(combined)

        asan_report = None
        if any(cat == "ASAN" for cat, _ in matched):
            asan_report = self._parse_asan_report(combined)

        cwe_id = self._detect_cwe(matched, combined)
        sanitizer_source = self._detect_sanitizer_source(matched)

        if matched:
            confidence = self._compute_confidence(matched, valgrind_matched, exit_code)
            reasons = [f"{cat}: {pat}" for cat, pat in matched[:5]]
            return ClassificationResult(
                level=EvidenceLevel.CONFIRMED,
                reason=f"Direct exploitation evidence: {'; '.join(reasons)}",
                matched_patterns=[p for _, p in matched],
                confidence=confidence,
                asan_report=asan_report,
                valgrind_errors=valgrind_errors,
                cwe_id=cwe_id,
                sanitizer_source=sanitizer_source,
            )

        sig = SIGNAL_EXIT_CODES.get(exit_code)
        if sig:
            return ClassificationResult(
                level=EvidenceLevel.CONFIRMED,
                reason=f"Process killed by {sig} (exit code {exit_code})",
                matched_patterns=[sig],
                confidence=0.9,
                valgrind_errors=valgrind_errors,
                cwe_id=cwe_id,
                sanitizer_source=sanitizer_source,
            )

        if expected_signals:
            for signal in expected_signals:
                if signal.lower() in combined.lower():
                    return ClassificationResult(
                        level=EvidenceLevel.CONFIRMED,
                        reason=f"Expected signal found: {signal}",
                        matched_patterns=[signal],
                        confidence=0.85,
                        valgrind_errors=valgrind_errors,
                        cwe_id=cwe_id,
                        sanitizer_source=sanitizer_source,
                    )

        if valgrind_matched:
            has_definite_leak = any("definitely lost" in p for _, p in valgrind_matched)
            has_invalid_access = any("Invalid" in p for _, p in valgrind_matched)

            if has_invalid_access:
                return ClassificationResult(
                    level=EvidenceLevel.CONFIRMED,
                    reason=f"Valgrind detected invalid memory access",
                    matched_patterns=[p for _, p in valgrind_matched],
                    confidence=0.85,
                    valgrind_errors=valgrind_errors,
                    cwe_id=cwe_id,
                    sanitizer_source="valgrind",
                )

            if has_definite_leak or valgrind_errors > 0:
                return ClassificationResult(
                    level=EvidenceLevel.SUGGESTIVE,
                    reason=f"Valgrind: {valgrind_errors} errors, "
                           f"memory issues detected",
                    matched_patterns=[p for _, p in valgrind_matched],
                    confidence=0.6,
                    valgrind_errors=valgrind_errors,
                    cwe_id="CWE-401",
                    sanitizer_source="valgrind",
                )

        suggestive_matched = []
        for pattern in SUGGESTIVE_PATTERNS:
            if re.search(pattern, combined):
                suggestive_matched.append(pattern)

        if exit_code != 0 and not timed_out:
            if suggestive_matched:
                return ClassificationResult(
                    level=EvidenceLevel.SUGGESTIVE,
                    reason=f"Non-zero exit ({exit_code}) with suspicious output: "
                           f"{'; '.join(suggestive_matched[:3])}",
                    matched_patterns=suggestive_matched,
                    confidence=0.5,
                    valgrind_errors=valgrind_errors,
                )
            return ClassificationResult(
                level=EvidenceLevel.SUGGESTIVE,
                reason=f"Non-zero exit code: {exit_code}",
                matched_patterns=[],
                confidence=0.3,
                valgrind_errors=valgrind_errors,
            )

        if timed_out:
            return ClassificationResult(
                level=EvidenceLevel.NEUTRAL,
                reason="Process timed out — no conclusive evidence",
                matched_patterns=[],
                confidence=0.0,
                valgrind_errors=valgrind_errors,
            )

        if exit_code == 0 and not matched and not suggestive_matched:
            return ClassificationResult(
                level=EvidenceLevel.SAFE,
                reason="Clean exit with no sanitizer reports",
                matched_patterns=[],
                confidence=1.0,
                valgrind_errors=valgrind_errors,
            )

        return ClassificationResult(
            level=EvidenceLevel.NEUTRAL,
            reason="Inconclusive — no clear signal",
            matched_patterns=[],
            confidence=0.0,
            valgrind_errors=valgrind_errors,
        )

    @staticmethod
    def _compute_confidence(matched: list, valgrind_matched: list,
                            exit_code: int) -> float:
        confidence = 0.7

        categories = set(cat for cat, _ in matched)
        if "ASAN" in categories:
            confidence = max(confidence, 0.95)
        if "MSAN" in categories:
            confidence = max(confidence, 0.93)
        if "TSAN" in categories:
            confidence = max(confidence, 0.91)
        if "UBSAN" in categories:
            confidence = max(confidence, 0.90)
        if "CRASH" in categories:
            confidence = max(confidence, 0.85)

        if len(categories) > 1:
            confidence = min(confidence + 0.05, 1.0)

        if valgrind_matched:
            confidence = min(confidence + 0.03, 1.0)

        sig = SIGNAL_EXIT_CODES.get(exit_code)
        if sig:
            confidence = min(confidence + 0.02, 1.0)

        return round(confidence, 3)

    @staticmethod
    def _parse_asan_report(output: str) -> AsanReport:
        report = AsanReport()

        error_match = re.search(
            r'ERROR: AddressSanitizer: (\S+)', output
        )
        if error_match:
            report.error_type = error_match.group(1)

        access_match = re.search(
            r'(READ|WRITE) of size (\d+) at (0x[0-9a-fA-F]+)', output
        )
        if access_match:
            report.access_type = access_match.group(1)
            report.access_size = int(access_match.group(2))
            report.address = access_match.group(3)

        desc_match = re.search(
            r'(Address 0x[0-9a-fA-F]+ is .+?)(?:\n|$)', output
        )
        if desc_match:
            report.description = desc_match.group(1)

        frame_pattern = re.compile(
            r'#(\d+) (0x[0-9a-fA-F]+) in (\S+) (.+?)(?:\n|$)'
        )
        for m in frame_pattern.finditer(output):
            report.stack_frames.append(
                f"#{m.group(1)} {m.group(3)} ({m.group(4).strip()})"
            )

        return report

    @staticmethod
    def _count_valgrind_errors(output: str) -> int:
        m = re.search(r'ERROR SUMMARY: (\d+) errors', output)
        if m:
            return int(m.group(1))
        return 0

    @staticmethod
    def _detect_cwe(matched: list, combined: str) -> str:
        for _, pattern in matched:
            for key, cwe in SANITIZER_TO_CWE.items():
                if key in pattern or re.search(re.escape(key), combined):
                    return cwe
        return ""

    @staticmethod
    def _detect_sanitizer_source(matched: list) -> str:
        categories = set(cat for cat, _ in matched)
        if "ASAN" in categories:
            return "asan"
        if "MSAN" in categories:
            return "msan"
        if "TSAN" in categories:
            return "tsan"
        if "UBSAN" in categories:
            return "ubsan"
        if "CRASH" in categories:
            return "crash"
        return ""
