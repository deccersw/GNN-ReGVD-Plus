"""End-to-end tests for the VulnerabilityScanner pipeline.

Uses mocked detector and exploit DB to test orchestration logic
without requiring GPU or trained models.
"""

import os
import sys
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scanner.config import ScannerConfig
from scanner.pipeline import VulnerabilityScanner, ScanResult
from scanner.report import format_report, to_json, batch_summary
from sandbox.evidence import EvidenceLevel, ClassificationResult
from sandbox.executor import ExecutionResult


def make_scanner(**overrides):
    config = ScannerConfig(**overrides)
    return VulnerabilityScanner(config)


def mock_detection(hybrid_score=0.9, cls_prob=0.85, faiss_score=0.8,
                   vuln_type="buffer_overflow", embedding=None):
    return {
        "hybrid_score": hybrid_score,
        "cls_prob": cls_prob,
        "faiss_score": faiss_score,
        "vuln_type": vuln_type,
        "vuln_type_confidence": 0.8,
        "embedding": embedding or [0.1] * 512,
    }


def mock_exploit(exploit_id="EX-001", vuln_type="buffer_overflow"):
    return {
        "exploit_id": exploit_id,
        "vuln_type": vuln_type,
        "similarity": 0.85,
        "payload_template": "AAAA",
        "harness_template": "",
    }


def mock_exec_result(level=EvidenceLevel.CONFIRMED, reason="test",
                     confidence=0.95):
    return ExecutionResult(
        exit_code=1 if level == EvidenceLevel.CONFIRMED else 0,
        stdout="",
        stderr="ASan report" if level == EvidenceLevel.CONFIRMED else "",
        evidence=ClassificationResult(
            level=level,
            reason=reason,
            matched_patterns=["AddressSanitizer"] if level == EvidenceLevel.CONFIRMED else [],
            confidence=confidence,
        ),
        execution_time=0.5,
        harness_compiled=True,
    )


class TestScannerConfig:
    def test_default_config(self):
        c = ScannerConfig()
        assert c.detection_threshold == 0.5
        assert c.max_exploits_per_sample == 3
        assert c.early_stop_on_confirmed is True
        assert c.use_multi_harness is True

    def test_config_overrides(self):
        c = ScannerConfig(
            detection_threshold=0.7,
            sandbox_use_valgrind=True,
            llm_cache_results=False,
        )
        assert c.detection_threshold == 0.7
        assert c.sandbox_use_valgrind is True
        assert c.llm_cache_results is False


class TestPipelineOrchestration:
    """Test pipeline flow with mocked modules."""

    def _scanner_with_mocks(self, detection=None, exploits=None,
                            exec_result=None):
        scanner = make_scanner()

        det = detection or mock_detection()
        scanner._detect = MagicMock(return_value=det)

        scanner._retrieve_exploits = MagicMock(
            return_value=exploits if exploits is not None else [mock_exploit()]
        )

        from exploit_db.exploit_adapter import ExploitAdapter, ExploitHarness
        adapter = ExploitAdapter()
        scanner._adapter = adapter

        from sandbox.executor import SandboxExecutor
        sandbox = SandboxExecutor()
        er = exec_result or mock_exec_result()
        sandbox.execute_batch = MagicMock(return_value=[er])
        sandbox.execute = MagicMock(return_value=er)
        scanner._sandbox = sandbox

        return scanner

    def test_safe_below_threshold(self):
        scanner = self._scanner_with_mocks(
            detection=mock_detection(hybrid_score=0.3)
        )
        result = scanner.scan("void f(){}")
        assert result.verdict == "SAFE"
        assert result.confidence > 0.5

    def test_confirmed_with_exploit(self):
        scanner = self._scanner_with_mocks()
        result = scanner.scan(
            "void f(const char *s){char b[8];strcpy(b,s);}"
        )
        assert result.verdict == "CONFIRMED"
        assert result.exploits_confirmed >= 1

    def test_neutral_no_exploits_safe_exec(self):
        scanner = self._scanner_with_mocks(
            exploits=[],
            exec_result=mock_exec_result(
                level=EvidenceLevel.SAFE, confidence=1.0
            ),
        )
        result = scanner.scan("void f(){}")
        assert result.verdict == "NEUTRAL"

    def test_neutral_safe_execution(self):
        scanner = self._scanner_with_mocks(
            exec_result=mock_exec_result(
                level=EvidenceLevel.SAFE, confidence=1.0
            )
        )
        result = scanner.scan(
            "void f(const char *s){char b[8];strcpy(b,s);}"
        )
        assert result.verdict == "NEUTRAL"

    def test_suggestive_execution(self):
        scanner = self._scanner_with_mocks(
            exec_result=mock_exec_result(
                level=EvidenceLevel.SUGGESTIVE, confidence=0.5
            )
        )
        result = scanner.scan(
            "void f(const char *s){char b[8];strcpy(b,s);}"
        )
        assert result.verdict == "SUGGESTIVE"

    def test_scan_result_fields(self):
        scanner = self._scanner_with_mocks()
        result = scanner.scan("void f(char *s){strcpy(s,s);}")
        assert isinstance(result, ScanResult)
        assert result.total_time >= 0
        assert result.vuln_type == "buffer_overflow"

    def test_scan_batch(self):
        scanner = self._scanner_with_mocks(
            detection=mock_detection(hybrid_score=0.3)
        )
        results = scanner.scan_batch(["void f(){}", "int g(){return 0;}"])
        assert len(results) == 2
        assert all(r.verdict == "SAFE" for r in results)


class TestVerdictAggregation:
    def test_confirmed_verdict(self):
        scanner = make_scanner()
        verdict = scanner._aggregate_verdict(0.9, [
            {"evidence_level": "CONFIRMED"},
        ], confirmed=1)
        assert verdict == "CONFIRMED"

    def test_suggestive_verdict(self):
        scanner = make_scanner()
        verdict = scanner._aggregate_verdict(0.9, [
            {"evidence_level": "SUGGESTIVE"},
        ], confirmed=0)
        assert verdict == "SUGGESTIVE"

    def test_neutral_verdict(self):
        scanner = make_scanner()
        verdict = scanner._aggregate_verdict(0.9, [
            {"evidence_level": "SAFE"},
        ], confirmed=0)
        assert verdict == "NEUTRAL"

    def test_high_score_no_evidence_is_suggestive(self):
        scanner = make_scanner()
        verdict = scanner._aggregate_verdict(0.8, [], confirmed=0)
        assert verdict == "SUGGESTIVE"

    def test_low_score_no_evidence_is_neutral(self):
        scanner = make_scanner()
        verdict = scanner._aggregate_verdict(0.6, [], confirmed=0)
        assert verdict == "NEUTRAL"


class TestConfidenceComputation:
    def test_confirmed_high_confidence(self):
        scanner = make_scanner()
        conf = scanner._compute_confidence(0.9, [
            {"evidence_level": "CONFIRMED", "confidence": 0.95},
        ], confirmed=1)
        assert conf > 0.9

    def test_no_evidence_uses_hybrid(self):
        scanner = make_scanner()
        conf = scanner._compute_confidence(0.75, [], confirmed=0)
        assert conf == 0.75

    def test_suggestive_reduces_confidence(self):
        scanner = make_scanner()
        conf = scanner._compute_confidence(0.9, [
            {"evidence_level": "SUGGESTIVE", "confidence": 0.5},
        ], confirmed=0)
        assert conf < 0.9

    def test_safe_evidence_reduces_more(self):
        scanner = make_scanner()
        conf = scanner._compute_confidence(0.9, [
            {"evidence_level": "SAFE", "confidence": 1.0},
        ], confirmed=0)
        assert conf < 0.7


class TestReportGeneration:
    def test_format_report(self):
        result = ScanResult(
            verdict="CONFIRMED", confidence=0.95,
            cls_prob=0.9, faiss_score=0.85, hybrid_score=0.88,
            vuln_type="buffer_overflow", vuln_type_confidence=0.8,
            exploits_tried=2, exploits_confirmed=1,
            evidence_details=[
                {"exploit_id": "EX-1", "evidence_level": "CONFIRMED",
                 "reason": "ASan", "exit_code": 1,
                 "compiled": True, "execution_time": 0.3},
            ],
            total_time=1.5,
        )
        report = format_report(result, verbose=True)
        assert "CONFIRMED" in report
        assert "buffer_overflow" in report
        assert "EX-1" in report

    def test_to_json(self):
        result = ScanResult(verdict="SAFE", confidence=0.95)
        j = to_json(result)
        data = json.loads(j)
        assert data["verdict"] == "SAFE"
        assert data["confidence"] == 0.95

    def test_batch_summary(self):
        results = [
            ScanResult(verdict="CONFIRMED", confidence=0.95, total_time=1.0),
            ScanResult(verdict="SAFE", confidence=0.9, total_time=0.5),
            ScanResult(verdict="SAFE", confidence=0.85, total_time=0.3),
        ]
        summary = batch_summary(results)
        assert "3 samples" in summary
        assert "CONFIRMED" in summary
        assert "SAFE" in summary

    def test_batch_summary_empty(self):
        assert batch_summary([]) == "No results."


class TestCLIParsing:
    def test_cli_help(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "scan_cli.py", "--help"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode == 0
        assert "GNN-ReGVD" in result.stdout


import json


HAS_GCC = os.path.exists("/usr/bin/gcc") or os.path.exists(
    "/opt/homebrew/bin/gcc-14"
)


@pytest.mark.skipif(not HAS_GCC, reason="GCC not available")
class TestSandboxOnlyMode:
    """Test sandbox-only mode with real compilation."""

    def test_sandbox_vuln_code(self):
        from sandbox.executor import SandboxExecutor, SandboxConfig
        from exploit_db.exploit_adapter import ExploitAdapter

        code = "void vuln(const char *s){char b[8];strcpy(b,s);}"
        adapter = ExploitAdapter()
        harness = adapter.adapt(code, {}, "buffer_overflow")

        assert harness.is_valid

        executor = SandboxExecutor(SandboxConfig(timeout=10))
        result = executor.execute(
            harness_code=harness.harness_code,
            payload=harness.payload,
            compilation_cmd=harness.compilation_cmd,
            expected_signals=harness.expected_signals,
        )
        assert result.harness_compiled

    def test_sandbox_safe_code(self):
        from sandbox.executor import SandboxExecutor, SandboxConfig
        from exploit_db.exploit_adapter import ExploitAdapter

        code = 'void safe(const char *s){char b[64];strncpy(b,s,63);b[63]=0;}'
        adapter = ExploitAdapter()
        harness = adapter.adapt(code, {}, "buffer_overflow")

        executor = SandboxExecutor(SandboxConfig(timeout=10))
        result = executor.execute(
            harness_code=harness.harness_code,
            payload=harness.payload,
            compilation_cmd=harness.compilation_cmd,
            expected_signals=harness.expected_signals,
        )
        assert result.harness_compiled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
