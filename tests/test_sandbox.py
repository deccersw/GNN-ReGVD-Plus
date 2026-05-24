"""Tests for SandboxExecutor and EvidenceClassifier."""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sandbox.evidence import (
    EvidenceClassifier, EvidenceLevel, ClassificationResult, AsanReport,
)
from sandbox.executor import SandboxExecutor, SandboxConfig, ExecutionResult
from sandbox import docker_manager


class TestEvidenceClassifier:
    def setup_method(self):
        self.classifier = EvidenceClassifier()

    def test_asan_confirmed(self):
        result = self.classifier.classify(
            exit_code=1,
            stdout="",
            stderr="ERROR: AddressSanitizer: stack-buffer-overflow on address 0x7fff...",
            expected_signals=["AddressSanitizer"],
        )
        assert result.level == EvidenceLevel.CONFIRMED
        assert "AddressSanitizer" in result.matched_patterns

    def test_ubsan_confirmed(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="test.c:10:15: runtime error: signed integer overflow: "
                   "2147483647 + 1 cannot be represented in type 'int'",
            stderr="",
            expected_signals=["UndefinedBehaviorSanitizer"],
        )
        assert result.level == EvidenceLevel.CONFIRMED

    def test_segfault_confirmed(self):
        result = self.classifier.classify(
            exit_code=-11,
            stdout="",
            stderr="",
        )
        assert result.level == EvidenceLevel.CONFIRMED
        assert "SIGSEGV" in result.matched_patterns

    def test_sigabrt_confirmed(self):
        result = self.classifier.classify(
            exit_code=134,
            stdout="",
            stderr="",
        )
        assert result.level == EvidenceLevel.CONFIRMED

    def test_nonzero_exit_suggestive(self):
        result = self.classifier.classify(
            exit_code=1,
            stdout="some output",
            stderr="",
        )
        assert result.level == EvidenceLevel.SUGGESTIVE

    def test_timeout_neutral(self):
        result = self.classifier.classify(
            exit_code=-1,
            stdout="",
            stderr="",
            timed_out=True,
        )
        assert result.level == EvidenceLevel.NEUTRAL

    def test_clean_exit_safe(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="No crash detected",
            stderr="",
        )
        assert result.level == EvidenceLevel.SAFE

    def test_expected_signal_match(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="heap-use-after-free detected",
            stderr="",
            expected_signals=["heap-use-after-free"],
        )
        assert result.level == EvidenceLevel.CONFIRMED

    def test_stack_smashing(self):
        result = self.classifier.classify(
            exit_code=134,
            stdout="",
            stderr="*** stack smashing detected ***: terminated",
        )
        assert result.level == EvidenceLevel.CONFIRMED


class TestSandboxExecutor:
    def test_init(self):
        executor = SandboxExecutor()
        assert executor.config.timeout == 30
        assert executor.config.memory_limit == "1g"

    def test_custom_config(self):
        config = SandboxConfig(timeout=60, memory_limit="2g")
        executor = SandboxExecutor(config)
        assert executor.config.timeout == 60
        assert executor.config.memory_limit == "2g"

    @pytest.mark.skipif(
        not os.path.exists("/usr/bin/gcc") and not os.path.exists("/opt/homebrew/bin/gcc-14"),
        reason="GCC not available"
    )
    def test_local_execution_safe_code(self):
        executor = SandboxExecutor()
        result = executor._execute_local(
            harness_code='#include <stdio.h>\nint main() { printf("hello\\n"); return 0; }',
            payload="",
            compilation_cmd="gcc -g",
            expected_signals=[],
            timeout=10,
        )
        assert result.harness_compiled
        assert result.exit_code == 0
        assert result.evidence.level == EvidenceLevel.SAFE

    @pytest.mark.skipif(
        not os.path.exists("/usr/bin/gcc") and not os.path.exists("/opt/homebrew/bin/gcc-14"),
        reason="GCC not available"
    )
    def test_local_execution_buffer_overflow(self):
        executor = SandboxExecutor()
        vuln_code = """
#include <stdio.h>
#include <string.h>

int main() {
    char buf[8];
    strcpy(buf, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
    printf("%s\\n", buf);
    return 0;
}
"""
        result = executor._execute_local(
            harness_code=vuln_code,
            payload="",
            compilation_cmd="gcc -fsanitize=address -g",
            expected_signals=["AddressSanitizer"],
            timeout=10,
        )
        assert result.harness_compiled
        if result.evidence.level == EvidenceLevel.CONFIRMED:
            assert any("AddressSanitizer" in p or "SEGV" in str(p)
                        for p in result.evidence.matched_patterns)


class TestValgrindEvidence:
    def setup_method(self):
        self.classifier = EvidenceClassifier()

    def test_valgrind_invalid_read_confirmed(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="",
            stderr="==12345== Invalid read of size 4\n"
                   "==12345== ERROR SUMMARY: 1 errors from 1 contexts",
        )
        assert result.level == EvidenceLevel.CONFIRMED
        assert result.valgrind_errors == 1

    def test_valgrind_invalid_write_confirmed(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="",
            stderr="==999== Invalid write of size 8\n"
                   "==999== ERROR SUMMARY: 2 errors from 1 contexts",
        )
        assert result.level == EvidenceLevel.CONFIRMED

    def test_valgrind_memory_leak_suggestive(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="",
            stderr="==100== definitely lost: 48 bytes in 1 blocks\n"
                   "==100== ERROR SUMMARY: 0 errors from 0 contexts",
        )
        assert result.level == EvidenceLevel.SUGGESTIVE

    def test_valgrind_uninitialised_suggestive(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="",
            stderr="==200== Use of uninitialised value of size 8\n"
                   "==200== ERROR SUMMARY: 3 errors from 2 contexts",
        )
        assert result.level == EvidenceLevel.SUGGESTIVE
        assert result.valgrind_errors == 3

    def test_valgrind_error_count_zero(self):
        result = self.classifier.classify(
            exit_code=0, stdout="ok", stderr="",
        )
        assert result.valgrind_errors == 0

    def test_valgrind_invalid_free_confirmed(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="",
            stderr="==555== Invalid free()\n"
                   "==555== ERROR SUMMARY: 1 errors from 1 contexts",
        )
        assert result.level == EvidenceLevel.CONFIRMED


class TestConfidenceScoring:
    def setup_method(self):
        self.classifier = EvidenceClassifier()

    def test_asan_high_confidence(self):
        result = self.classifier.classify(
            exit_code=1,
            stdout="",
            stderr="ERROR: AddressSanitizer: heap-buffer-overflow",
        )
        assert result.confidence >= 0.95

    def test_ubsan_confidence(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="runtime error: signed integer overflow",
            stderr="",
        )
        assert result.confidence >= 0.90

    def test_crash_confidence(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="",
            stderr="Segmentation fault (core dumped)",
        )
        assert result.confidence >= 0.85

    def test_multi_source_boosts_confidence(self):
        result = self.classifier.classify(
            exit_code=-11,
            stdout="",
            stderr="ERROR: AddressSanitizer: SEGV on unknown address\n"
                   "Segmentation fault",
        )
        assert result.confidence >= 0.95

    def test_signal_exit_code_confidence(self):
        result = self.classifier.classify(
            exit_code=-11, stdout="", stderr="",
        )
        assert result.confidence == 0.9

    def test_safe_exit_confidence(self):
        result = self.classifier.classify(
            exit_code=0, stdout="ok", stderr="",
        )
        assert result.confidence == 1.0

    def test_timeout_zero_confidence(self):
        result = self.classifier.classify(
            exit_code=-1, stdout="", stderr="", timed_out=True,
        )
        assert result.confidence == 0.0

    def test_suggestive_low_confidence(self):
        result = self.classifier.classify(
            exit_code=1, stdout="", stderr="",
        )
        assert result.confidence <= 0.5


class TestAsanReportParsing:
    def setup_method(self):
        self.classifier = EvidenceClassifier()

    def test_parse_asan_error_type(self):
        stderr = (
            "==1234==ERROR: AddressSanitizer: heap-buffer-overflow on address "
            "0x602000000014 at pc 0x5555 bp 0x7fff sp 0x7fff\n"
            "READ of size 4 at 0x602000000014 thread T0\n"
            "    #0 0x555555 in main /test.c:5\n"
            "Address 0x602000000014 is a wild pointer"
        )
        result = self.classifier.classify(exit_code=1, stdout="", stderr=stderr)
        assert result.asan_report is not None
        assert result.asan_report.error_type == "heap-buffer-overflow"
        assert result.asan_report.access_type == "READ"
        assert result.asan_report.access_size == 4
        assert result.asan_report.address == "0x602000000014"

    def test_parse_asan_stack_frames(self):
        stderr = (
            "ERROR: AddressSanitizer: stack-buffer-overflow\n"
            "WRITE of size 50 at 0x7ffc00000020 thread T0\n"
            "    #0 0xaaa in strcpy (/lib/libc.so.6)\n"
            "    #1 0xbbb in main (/test.c:10)\n"
        )
        result = self.classifier.classify(exit_code=1, stdout="", stderr=stderr)
        assert result.asan_report is not None
        assert len(result.asan_report.stack_frames) == 2
        assert "strcpy" in result.asan_report.stack_frames[0]
        assert "main" in result.asan_report.stack_frames[1]

    def test_no_asan_report_for_non_asan(self):
        result = self.classifier.classify(
            exit_code=-11, stdout="", stderr="",
        )
        assert result.asan_report is None

    def test_asan_write_access(self):
        stderr = (
            "ERROR: AddressSanitizer: use-after-free\n"
            "WRITE of size 8 at 0x604000000040 thread T0\n"
        )
        result = self.classifier.classify(exit_code=1, stdout="", stderr=stderr)
        assert result.asan_report.access_type == "WRITE"
        assert result.asan_report.access_size == 8


class TestExtendedPatterns:
    def setup_method(self):
        self.classifier = EvidenceClassifier()

    def test_container_overflow(self):
        result = self.classifier.classify(
            exit_code=1, stdout="",
            stderr="ERROR: AddressSanitizer: container-overflow",
        )
        assert result.level == EvidenceLevel.CONFIRMED

    def test_double_free_detected(self):
        result = self.classifier.classify(
            exit_code=134, stdout="",
            stderr="free(): double free detected in tcache 2",
        )
        assert result.level == EvidenceLevel.CONFIRMED

    def test_malloc_corruption(self):
        result = self.classifier.classify(
            exit_code=134, stdout="",
            stderr="malloc(): corrupted top size",
        )
        assert result.level == EvidenceLevel.CONFIRMED

    def test_corrupted_size_vs_prev(self):
        result = self.classifier.classify(
            exit_code=134, stdout="",
            stderr="corrupted size vs. prev_size while consolidating",
        )
        assert result.level == EvidenceLevel.CONFIRMED

    def test_index_out_of_bounds_ubsan(self):
        result = self.classifier.classify(
            exit_code=0, stdout="",
            stderr="runtime error: index 10 out of bounds for type 'int [5]'",
        )
        assert result.level == EvidenceLevel.CONFIRMED

    def test_sigfpe_exit_code(self):
        result = self.classifier.classify(
            exit_code=136, stdout="", stderr="",
        )
        assert result.level == EvidenceLevel.CONFIRMED
        assert "SIGFPE" in result.matched_patterns

    def test_sigill_exit_code(self):
        result = self.classifier.classify(
            exit_code=132, stdout="", stderr="",
        )
        assert result.level == EvidenceLevel.CONFIRMED
        assert "SIGILL" in result.matched_patterns

    def test_trace_breakpoint_trap(self):
        result = self.classifier.classify(
            exit_code=133, stdout="",
            stderr="Trace/breakpoint trap",
        )
        assert result.level == EvidenceLevel.CONFIRMED

    def test_suggestive_panic(self):
        result = self.classifier.classify(
            exit_code=1, stdout="", stderr="PANIC: unexpected state",
        )
        assert result.level == EvidenceLevel.SUGGESTIVE

    def test_suggestive_assertion(self):
        result = self.classifier.classify(
            exit_code=1, stdout="assertion failed: x > 0", stderr="",
        )
        assert result.level == EvidenceLevel.SUGGESTIVE


class TestBatchExecution:
    def test_batch_early_stop(self):
        executor = SandboxExecutor()
        with patch.object(executor, "execute") as mock_exec:
            safe_result = ExecutionResult(
                exit_code=0, stdout="ok", stderr="",
                evidence=ClassificationResult(
                    level=EvidenceLevel.SAFE, reason="clean", matched_patterns=[],
                ),
            )
            confirmed_result = ExecutionResult(
                exit_code=1, stdout="", stderr="crash",
                evidence=ClassificationResult(
                    level=EvidenceLevel.CONFIRMED, reason="crash", matched_patterns=["SIGSEGV"],
                ),
            )
            mock_exec.side_effect = [safe_result, confirmed_result, safe_result]

            harnesses = [
                {"harness_code": "c1", "payload": "p1", "compilation_cmd": "gcc"},
                {"harness_code": "c2", "payload": "p2", "compilation_cmd": "gcc"},
                {"harness_code": "c3", "payload": "p3", "compilation_cmd": "gcc"},
            ]
            results = executor.execute_batch(harnesses, early_stop=True)

            assert len(results) == 2
            assert results[1].evidence.level == EvidenceLevel.CONFIRMED

    def test_batch_no_early_stop(self):
        executor = SandboxExecutor()
        with patch.object(executor, "execute") as mock_exec:
            confirmed = ExecutionResult(
                exit_code=1, stdout="", stderr="crash",
                evidence=ClassificationResult(
                    level=EvidenceLevel.CONFIRMED, reason="crash", matched_patterns=[],
                ),
            )
            mock_exec.return_value = confirmed

            harnesses = [
                {"harness_code": f"c{i}", "payload": f"p{i}", "compilation_cmd": "gcc"}
                for i in range(3)
            ]
            results = executor.execute_batch(harnesses, early_stop=False)

            assert len(results) == 3


class TestDockerManager:
    def test_get_image_name_default(self):
        assert docker_manager.get_image_name() == "gnn-regvd-sandbox:c-cpp"

    def test_get_image_name_custom(self):
        assert docker_manager.get_image_name("python") == "gnn-regvd-sandbox:python"

    def test_seccomp_profile_path(self):
        path = docker_manager.get_seccomp_profile_path()
        assert path.endswith("seccomp_sandbox.json")
        assert os.path.exists(path)

    @patch("subprocess.run")
    def test_image_exists_true(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert docker_manager.image_exists("c-cpp") is True

    @patch("subprocess.run")
    def test_image_exists_false(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        assert docker_manager.image_exists("c-cpp") is False

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_image_exists_no_docker(self, mock_run):
        assert docker_manager.image_exists("c-cpp") is False

    @patch("sandbox.docker_manager.image_exists", return_value=True)
    def test_build_image_skip_existing(self, mock_exists):
        assert docker_manager.build_image() is True


class TestSandboxConfigFields:
    def test_default_config_fields(self):
        config = SandboxConfig()
        assert config.use_seccomp is True
        assert config.use_valgrind is False
        assert config.cap_drop == "ALL"
        assert config.security_opt == "no-new-privileges"
        assert config.pids_limit == 128
        assert config.network == "none"
        assert config.read_only is True

    def test_valgrind_enabled(self):
        config = SandboxConfig(use_valgrind=True)
        assert config.use_valgrind is True


class TestBuildRunScript:
    def setup_method(self):
        self.executor = SandboxExecutor()

    def test_script_contains_compilation(self):
        script = self.executor._build_run_script(
            "gcc -fsanitize=address -g", 30, False,
        )
        assert "gcc -fsanitize=address -g" in script
        assert "===COMPILATION_OK===" in script
        assert "EXIT_CODE=" in script

    def test_script_with_valgrind(self):
        script = self.executor._build_run_script(
            "gcc -g", 30, True,
        )
        assert "===VALGRIND_START===" in script
        assert "===VALGRIND_END===" in script
        assert "valgrind" in script
        assert "--leak-check=full" in script

    def test_script_without_valgrind(self):
        script = self.executor._build_run_script(
            "gcc -g", 30, False,
        )
        assert "VALGRIND" not in script


HAS_GCC = os.path.exists("/usr/bin/gcc") or os.path.exists("/opt/homebrew/bin/gcc-14")


@pytest.mark.skipif(not HAS_GCC, reason="GCC not available")
class TestKnownVulnerableCode:
    """Tests with known-vulnerable C code patterns (PLAN.md Phase 2 item 4)."""

    def setup_method(self):
        self.executor = SandboxExecutor()

    def test_heap_use_after_free(self):
        code = """
#include <stdlib.h>
#include <string.h>
int main() {
    char *p = (char *)malloc(16);
    strcpy(p, "hello");
    free(p);
    p[0] = 'X';
    return 0;
}
"""
        result = self.executor._execute_local(
            harness_code=code, payload="",
            compilation_cmd="gcc -fsanitize=address -g",
            expected_signals=["AddressSanitizer"], timeout=10,
        )
        assert result.harness_compiled
        if result.evidence.level == EvidenceLevel.CONFIRMED:
            assert result.evidence.confidence >= 0.85

    def test_double_free(self):
        code = """
#include <stdlib.h>
int main() {
    char *p = (char *)malloc(32);
    free(p);
    free(p);
    return 0;
}
"""
        result = self.executor._execute_local(
            harness_code=code, payload="",
            compilation_cmd="gcc -fsanitize=address -g",
            expected_signals=["AddressSanitizer"], timeout=10,
        )
        assert result.harness_compiled
        if result.evidence.level == EvidenceLevel.CONFIRMED:
            assert result.evidence.asan_report is not None or \
                   len(result.evidence.matched_patterns) > 0

    def test_integer_overflow(self):
        code = """
#include <limits.h>
#include <stdio.h>
int main() {
    int x = INT_MAX;
    int y = x + 1;
    printf("%d\\n", y);
    return 0;
}
"""
        result = self.executor._execute_local(
            harness_code=code, payload="",
            compilation_cmd="gcc -fsanitize=undefined -g",
            expected_signals=["runtime error"], timeout=10,
        )
        assert result.harness_compiled

    def test_null_pointer_deref(self):
        code = """
#include <stdlib.h>
int main() {
    int *p = NULL;
    *p = 42;
    return 0;
}
"""
        result = self.executor._execute_local(
            harness_code=code, payload="",
            compilation_cmd="gcc -fsanitize=address -g",
            expected_signals=["AddressSanitizer", "SEGV"], timeout=10,
        )
        assert result.harness_compiled
        assert result.evidence.level in (EvidenceLevel.CONFIRMED, EvidenceLevel.SUGGESTIVE)

    def test_format_string_vuln(self):
        code = """
#include <stdio.h>
int main() {
    char buf[64];
    snprintf(buf, sizeof(buf), "%s%s%s%s%s%s%s%s%s%s",
        "AAAA","AAAA","AAAA","AAAA","AAAA",
        "AAAA","AAAA","AAAA","AAAA","AAAA");
    printf(buf);
    return 0;
}
"""
        result = self.executor._execute_local(
            harness_code=code, payload="",
            compilation_cmd="gcc -g",
            expected_signals=[], timeout=10,
        )
        assert result.harness_compiled

    def test_safe_code_stays_safe(self):
        code = """
#include <stdio.h>
#include <string.h>
int main() {
    char buf[32];
    strncpy(buf, "safe", sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\\0';
    printf("%s\\n", buf);
    return 0;
}
"""
        result = self.executor._execute_local(
            harness_code=code, payload="",
            compilation_cmd="gcc -fsanitize=address,undefined -g",
            expected_signals=[], timeout=10,
        )
        assert result.harness_compiled
        assert result.evidence.level == EvidenceLevel.SAFE
        assert result.evidence.confidence == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
