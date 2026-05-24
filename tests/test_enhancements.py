"""Tests for new enhancements: MSan/TSan patterns, multi-sanitizer,
LibFuzzer support, taint analysis, new vuln types."""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sandbox.evidence import (
    EvidenceClassifier, EvidenceLevel, ClassificationResult,
    MSAN_PATTERNS, TSAN_PATTERNS, SANITIZER_TO_CWE,
)
from sandbox.executor import (
    SandboxExecutor, SandboxConfig, ExecutionResult, SANITIZER_PROFILES,
)
from exploit_db.exploit_adapter import ExploitAdapter, ExploitHarness
from analysis.taint import TaintAnalyzer, TaintResult, TaintPath


# ─── MSan / TSan Evidence ──────────────────────────────────────────

class TestMsanEvidence:
    def setup_method(self):
        self.classifier = EvidenceClassifier()

    def test_msan_confirmed(self):
        result = self.classifier.classify(
            exit_code=77,
            stdout="",
            stderr="==1234==WARNING: MemorySanitizer: use-of-uninitialized-value\n"
                   "    #0 0x55555 in main /test.c:5",
        )
        assert result.level == EvidenceLevel.CONFIRMED
        assert result.confidence >= 0.93

    def test_msan_heap_alloc(self):
        result = self.classifier.classify(
            exit_code=77,
            stdout="",
            stderr="MemorySanitizer: use-of-uninitialized-value\n"
                   "Uninitialized value was created by a heap allocation",
        )
        assert result.level == EvidenceLevel.CONFIRMED
        assert result.sanitizer_source == "msan"

    def test_msan_cwe_detection(self):
        result = self.classifier.classify(
            exit_code=77,
            stdout="",
            stderr="MemorySanitizer: use-of-uninitialized-value",
        )
        assert result.cwe_id == "CWE-457"


class TestTsanEvidence:
    def setup_method(self):
        self.classifier = EvidenceClassifier()

    def test_tsan_data_race_confirmed(self):
        result = self.classifier.classify(
            exit_code=66,
            stdout="",
            stderr="WARNING: ThreadSanitizer: data race (pid=1234)\n"
                   "  Write of size 4 at 0x7f...\n"
                   "    #0 worker /test.c:10",
        )
        assert result.level == EvidenceLevel.CONFIRMED
        assert result.confidence >= 0.91

    def test_tsan_lock_order_inversion(self):
        result = self.classifier.classify(
            exit_code=66,
            stdout="",
            stderr="WARNING: ThreadSanitizer: lock-order-inversion",
        )
        assert result.level == EvidenceLevel.CONFIRMED
        assert result.sanitizer_source == "tsan"

    def test_tsan_cwe_detection(self):
        result = self.classifier.classify(
            exit_code=66,
            stdout="",
            stderr="ThreadSanitizer: data race",
        )
        assert result.cwe_id == "CWE-362"

    def test_tsan_thread_leak(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="",
            stderr="WARNING: ThreadSanitizer: thread leak",
        )
        assert result.level == EvidenceLevel.CONFIRMED


# ─── CWE Mapping ───────────────────────────────────────────────────

class TestCweMapping:
    def setup_method(self):
        self.classifier = EvidenceClassifier()

    def test_asan_heap_overflow_cwe(self):
        result = self.classifier.classify(
            exit_code=1,
            stdout="",
            stderr="ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602...",
        )
        assert result.cwe_id == "CWE-122"

    def test_asan_uaf_cwe(self):
        result = self.classifier.classify(
            exit_code=1,
            stdout="",
            stderr="ERROR: AddressSanitizer: heap-use-after-free on address 0x602...",
        )
        assert result.cwe_id == "CWE-416"

    def test_ubsan_div_zero_cwe(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="runtime error: division by zero",
            stderr="",
        )
        assert result.cwe_id == "CWE-369"

    def test_ubsan_integer_overflow_cwe(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="runtime error: signed integer overflow: 2147483647 + 1",
            stderr="",
        )
        assert result.cwe_id == "CWE-190"

    def test_sanitizer_source_asan(self):
        result = self.classifier.classify(
            exit_code=1, stdout="",
            stderr="ERROR: AddressSanitizer: stack-buffer-overflow",
        )
        assert result.sanitizer_source == "asan"

    def test_sanitizer_source_ubsan(self):
        result = self.classifier.classify(
            exit_code=0,
            stdout="runtime error: signed integer overflow",
            stderr="",
        )
        assert result.sanitizer_source == "ubsan"

    def test_sanitizer_to_cwe_dict_coverage(self):
        assert len(SANITIZER_TO_CWE) >= 20
        assert "heap-buffer-overflow" in SANITIZER_TO_CWE
        assert "data race" in SANITIZER_TO_CWE
        assert "use-of-uninitialized-value" in SANITIZER_TO_CWE


# ─── Multi-Sanitizer Config ────────────────────────────────────────

class TestMultiSanitizer:
    def test_sanitizer_profiles_defined(self):
        assert "asan_ubsan" in SANITIZER_PROFILES
        assert "msan" in SANITIZER_PROFILES
        assert "tsan" in SANITIZER_PROFILES

    def test_asan_profile_has_gcc(self):
        assert SANITIZER_PROFILES["asan_ubsan"]["compile_gcc"] is not None

    def test_msan_requires_clang(self):
        assert SANITIZER_PROFILES["msan"]["compile_gcc"] is None
        assert "clang" in SANITIZER_PROFILES["msan"]["compile_clang"]

    def test_tsan_requires_clang(self):
        assert SANITIZER_PROFILES["tsan"]["compile_gcc"] is None
        assert "clang" in SANITIZER_PROFILES["tsan"]["compile_clang"]

    def test_fuzz_profiles(self):
        for name, profile in SANITIZER_PROFILES.items():
            assert "fuzzer" in profile["compile_fuzz"]

    def test_select_profiles_buffer_overflow(self):
        executor = SandboxExecutor(SandboxConfig(use_multi_sanitizer=True))
        profiles = executor._select_profiles("buffer_overflow")
        profile_names = [n for n, _ in profiles]
        assert "asan_ubsan" in profile_names

    def test_select_profiles_uninitialized(self):
        executor = SandboxExecutor(SandboxConfig(use_multi_sanitizer=True))
        profiles = executor._select_profiles("uninitialized_read")
        profile_names = [n for n, _ in profiles]
        assert "msan" in profile_names

    def test_select_profiles_race(self):
        executor = SandboxExecutor(SandboxConfig(use_multi_sanitizer=True))
        profiles = executor._select_profiles("race_condition")
        profile_names = [n for n, _ in profiles]
        assert "tsan" in profile_names

    def test_select_profiles_unknown_defaults(self):
        executor = SandboxExecutor(SandboxConfig(use_multi_sanitizer=True))
        profiles = executor._select_profiles("unknown_type")
        profile_names = [n for n, _ in profiles]
        assert "asan_ubsan" in profile_names

    def test_config_fields(self):
        config = SandboxConfig(
            use_multi_sanitizer=True,
            use_libfuzzer=True,
            fuzz_time=20,
            prefer_clang=True,
        )
        assert config.use_multi_sanitizer is True
        assert config.use_libfuzzer is True
        assert config.fuzz_time == 20
        assert config.prefer_clang is True


# ─── LibFuzzer Harness Wrapping ─────────────────────────────────────

class TestFuzzHarnessWrapping:
    def test_wrap_simple_harness(self):
        code = """#include <stdio.h>
#include <string.h>

void vulnerable(const char *s) {
    char buf[8];
    strcpy(buf, s);
}

int main(int argc, char *argv[]) {
    vulnerable("AAAA");
    return 0;
}
"""
        wrapped = SandboxExecutor._wrap_as_fuzz_harness(code)
        assert wrapped is not None
        assert "LLVMFuzzerTestOneInput" in wrapped
        assert "vulnerable(input)" in wrapped
        assert "int main" not in wrapped

    def test_wrap_no_main_returns_none(self):
        code = "void foo() {}\n"
        wrapped = SandboxExecutor._wrap_as_fuzz_harness(code)
        assert wrapped is None

    def test_wrap_adds_headers(self):
        code = """#include <stdio.h>
void vuln(const char *s) { printf("%s", s); }
int main() { vuln("test"); return 0; }
"""
        wrapped = SandboxExecutor._wrap_as_fuzz_harness(code)
        assert wrapped is not None
        assert "#include <stdint.h>" in wrapped
        assert "#include <stddef.h>" in wrapped

    def test_wrap_preserves_includes(self):
        code = """#include <stdio.h>
#include <string.h>
#include <stdint.h>
void process(const char *data) { char b[4]; strcpy(b, data); }
int main() { process("test"); return 0; }
"""
        wrapped = SandboxExecutor._wrap_as_fuzz_harness(code)
        assert wrapped is not None
        assert wrapped.count("#include <stdint.h>") == 1


# ─── New Generic Harness Generators ────────────────────────────────

class TestNewGenericHarnesses:
    def setup_method(self):
        self.adapter = ExploitAdapter()

    def test_off_by_one_harness(self):
        harness = self.adapter._generate_generic_harness(
            "void f(char *s){char b[64]; int i; for(i=0;i<=64;i++) b[i]=s[i];}",
            "off_by_one", "c",
        )
        assert harness.is_valid
        assert harness.vuln_type == "off_by_one"
        assert "AddressSanitizer" in harness.expected_signals

    def test_division_by_zero_harness(self):
        harness = self.adapter._generate_generic_harness(
            "int divide(int a, int b) { return a / b; }",
            "division_by_zero", "c",
        )
        assert harness.is_valid
        assert harness.vuln_type == "division_by_zero"
        assert "division by zero" in harness.expected_signals

    def test_uninitialized_read_harness(self):
        harness = self.adapter._generate_generic_harness(
            "void f(char *s) { printf(\"%s\", s); }",
            "uninitialized_read", "c",
        )
        assert harness.is_valid
        assert harness.vuln_type == "uninitialized_read"
        assert "clang" in harness.compilation_cmd
        assert "memory" in harness.compilation_cmd

    def test_race_condition_harness(self):
        harness = self.adapter._generate_generic_harness(
            "void f(char *s) { printf(\"%s\", s); }",
            "race_condition", "c",
        )
        assert harness.is_valid
        assert harness.vuln_type == "race_condition"
        assert "thread" in harness.compilation_cmd
        assert "pthread" in harness.harness_code

    def test_stack_exhaustion_harness(self):
        harness = self.adapter._generate_generic_harness(
            "void f(int n) { if(n>0) f(n-1); }",
            "stack_exhaustion", "c",
        )
        assert harness.is_valid
        assert harness.vuln_type == "stack_exhaustion"
        assert "recurse" in harness.harness_code

    def test_alloc_dealloc_mismatch_harness(self):
        harness = self.adapter._generate_generic_harness(
            "void cleanup(char *p) { free(p); }",
            "alloc_dealloc_mismatch", "c",
        )
        assert harness.is_valid
        assert harness.vuln_type == "alloc_dealloc_mismatch"

    def test_all_vuln_types_have_hints(self):
        from exploit_db.exploit_adapter import VULN_TYPE_HINTS
        expected = [
            "buffer_overflow", "stack_buffer_overflow", "heap_buffer_overflow",
            "use_after_free", "double_free", "integer_overflow",
            "format_string", "null_pointer_deref",
            "off_by_one", "division_by_zero", "uninitialized_read",
            "race_condition", "stack_exhaustion", "alloc_dealloc_mismatch",
        ]
        for vt in expected:
            assert vt in VULN_TYPE_HINTS, f"Missing hint for {vt}"


# ─── Taint Analysis ─────────────────────────────────────────────────

class TestTaintAnalyzer:
    def test_heuristic_finds_strcpy_sink(self):
        code = """
#include <stdio.h>
#include <string.h>

void process(char *input) {
    char buf[32];
    strcpy(buf, input);
    printf("%s\\n", buf);
}

int main(int argc, char *argv[]) {
    process(argv[1]);
    return 0;
}
"""
        analyzer = TaintAnalyzer()
        result = analyzer._analyze_heuristic(code, "test.c")
        assert result.analysis_method == "heuristic"
        assert len(result.functions) >= 2
        func_names = [f.name for f in result.functions]
        assert "process" in func_names
        assert "main" in func_names

    def test_heuristic_detects_taint_path(self):
        code = """
#include <string.h>
void vuln(char *s) {
    char buf[8];
    strcpy(buf, s);
}
int main(int argc, char *argv[]) {
    vuln(argv[1]);
    return 0;
}
"""
        analyzer = TaintAnalyzer()
        result = analyzer._analyze_heuristic(code, "test.c")
        assert result.has_taint
        sink_funcs = [p.sink_func for p in result.taint_paths]
        assert "strcpy" in sink_funcs

    def test_heuristic_detects_system_call(self):
        code = """
#include <stdlib.h>
int main(int argc, char *argv[]) {
    system(argv[1]);
    return 0;
}
"""
        analyzer = TaintAnalyzer()
        result = analyzer._analyze_heuristic(code, "test.c")
        assert result.has_taint
        sink_funcs = [p.sink_func for p in result.taint_paths]
        assert "system" in sink_funcs
        categories = [p.sink_category for p in result.taint_paths]
        assert "exec" in categories

    def test_no_taint_in_safe_code(self):
        code = """
#include <stdio.h>
int main() {
    printf("hello\\n");
    return 0;
}
"""
        analyzer = TaintAnalyzer()
        result = analyzer._analyze_heuristic(code, "test.c")
        assert not result.has_taint

    def test_format_for_llm(self):
        result = TaintResult(
            taint_paths=[TaintPath(
                source_func="argv",
                source_line=5,
                sink_func="strcpy",
                sink_line=3,
                sink_category="memory",
            )],
            call_graph=[],
            functions=[],
            analysis_method="heuristic",
        )
        formatted = result.format_for_llm()
        assert "argv" in formatted
        assert "strcpy" in formatted
        assert "memory" in formatted

    def test_caller_chain(self):
        from analysis.taint import CallEdge
        result = TaintResult(
            taint_paths=[],
            call_graph=[
                CallEdge(caller="main", callee="process", line=10),
                CallEdge(caller="process", callee="strcpy", line=5),
            ],
            functions=[],
        )
        chain = result.get_caller_chain("strcpy")
        assert "process" in chain
        assert "main" in chain

    def test_dangerous_functions(self):
        result = TaintResult(
            taint_paths=[
                TaintPath("argv", 1, "strcpy", 3, "memory"),
                TaintPath("argv", 1, "system", 5, "exec"),
            ],
            call_graph=[],
            functions=[],
        )
        dangerous = result.get_dangerous_functions()
        assert "strcpy" in dangerous
        assert "system" in dangerous

    def test_joern_not_available(self):
        analyzer = TaintAnalyzer(joern_path="/nonexistent/joern")
        assert not analyzer.joern_available

    def test_analyze_falls_back_to_heuristic(self):
        analyzer = TaintAnalyzer(joern_path="/nonexistent/joern")
        code = "void f(char *s) { char b[8]; strcpy(b, s); }"
        result = analyzer.analyze(code)
        assert result.analysis_method == "heuristic"


# ─── Pipeline Integration ───────────────────────────────────────────

class TestScannerConfigNewFields:
    def test_new_config_defaults(self):
        from scanner.config import ScannerConfig
        config = ScannerConfig()
        assert config.sandbox_use_multi_sanitizer is False
        assert config.sandbox_use_libfuzzer is False
        assert config.sandbox_fuzz_time == 15
        assert config.sandbox_prefer_clang is False
        assert config.use_taint_analysis is False

    def test_new_config_overrides(self):
        from scanner.config import ScannerConfig
        config = ScannerConfig(
            sandbox_use_multi_sanitizer=True,
            sandbox_use_libfuzzer=True,
            sandbox_fuzz_time=30,
            use_taint_analysis=True,
        )
        assert config.sandbox_use_multi_sanitizer is True
        assert config.sandbox_use_libfuzzer is True
        assert config.sandbox_fuzz_time == 30
        assert config.use_taint_analysis is True


class TestScanResultNewFields:
    def test_scan_result_has_cwe_field(self):
        from scanner.pipeline import ScanResult
        result = ScanResult(
            verdict="CONFIRMED", confidence=0.95,
            cwe_id="CWE-122", sanitizer_source="asan",
        )
        assert result.cwe_id == "CWE-122"
        assert result.sanitizer_source == "asan"

    def test_scan_result_has_taint_context(self):
        from scanner.pipeline import ScanResult
        result = ScanResult(
            verdict="CONFIRMED", confidence=0.95,
            taint_context="argv -> strcpy",
        )
        assert result.taint_context == "argv -> strcpy"


# ─── New Template Files ─────────────────────────────────────────────

class TestNewTemplateFiles:
    def test_heap_buffer_overflow_template_exists(self):
        path = os.path.join(
            os.path.dirname(__file__), "..",
            "exploit_db", "templates", "heap_buffer_overflow.json",
        )
        assert os.path.exists(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert len(data) >= 2
        assert data[0]["vuln_type"] == "heap_buffer_overflow"

    def test_null_pointer_deref_template_exists(self):
        path = os.path.join(
            os.path.dirname(__file__), "..",
            "exploit_db", "templates", "null_pointer_deref.json",
        )
        assert os.path.exists(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert data[0]["vuln_type"] == "null_pointer_deref"

    def test_off_by_one_template_exists(self):
        path = os.path.join(
            os.path.dirname(__file__), "..",
            "exploit_db", "templates", "off_by_one.json",
        )
        assert os.path.exists(path)

    def test_division_by_zero_template_exists(self):
        path = os.path.join(
            os.path.dirname(__file__), "..",
            "exploit_db", "templates", "division_by_zero.json",
        )
        assert os.path.exists(path)

    def test_uninitialized_read_template_exists(self):
        path = os.path.join(
            os.path.dirname(__file__), "..",
            "exploit_db", "templates", "uninitialized_read.json",
        )
        assert os.path.exists(path)

    def test_race_condition_template_exists(self):
        path = os.path.join(
            os.path.dirname(__file__), "..",
            "exploit_db", "templates", "race_condition.json",
        )
        assert os.path.exists(path)

    def test_total_template_count(self):
        import glob
        templates_dir = os.path.join(
            os.path.dirname(__file__), "..",
            "exploit_db", "templates",
        )
        files = glob.glob(os.path.join(templates_dir, "*.json"))
        assert len(files) >= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
