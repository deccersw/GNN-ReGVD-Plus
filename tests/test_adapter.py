"""Tests for ExploitAdapter (Phases 1 + 3)."""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exploit_db.exploit_adapter import (
    ExploitAdapter, ExploitHarness, VULN_TYPE_HINTS,
)


class TestExploitHarness:
    def test_valid_harness(self):
        h = ExploitHarness(
            harness_code="#include <stdio.h>\nint main() { return 0; }",
            payload="AAAA",
            compilation_cmd="gcc -o test test.c",
        )
        assert h.is_valid

    def test_invalid_harness_empty_code(self):
        h = ExploitHarness(harness_code="", payload="test", compilation_cmd="")
        assert not h.is_valid

    def test_invalid_harness_empty_payload(self):
        h = ExploitHarness(harness_code="code", payload="", compilation_cmd="")
        assert not h.is_valid


class TestExploitAdapter:
    def setup_method(self):
        self.adapter = ExploitAdapter()

    def test_extract_function_name(self):
        code = "void process_input(const char *data) { char buf[32]; strcpy(buf, data); }"
        name = ExploitAdapter._extract_function_name(code)
        assert name == "process_input"

    def test_extract_function_name_int_return(self):
        code = "int calculate_sum(int a, int b) { return a + b; }"
        name = ExploitAdapter._extract_function_name(code)
        assert name == "calculate_sum"

    def test_extract_function_name_static(self):
        code = "static int parse_header(const char *buf, int size) { return 0; }"
        name = ExploitAdapter._extract_function_name(code)
        assert name == "parse_header"

    def test_extract_function_name_fallback(self):
        code = "x = 1 + 2;"
        name = ExploitAdapter._extract_function_name(code)
        assert name == "vulnerable_function"

    def test_adapt_from_template(self):
        vuln_code = "void process(const char *input) { char buf[16]; strcpy(buf, input); }"
        template = {
            "exploit_id": "BO-TEST",
            "harness_template": '#include <stdio.h>\n{vulnerable_code}\nint main() { {func_name}("AAAA"); return 0; }',
            "payload_template": "echo AAAA | ./{func_name}",
            "expected_signals": ["AddressSanitizer"],
        }
        harness = self.adapter._adapt_from_template(vuln_code, template,
                                                     "buffer_overflow", "c")
        assert harness.is_valid
        assert "process" in harness.harness_code
        assert vuln_code in harness.harness_code

    def test_generic_buffer_overflow(self):
        vuln_code = "void handle(const char *s) { char b[8]; strcpy(b, s); }"
        harness = self.adapter._generic_buffer_overflow(vuln_code, "handle")
        assert harness.is_valid
        assert "handle" in harness.harness_code
        assert "-fsanitize" in harness.compilation_cmd

    def test_generic_use_after_free(self):
        vuln_code = "void test_uaf() { int *p = malloc(4); free(p); *p = 42; }"
        harness = self.adapter._generic_use_after_free(vuln_code, "test_uaf")
        assert harness.is_valid
        assert harness.vuln_type == "use_after_free"

    def test_generic_integer_overflow(self):
        vuln_code = "int add(int a, int b) { return a + b; }"
        harness = self.adapter._generic_integer_overflow(vuln_code, "add")
        assert harness.is_valid
        assert "-fsanitize=undefined" in harness.compilation_cmd

    def test_generic_format_string(self):
        vuln_code = "void log_msg(const char *msg) { printf(msg); }"
        harness = self.adapter._generic_format_string(vuln_code, "log_msg")
        assert harness.is_valid
        assert "%x" in harness.payload

    def test_adapt_full_fallback_chain(self):
        vuln_code = "void process(const char *in) { char buf[8]; strcpy(buf, in); }"
        template = {"exploit_id": "BO-FAIL", "harness_template": "",
                     "payload_template": ""}
        harness = self.adapter.adapt(vuln_code, template, "buffer_overflow")
        assert harness.is_valid
        assert "process" in harness.harness_code


class TestGenericDoubleFree:
    def test_double_free_harness(self):
        adapter = ExploitAdapter()
        vuln_code = "void bad_free(void *p) { free(p); free(p); }"
        harness = adapter._generic_double_free(vuln_code, "bad_free")
        assert harness.is_valid
        assert harness.vuln_type == "double_free"
        assert "double-free" in harness.expected_signals

    def test_generic_dispatches_double_free(self):
        adapter = ExploitAdapter()
        vuln_code = "void bad_free(void *p) { free(p); free(p); }"
        harness = adapter._generate_generic_harness(vuln_code, "double_free", "c")
        assert harness.vuln_type == "double_free"


class TestJSONExtraction:
    def test_extract_json_plain(self):
        text = '{"harness_code": "int main(){}", "payload": "AA"}'
        data = ExploitAdapter._extract_json(text)
        assert data is not None
        assert data["payload"] == "AA"

    def test_extract_json_markdown_wrapped(self):
        text = 'Here is the exploit:\n```json\n{"harness_code": "x", "payload": "y"}\n```'
        data = ExploitAdapter._extract_json(text)
        assert data is not None
        assert data["harness_code"] == "x"

    def test_extract_json_no_json(self):
        text = "No JSON here, just text"
        data = ExploitAdapter._extract_json(text)
        assert data is None

    def test_extract_code_block_c(self):
        text = 'Here:\n```c\n#include <stdio.h>\nint main() { return 0; }\n```'
        code = ExploitAdapter._extract_code_block(text)
        assert code is not None
        assert "int main" in code

    def test_extract_code_block_no_main(self):
        text = '```c\nvoid helper() {}\n```'
        code = ExploitAdapter._extract_code_block(text)
        assert code is None


class TestVulnTypeHints:
    def test_all_generic_types_have_hints(self):
        adapter = ExploitAdapter()
        expected_types = [
            "buffer_overflow", "heap_buffer_overflow", "use_after_free",
            "integer_overflow", "format_string", "null_pointer_deref",
        ]
        for vt in expected_types:
            assert vt in VULN_TYPE_HINTS, f"Missing hint for {vt}"


class TestLLMRetryWithRefinement:
    def test_retry_sends_error_feedback(self):
        adapter = ExploitAdapter(llm_backend="api", api_url="http://fake")

        call_count = 0
        prompts_seen = []

        def mock_call_llm(prompt):
            nonlocal call_count
            call_count += 1
            prompts_seen.append(prompt)
            if call_count == 1:
                return '{"harness_code": "bad code", "payload": "x"}'
            return ('{"harness_code": "#include <stdio.h>\\nint main(){return 0;}", '
                    '"payload": "AAAA", "compilation_cmd": "gcc -g"}')

        with patch.object(adapter, "_call_llm", side_effect=mock_call_llm):
            with patch.object(
                ExploitAdapter, "_validate_syntax_detailed",
                side_effect=[(False, "error: expected ';'"), (True, "")]
            ):
                result = adapter._adapt_with_llm_retry(
                    "void f(){}", {}, "buffer_overflow", "c"
                )

        assert call_count == 2
        assert "expected ';'" in prompts_seen[1]
        assert result is not None
        assert result.is_valid

    def test_all_retries_fail_returns_none(self):
        adapter = ExploitAdapter(llm_backend="api", api_url="http://fake",
                                 max_retries=2)

        with patch.object(adapter, "_call_llm", return_value='not json'):
            result = adapter._adapt_with_llm_retry(
                "void f(){}", {}, "buffer_overflow", "c"
            )

        assert result is None

    def test_llm_exception_triggers_retry(self):
        adapter = ExploitAdapter(llm_backend="api", api_url="http://fake",
                                 max_retries=2)
        calls = []

        def mock_call(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                raise ConnectionError("timeout")
            return ('{"harness_code": "#include <stdio.h>\\nint main(){return 0;}", '
                    '"payload": "AA", "compilation_cmd": "gcc"}')

        with patch.object(adapter, "_call_llm", side_effect=mock_call):
            with patch.object(
                ExploitAdapter, "_validate_syntax_detailed",
                return_value=(True, "")
            ):
                result = adapter._adapt_with_llm_retry(
                    "void f(){}", {}, "buffer_overflow", "c"
                )

        assert len(calls) == 2
        assert result is not None


class TestMultiHarness:
    def test_adapt_multi_basic(self):
        adapter = ExploitAdapter()
        vuln_code = "void proc(const char *s) { char b[8]; strcpy(b, s); }"
        templates = [
            {
                "exploit_id": "T1",
                "harness_template": '#include <stdio.h>\n{vulnerable_code}\nint main() { {func_name}("AAAA"); return 0; }',
                "payload_template": "AAAA",
                "expected_signals": ["AddressSanitizer"],
            },
            {
                "exploit_id": "T2",
                "harness_template": '#include <stdio.h>\n{vulnerable_code}\nint main() { {func_name}("BBBB"); return 0; }',
                "payload_template": "BBBB",
                "expected_signals": ["AddressSanitizer"],
            },
        ]
        harnesses = adapter.adapt_multi(vuln_code, templates, "buffer_overflow")
        assert len(harnesses) >= 1
        assert all(h.is_valid for h in harnesses)

    def test_adapt_multi_deduplicates(self):
        adapter = ExploitAdapter()
        vuln_code = "void f(const char *s) { char b[8]; strcpy(b, s); }"
        same_template = {
            "exploit_id": "T1",
            "harness_template": '#include <stdio.h>\n{vulnerable_code}\nint main() { {func_name}("A"); return 0; }',
            "payload_template": "A",
        }
        harnesses = adapter.adapt_multi(
            vuln_code, [same_template, same_template, same_template],
            "buffer_overflow"
        )
        assert len(harnesses) == 1

    def test_adapt_multi_fallback_to_generic(self):
        adapter = ExploitAdapter()
        vuln_code = "void f(const char *s) { char b[8]; strcpy(b, s); }"
        harnesses = adapter.adapt_multi(vuln_code, [], "buffer_overflow")
        assert len(harnesses) == 1
        assert harnesses[0].is_valid

    def test_adapt_multi_respects_max(self):
        adapter = ExploitAdapter()
        vuln_code = "void f(const char *s) { char b[8]; strcpy(b, s); }"
        templates = [
            {
                "exploit_id": f"T{i}",
                "harness_template": f'#include <stdio.h>\n{{vulnerable_code}}\nint main() {{ {{func_name}}("{chr(65+i)}"); return 0; }}',
                "payload_template": chr(65 + i),
            }
            for i in range(10)
        ]
        harnesses = adapter.adapt_multi(
            vuln_code, templates, "buffer_overflow", max_harnesses=2
        )
        assert len(harnesses) <= 2


class TestCaching:
    def test_cache_hit(self):
        adapter = ExploitAdapter(llm_backend="api", api_url="http://fake",
                                 cache_results=True)
        vuln_code = "void f() {}"
        template = {"exploit_id": "T1"}

        cached = ExploitHarness(
            harness_code="cached code", payload="cached",
            compilation_cmd="gcc"
        )
        key = adapter._cache_key(vuln_code, "buffer_overflow", "T1")
        adapter._cache[key] = cached

        result = adapter.adapt(vuln_code, template, "buffer_overflow")
        assert result.harness_code == "cached code"

    def test_cache_key_deterministic(self):
        k1 = ExploitAdapter._cache_key("code", "type", "id")
        k2 = ExploitAdapter._cache_key("code", "type", "id")
        assert k1 == k2

    def test_cache_key_varies_by_input(self):
        k1 = ExploitAdapter._cache_key("code1", "type", "id")
        k2 = ExploitAdapter._cache_key("code2", "type", "id")
        assert k1 != k2


class TestSyntaxValidation:
    def test_validate_syntax_detailed_returns_tuple(self):
        h = ExploitHarness(
            harness_code="#include <stdio.h>\nint main() { return 0; }",
            payload="A", compilation_cmd="gcc"
        )
        result = ExploitAdapter._validate_syntax_detailed(h, "c")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_validate_non_c_always_valid(self):
        h = ExploitHarness(
            harness_code="not real code", payload="A", compilation_cmd=""
        )
        valid, msg = ExploitAdapter._validate_syntax_detailed(h, "python")
        assert valid is True


class TestParseResponse:
    def test_parse_json_response(self):
        adapter = ExploitAdapter()
        response = '{"harness_code": "int main(){}", "payload": "test", "compilation_cmd": "gcc"}'
        result = adapter._parse_llm_response(response, "buffer_overflow", "E1")
        assert result.harness_code == "int main(){}"
        assert result.vuln_type == "buffer_overflow"
        assert result.source_exploit_id == "E1"

    def test_parse_code_block_fallback(self):
        adapter = ExploitAdapter()
        response = "Here's the code:\n```c\n#include <stdio.h>\nint main() { return 0; }\n```"
        result = adapter._parse_llm_response(response, "buffer_overflow", "")
        assert result.is_valid
        assert "int main" in result.harness_code

    def test_parse_garbage_returns_empty(self):
        adapter = ExploitAdapter()
        result = adapter._parse_llm_response("garbage output", "bo", "")
        assert not result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
