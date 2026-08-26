"""
Tests for Module 0 (interprocedural inlining).

No GPU, no Docker, no trained model, no network. The tokenizer is disabled
throughout so the suite does not depend on transformers being installed;
token counts become estimates, which is fine because every assertion here is
about structure, not about exact token numbers.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from interproc import InliningConfig, InterprocPipeline           # noqa: E402
from interproc import clex                                        # noqa: E402
from interproc.parsers.lex_backend import LexerBackend, parse_params  # noqa: E402

PROJECTS = os.path.join(os.path.dirname(__file__), "..",
                        "test_samples", "projects")


def project(name: str) -> str:
    return os.path.normpath(os.path.join(PROJECTS, name))


def build(name: str, depth: int = 2, **kwargs):
    config = InliningConfig(max_depth=depth, use_real_tokenizer=False,
                            use_cache=False, **kwargs)
    pipeline = InterprocPipeline(config)
    units = pipeline.build(project(name))
    return pipeline, {u.root.name: u for u in units}


def has_gcc() -> bool:
    try:
        subprocess.run(["gcc", "--version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


# ======================================================================
# Lexical scanner
# ======================================================================

class TestCodeMask:
    def test_string_contents_are_not_code(self):
        text = 'char *s = "a { b ) c";'
        mask = clex.code_mask(text)
        brace = text.index('{')
        assert mask[brace] != clex.CODE
        assert mask[0] == clex.CODE

    def test_comments_are_not_code(self):
        text = "int a; /* } ) */ int b;"
        mask = clex.code_mask(text)
        assert mask[text.index('}')] != clex.CODE

    def test_line_comment(self):
        text = "int a; // ) } noise\nint b;"
        mask = clex.code_mask(text)
        assert mask[text.index(')')] != clex.CODE
        assert mask[text.rindex("int")] == clex.CODE

    def test_preprocessor_line_with_continuation(self):
        text = "#define X(a) \\\n    foo(a)\nint b;"
        mask = clex.code_mask(text)
        assert mask[text.index("foo")] == clex.PREPROC
        assert mask[text.rindex("int")] == clex.CODE

    def test_escaped_quote_inside_string(self):
        text = r'char *s = "he said \" }"; int after;'
        mask = clex.code_mask(text)
        assert mask[text.index("int after")] == clex.CODE

    def test_brace_matching_ignores_literals(self):
        text = 'void f() { char *s = "}"; }'
        mask = clex.code_mask(text)
        open_idx = text.index('{')
        assert clex.match_pair(text, mask, open_idx, '{', '}') == len(text)

    def test_unbalanced_braces_report_failure(self):
        text = "void f() { int a;"
        mask = clex.code_mask(text)
        assert clex.match_pair(text, mask, text.index('{'), '{', '}') == -1

    def test_split_top_level_respects_nesting(self):
        assert clex.split_top_level("a, f(b, c), d") == ["a", " f(b, c)", " d"]

    def test_substitute_skips_strings(self):
        out = clex.substitute_identifiers('buf; "buf"; buf', {"buf": "x"})
        assert out == 'x; "buf"; x'


class TestParamParsing:
    def test_simple(self):
        params, vararg = parse_params("char *dst, const char *src")
        assert [p.name for p in params] == ["dst", "src"]
        assert not vararg

    def test_void_and_empty(self):
        assert parse_params("void")[0] == []
        assert parse_params("")[0] == []

    def test_vararg_detected(self):
        params, vararg = parse_params("const char *fmt, ...")
        assert vararg and [p.name for p in params] == ["fmt"]

    def test_array_parameter(self):
        params, _ = parse_params("char buf[64]")
        assert params[0].name == "buf"

    def test_unnamed_parameter_gets_placeholder(self):
        params, _ = parse_params("int, char *")
        assert len(params) == 2 and all(p.name for p in params)


class TestFunctionExtraction:
    def parse(self, source: str):
        return LexerBackend().parse("/tmp/x.c", "x.c", source, "c")

    def test_finds_definitions_not_prototypes(self):
        facts = self.parse("int f(int a);\nint g(int a) { return a; }\n")
        assert [fn.name for fn in facts.functions] == ["g"]
        assert "f" in facts.prototypes

    def test_static_flag(self):
        facts = self.parse("static void h(void) { }\n")
        assert facts.functions[0].is_static

    def test_nested_braces_and_strings(self):
        facts = self.parse('void f(void) { if (1) { char *s = "}"; } }\nint z;\n')
        assert len(facts.functions) == 1
        assert facts.functions[0].func_text.endswith("}")

    def test_call_sites_recorded(self):
        facts = self.parse("void f(void) { g(1); h(k(2)); }\n")
        names = [c.callee_name for c in facts.functions[0].calls]
        assert names == ["g", "h", "k"]

    def test_keywords_are_not_calls(self):
        facts = self.parse("void f(int n) { if (n) { while (n) { n--; } } }\n")
        assert facts.functions[0].calls == []

    def test_result_used_detection(self):
        facts = self.parse("void f(void) { g(); int x = h(); }\n")
        by_name = {c.callee_name: c for c in facts.functions[0].calls}
        assert by_name["g"].result_used is False
        assert by_name["h"].result_used is True

    def test_offsets_are_relative_to_func_text(self):
        facts = self.parse("int pad;\nvoid f(void) { g(); }\n")
        fn = facts.functions[0]
        site = fn.calls[0]
        assert fn.func_text[site.start:site.end] == "g()"

    def test_flags(self):
        facts = self.parse("void f(void) { goto end; end: return; }\n")
        assert "has_goto" in facts.functions[0].flags
        assert "has_label" in facts.functions[0].flags

    def test_truncated_body_is_recovered_and_flagged(self):
        """Devign records really do arrive with no closing brace."""
        facts = self.parse("int f(int a) {\n    if (a) {\n        return 1;\n")
        assert [fn.name for fn in facts.functions] == ["f"]
        assert "truncated_body" in facts.functions[0].flags

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..",
                                        "dataset", "train.jsonl")),
        reason="Devign dataset not present")
    def test_parse_rate_on_real_corpus(self):
        path = os.path.join(os.path.dirname(__file__), "..", "dataset",
                            "train.jsonl")
        backend = LexerBackend()
        parsed = 0
        total = 0
        with open(path) as fh:
            for i, line in enumerate(fh):
                if i >= 200:
                    break
                total += 1
                facts = backend.parse("/tmp/x.c", "x.c",
                                      json.loads(line)["func"], "c")
                parsed += len(facts.functions) >= 1
        assert parsed / total >= 0.98, f"parse rate {parsed}/{total}"


# ======================================================================
# Call graph resolution
# ======================================================================

class TestCallGraph:
    def test_cross_file_resolution_through_header(self):
        pipeline, _ = build("p01_simple")
        summary = pipeline.graph.summary()
        assert summary.get("exact_include", 0) >= 2
        assert summary.get("never_inline", 0) >= 2  # strcpy/printf & friends

    def test_static_stays_in_its_translation_unit(self):
        pipeline, units = build("p05_static_dup")
        # two same-named statics must resolve to their own file, never across
        for root_name, expected_file in (("run_a", "src/a.c"), ("run_b", "src/b.c")):
            unit = units[root_name]
            assert len(unit.inlined) == 1
            assert unit.inlined[0].callee.startswith(expected_file)

    def test_library_calls_are_never_inlined(self):
        _, units = build("p01_simple")
        reasons = {(s.callee_name, s.reason)
                   for s in units["handle_request"].skipped}
        assert ("strlen", "never_inline") in reasons
        assert ("printf", "never_inline") in reasons


# ======================================================================
# Expansion
# ======================================================================

class TestExpansion:
    def test_depth_zero_reproduces_the_original_function(self):
        _, units = build("p01_simple", depth=0)
        source = open(os.path.join(project("p01_simple"), "src", "main.c")).read()
        original = source[source.index("void handle_request"):].strip()
        assert units["handle_request"].code_for_gnn.strip() == original
        assert units["handle_request"].inlined == []

    def test_cross_file_body_becomes_visible(self):
        """The whole point: buf[64] and strcpy end up in one token window."""
        _, units = build("p01_simple", depth=1)
        code = units["handle_request"].code_for_gnn
        assert "char buf[64]" in code
        assert "strcpy" in code            # came from src/util.c
        assert "src/util.c" in units["handle_request"].provenance_files

    def test_depth_increases_or_keeps_size(self):
        _, u0 = build("p01_simple", depth=0)
        _, u1 = build("p01_simple", depth=1)
        _, u2 = build("p01_simple", depth=2)
        sizes = [u[("handle_request")].tokens_gnn for u in (u0, u1, u2)]
        assert sizes[0] < sizes[1] <= sizes[2]

    def test_budget_is_never_exceeded(self):
        for depth in (1, 2, 3):
            _, units = build("p01_simple", depth=depth, max_tokens=80)
            for unit in units.values():
                if not unit.truncated:
                    assert unit.tokens_gnn <= 80

    def test_oversized_callee_is_skipped_but_small_one_is_kept(self):
        _, units = build("p03_budget", depth=1)
        entry = units["entry"]
        inlined = {r.callee_name for r in entry.inlined}
        skipped = {(s.callee_name, s.reason) for s in entry.skipped}
        assert "tiny_helper" in inlined
        assert ("huge_helper", "callee_too_big") in skipped

    def test_direct_recursion_is_refused(self):
        _, units = build("p02_recursion", depth=3)
        assert {s.reason for s in units["fact"].skipped} == {"recursion"}
        assert units["fact"].inlined == []

    def test_mutual_recursion_is_refused(self):
        _, units = build("p02_recursion", depth=3)
        assert any(s.reason == "recursion_scc" for s in units["ping"].skipped)

    def test_repeated_callee_is_capped(self):
        _, units = build("p01_simple", depth=2, max_expansions_per_callee=1)
        for unit in units.values():
            counts = {}
            for rec in unit.inlined:
                counts[rec.callee] = counts.get(rec.callee, 0) + 1
            assert all(c <= 1 for c in counts.values())

    def test_name_collisions_are_renamed(self):
        _, units = build("p04_collision", depth=1)
        code = units["caller"].code_for_gnn
        assert "acc_i" in code                 # callee local renamed
        assert "int acc = 100;" in code        # caller's own name untouched

    def test_no_renaming_without_collision(self):
        _, units = build("p01_simple", depth=1)
        assert "_i1" not in units["handle_request"].code_for_gnn

    def test_return_is_rewritten_to_an_assignment(self):
        _, units = build("p04_collision", depth=1)
        code = units["caller"].code_for_gnn
        assert "__r_transform =" in code
        assert "return transform" not in code

    def test_markers_are_absent_by_default(self):
        _, units = build("p01_simple", depth=2)
        assert "/* inline" not in units["handle_request"].code_for_gnn

    def test_markers_can_be_enabled(self):
        _, units = build("p01_simple", depth=2, debug_markers=True)
        assert "/* inline" in units["handle_request"].code_for_gnn

    def test_determinism(self):
        _, a = build("p01_simple", depth=2)
        _, b = build("p01_simple", depth=2)
        assert {k: v.code_for_gnn for k, v in a.items()} == \
               {k: v.code_for_gnn for k, v in b.items()}
        assert {k: v.unit_id for k, v in a.items()} == \
               {k: v.unit_id for k, v in b.items()}

    def test_unit_id_changes_with_depth(self):
        _, a = build("p01_simple", depth=1)
        _, b = build("p01_simple", depth=2)
        assert a["handle_request"].unit_id != b["handle_request"].unit_id


class TestStrategies:
    def test_priority_prefers_the_sink_carrying_callee(self):
        _, units = build("p03_budget", depth=1, strategy="priority")
        assert units["entry"].inlined[0].callee_name == "tiny_helper"

    @pytest.mark.parametrize("strategy", ["priority", "dfs", "bfs"])
    def test_every_strategy_produces_valid_units(self, strategy):
        _, units = build("p01_simple", depth=2, strategy=strategy)
        assert units["handle_request"].code_for_gnn.strip()


# ======================================================================
# Root selection
# ======================================================================

class TestRootSelection:
    def test_every_function_becomes_a_root(self):
        _, units = build("p01_simple")
        assert set(units) == {"handle_request", "copy_data", "checksum",
                              "log_line"}

    def test_short_function_with_a_sink_is_still_a_root(self):
        """Guards decision D-005: triviality is not about length."""
        _, units = build("p01_simple", root_min_tokens=1000)
        assert "copy_data" in units          # one-liner, but it calls strcpy

    def test_trivial_accessor_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.c"), "w") as fh:
                fh.write("int get(void) { return 1; }\n"
                         "void use(char *d, const char *s) { strcpy(d, s); }\n")
            config = InliningConfig(use_real_tokenizer=False, use_cache=False)
            units = {u.root.name for u in InterprocPipeline(config).build(tmp)}
        assert units == {"use"}


# ======================================================================
# Sandbox bundle
# ======================================================================

class TestBundle:
    def test_bundle_contains_callee_definitions(self):
        _, units = build("p01_simple", depth=1)
        bundle = units["handle_request"].code_for_sandbox
        assert "void copy_data(" in bundle
        assert "int checksum(" in bundle
        assert "#include <string.h>" in bundle

    def test_bundle_is_not_inlined(self):
        _, units = build("p01_simple", depth=2)
        assert "copy_data(buf, user_input);" in \
               units["handle_request"].code_for_sandbox

    def test_bundle_independent_of_depth(self):
        _, u0 = build("p01_simple", depth=0)
        _, u2 = build("p01_simple", depth=2)
        assert u0["handle_request"].code_for_sandbox == \
               u2["handle_request"].code_for_sandbox

    @pytest.mark.skipif(not has_gcc(), reason="gcc not available")
    def test_bundle_compiles(self):
        _, units = build("p01_simple", depth=2)
        for name, unit in units.items():
            with tempfile.NamedTemporaryFile("w", suffix=".c",
                                             delete=False) as fh:
                fh.write(unit.code_for_sandbox)
                path = fh.name
            try:
                result = subprocess.run(["gcc", "-fsyntax-only", "-w", path],
                                        capture_output=True, text=True,
                                        timeout=30)
                assert result.returncode == 0, \
                    f"{name} did not compile:\n{result.stderr}"
            finally:
                os.unlink(path)


# ======================================================================
# Serialisation, source map, CLI
# ======================================================================

class TestOutput:
    def test_jsonl_record_matches_dataset_shape(self):
        _, units = build("p01_simple", depth=1)
        record = units["handle_request"].to_jsonl_record(idx=3)
        assert record["idx"] == 3
        assert record["func"] == units["handle_request"].code_for_gnn
        assert record["target"] == -1
        assert record["unit"]["file"] == "src/main.c"

    def test_source_map_covers_the_whole_text(self):
        _, units = build("p01_simple", depth=2)
        unit = units["handle_request"]
        assert sum(s.text.count('\n') for s in unit.segments) == \
               unit.code_for_gnn.count('\n')
        assert all(m["origin_file"] for m in unit.source_map())

    def test_source_map_attributes_inlined_code_to_its_file(self):
        _, units = build("p01_simple", depth=1)
        origins = {m["origin_file"] for m in units["handle_request"].source_map()}
        assert origins == {"src/main.c", "src/util.c"}


class TestCli:
    def test_end_to_end(self, tmp_path):
        from interproc.cli import main
        out = tmp_path / "units.jsonl"
        stats = tmp_path / "stats.json"
        code = main(["--project", project("p01_simple"),
                     "--inline-depth", "2",
                     "--out", str(out), "--stats", str(stats),
                     "--no-tokenizer", "--no-cache"])
        assert code == 0

        records = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(records) == 4
        assert any("strcpy" in r["func"] and "buf[64]" in r["func"]
                   for r in records)

        summary = json.loads(stats.read_text())
        assert summary["functions_found"] == 4
        assert summary["units_with_inlining"] == 1

    def test_missing_project_exits_cleanly(self, tmp_path):
        from interproc.cli import main
        assert main(["--project", str(tmp_path / "nope"), "--no-tokenizer"]) == 2

    def test_empty_project_exits_cleanly(self, tmp_path):
        from interproc.cli import main
        (tmp_path / "readme.txt").write_text("nothing here")
        assert main(["--project", str(tmp_path), "--no-tokenizer"]) == 3


class TestRobustness:
    def _build_dir(self, tmp_path, files):
        for name, content in files.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        config = InliningConfig(use_real_tokenizer=False, use_cache=False)
        return InterprocPipeline(config).build(str(tmp_path))

    def test_truncated_file_does_not_raise(self, tmp_path):
        self._build_dir(tmp_path, {"a.c": "void f(void) { int x = 1;"})

    def test_truncated_callee_is_not_inlined(self, tmp_path):
        units = self._build_dir(tmp_path, {
            "a.c": "void top(const char *s) { char b[8]; broken(b, s); }\n"
                   "void broken(char *d, const char *s) { strcpy(d, s);\n",
        })
        top = [u for u in units if u.root.name == "top"][0]
        assert top.inlined == []
        assert ("broken", "truncated_body") in \
               {(s.callee_name, s.reason) for s in top.skipped}

    def test_crlf_offsets_stay_correct(self, tmp_path):
        units = self._build_dir(tmp_path, {
            "a.c": "void helper(char *d, const char *s)\r\n{\r\n"
                   "    strcpy(d, s);\r\n}\r\n"
                   "void top(const char *in)\r\n{\r\n"
                   "    char b[8];\r\n    helper(b, in);\r\n}\r\n",
        })
        top = [u for u in units if u.root.name == "top"][0]
        assert "strcpy" in top.code_for_gnn
        assert "\r" not in top.code_for_gnn

    def test_binary_file_is_skipped(self, tmp_path):
        (tmp_path / "junk.c").write_bytes(b"\x00\x01\x02binary")
        (tmp_path / "ok.c").write_text(
            "void f(char *d, const char *s) { strcpy(d, s); }\n")
        config = InliningConfig(use_real_tokenizer=False, use_cache=False)
        units = InterprocPipeline(config).build(str(tmp_path))
        assert [u.root.name for u in units] == ["f"]

    def test_unicode_identifiers_and_comments(self, tmp_path):
        self._build_dir(tmp_path, {
            "a.c": "/* коммент */\nvoid f(char *d, const char *s)"
                   " { strcpy(d, s); /* ещё */ }\n",
        })
