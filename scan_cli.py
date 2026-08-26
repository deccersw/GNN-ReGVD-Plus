#!/usr/bin/env python3
"""
CLI interface for the GNN-ReGVD vulnerability scanner.

Usage:
    python scan_cli.py --code "void f(char *s){char b[8];strcpy(b,s);}"
    python scan_cli.py --file vuln.c
    python scan_cli.py --jsonl dataset/test.jsonl --limit 10
    python scan_cli.py --file vuln.c --sandbox-only
"""

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from scanner.config import ScannerConfig
from scanner.pipeline import VulnerabilityScanner, ScanResult
from scanner.report import format_report, to_json, batch_summary


def parse_args():
    p = argparse.ArgumentParser(
        description="GNN-ReGVD Vulnerability Scanner"
    )

    input_group = p.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--code", type=str,
                             help="C/C++ code snippet to scan")
    input_group.add_argument("--file", type=str,
                             help="Path to source file to scan")
    input_group.add_argument("--jsonl", type=str,
                             help="Path to JSONL dataset for batch scanning")
    input_group.add_argument("--project", type=str,
                             help="Path to a C/C++ project directory "
                                  "(Module 0 splits it into analysis units)")

    p.add_argument("--sandbox-only", action="store_true",
                   help="Run sandbox verification only (skip GNN)")

    # --- Module 0: interprocedural inlining -------------------------------
    g = p.add_argument_group("Module 0: interprocedural inlining")
    g.add_argument("--build-units-only", action="store_true",
                   help="Only split the project into inlined analysis units "
                        "and stop: no GNN, no LLM, no sandbox. Use with "
                        "--project to inspect the split on its own.")
    g.add_argument("--inline-depth", type=int, default=2,
                   help="How deep to inline callees into each function: "
                        "0 = no inlining (current single-function behaviour), "
                        "1 = direct callees, 2 = also their callees "
                        "(default: 2)")
    g.add_argument("--inline-strategy", type=str, default="priority",
                   choices=["priority", "dfs", "bfs"],
                   help="Order in which call sites spend the token budget "
                        "(default: priority — sinks and tainted arguments first)")
    g.add_argument("--inline-max-callee-tokens", type=int, default=200,
                   help="Largest callee that may be inlined (default: 200)")
    g.add_argument("--inline-parser", type=str, default="auto",
                   choices=["auto", "treesitter", "lexer"],
                   help="Parser backend for Module 0 (default: auto)")
    g.add_argument("--inline-markers", action="store_true",
                   help="Emit /* inline f */ comments in the GNN view (debug; "
                        "they cost tokens from the detector's window)")
    g.add_argument("--max-functions", type=int, default=0,
                   help="Analyse at most N functions of the project (0 = all)")
    g.add_argument("--units-out", type=str, default=None,
                   help="Write the analysis units to this JSONL file")
    g.add_argument("--units-stats", type=str, default=None,
                   help="Write Module 0 build statistics to this JSON file")

    p.add_argument("--model-path", type=str, default="",
                   help="Path to trained GNN model checkpoint")
    p.add_argument("--faiss-dir", type=str, default="",
                   help="Path to FAISS index directory")
    p.add_argument("--exploit-db-dir", type=str, default="",
                   help="Path to exploit DB directory")

    p.add_argument("--llm-backend", type=str, default=None,
                   choices=["transformers", "api"],
                   help="LLM backend for exploit adaptation")
    p.add_argument("--llm-model", type=str,
                   default="Qwen/Qwen2.5-Coder-7B-Instruct",
                   help="LLM model name")
    p.add_argument("--llm-api-url", type=str, default=None,
                   help="API URL for LLM backend")

    p.add_argument("--threshold", type=float, default=0.5,
                   help="Detection threshold (default: 0.5)")
    p.add_argument("--max-exploits", type=int, default=3,
                   help="Max exploit attempts per sample (default: 3)")
    p.add_argument("--sandbox-timeout", type=int, default=30,
                   help="Sandbox timeout in seconds (default: 30)")
    p.add_argument("--use-valgrind", action="store_true",
                   help="Enable Valgrind in sandbox")
    p.add_argument("--multi-sanitizer", action="store_true",
                   help="Enable multi-sanitizer mode (ASan+UBSan, MSan, TSan)")
    p.add_argument("--use-libfuzzer", action="store_true",
                   help="Enable LibFuzzer for automatic payload discovery")
    p.add_argument("--fuzz-time", type=int, default=15,
                   help="LibFuzzer run time in seconds (default: 15)")
    p.add_argument("--taint-analysis", action="store_true",
                   help="Enable taint analysis (Joern/heuristic)")
    p.add_argument("--joern-path", type=str, default=None,
                   help="Path to Joern binary (auto-detected if not set)")
    p.add_argument("--prefer-clang", action="store_true",
                   help="Prefer clang over gcc for compilation")

    p.add_argument("--triage", action="store_true",
                   help="Enable LLM triage head (requires OPENROUTER_API_KEY)")
    p.add_argument("--triage-model", type=str,
                   default="openrouter/anthropic/claude-sonnet-4-6",
                   help="LLM model for triage (default: claude-sonnet-4-6)")
    p.add_argument("--triage-only", action="store_true",
                   help="Run triage only (skip GNN and sandbox)")
    p.add_argument("--dry-run", action="store_true",
                   help="Run full pipeline without sandbox/triage API calls "
                        "(uses stubs, for testing pipeline flow)")

    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of samples for batch scanning")
    p.add_argument("--output", type=str, default=None,
                   help="Output file for results (JSON)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose output")
    p.add_argument("--json", action="store_true",
                   help="Output as JSON")

    return p.parse_args()


def build_config(args) -> ScannerConfig:
    return ScannerConfig(
        model_path=args.model_path,
        faiss_dir=args.faiss_dir,
        exploit_db_dir=args.exploit_db_dir,
        llm_backend=args.llm_backend,
        llm_model_name=args.llm_model,
        llm_api_url=args.llm_api_url,
        detection_threshold=args.threshold,
        max_exploits_per_sample=args.max_exploits,
        sandbox_timeout=args.sandbox_timeout,
        sandbox_use_valgrind=args.use_valgrind,
        sandbox_use_multi_sanitizer=args.multi_sanitizer,
        sandbox_use_libfuzzer=args.use_libfuzzer,
        sandbox_fuzz_time=args.fuzz_time,
        sandbox_prefer_clang=args.prefer_clang,
        use_taint_analysis=args.taint_analysis,
        joern_path=getattr(args, "joern_path", None),
        use_triage=getattr(args, "triage", False),
        triage_model=getattr(args, "triage_model",
                             "openrouter/anthropic/claude-sonnet-4-6"),
        dry_run=getattr(args, "dry_run", False),
        use_inlining=bool(getattr(args, "project", None)),
        inline_max_depth=getattr(args, "inline_depth", 2),
        inline_strategy=getattr(args, "inline_strategy", "priority"),
        inline_max_callee_tokens=getattr(args, "inline_max_callee_tokens", 200),
        inline_parser_backend=getattr(args, "inline_parser", "auto"),
        inline_debug_markers=getattr(args, "inline_markers", False),
        inline_max_roots=getattr(args, "max_functions", 0),
    )


def build_units(args, config):
    """Run Module 0 over a project directory and return (units, pipeline)."""
    from interproc import InliningConfig, InterprocPipeline

    logger = logging.getLogger(__name__)
    inline_config = InliningConfig.from_scanner_config(config)
    logger.info(
        "[Module 0] Building analysis units: project=%s, depth=%d, "
        "strategy=%s, budget=%d tokens, parser=%s",
        args.project, inline_config.max_depth, inline_config.strategy,
        inline_config.max_tokens, inline_config.parser_backend,
    )

    pipeline = InterprocPipeline(inline_config)
    pipeline.analyze(args.project)
    if not pipeline.index.functions:
        raise SystemExit(
            f"Error: no C/C++ functions found in {args.project}")

    units = pipeline.build(args.project)
    logger.info(
        "[Module 0] %d units built from %d functions "
        "(inlining applied in %d, truncated %d, avg growth x%.2f)",
        len(units), pipeline.stats.functions_found,
        pipeline.stats.units_with_inlining, pipeline.stats.units_truncated,
        pipeline.stats.avg_growth,
    )
    return units, pipeline


def build_units_only_mode(args):
    """--build-units-only: split the project and stop before the scanner."""
    from interproc.cli import _summary
    from interproc.units import write_units_jsonl

    config = build_config(args)
    units, pipeline = build_units(args, config)

    if args.units_out:
        write_units_jsonl(units, args.units_out)
        print(f"\n{len(units)} units written to {args.units_out}")

    stats = pipeline.stats.to_dict()
    if args.units_stats:
        with open(args.units_stats, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics written to {args.units_stats}")

    if args.json:
        print(json.dumps({
            "stats": stats,
            "units": [u.to_jsonl_record(i) for i, u in enumerate(units)],
        }, indent=2, ensure_ascii=False))
        return units

    print(_summary(stats, len(units)))

    ranked = sorted(units, key=lambda u: -len(u.inlined))[:10]
    if ranked and ranked[0].inlined:
        print("\nUnits with the most inlined context:")
        for unit in ranked:
            if not unit.inlined:
                break
            callees = ", ".join(r.callee_name for r in unit.inlined[:4])
            print(f"  {str(unit.root):<48} {len(unit.inlined)} inlined "
                  f"({callees}) -> {unit.tokens_gnn} tok"
                  f"{'  [TRUNCATED]' if unit.truncated else ''}")

    if args.verbose:
        for unit in units:
            print(f"\n--- {unit.root} ---")
            print(unit.code_for_gnn)

    return units


def scan_project(scanner, args, config):
    """Scan every unit of a project through the existing pipeline."""
    units, pipeline = build_units(args, config)
    results = []

    for i, unit in enumerate(units):
        print(f"\n--- Unit {i+1}/{len(units)}: {unit.root} "
              f"({len(unit.inlined)} inlined, {unit.tokens_gnn} tok) ---")
        result = scanner.scan(unit.code_for_gnn)
        result.unit_id = unit.unit_id
        result.file = unit.root.file
        result.function = unit.root.name
        result.start_line = unit.root.start_line
        result.end_line = unit.root.start_line
        result.inline_depth_used = unit.depth_used
        result.inlined_functions = [r.callee_name for r in unit.inlined]
        result.provenance_files = unit.provenance_files
        result.inline_truncated = unit.truncated

        if args.json:
            print(to_json(result))
        else:
            print(format_report(result, verbose=args.verbose))
        results.append(result)

    print("\n" + batch_summary(results))
    flagged = [r for r in results if r.verdict in ("CONFIRMED", "SUGGESTIVE")]
    if flagged:
        print("\nFlagged units:")
        for r in sorted(flagged, key=lambda r: -r.confidence):
            extra = f" via {', '.join(r.inlined_functions)}" \
                if r.inlined_functions else ""
            print(f"  [{r.verdict}] {r.file}:{r.function}:{r.start_line} "
                  f"({r.confidence:.0%}){extra}")
        print("\nNote: findings are reported per unit. Cross-unit "
              "deduplication is not implemented yet, so a vulnerability in a "
              "callee may appear both on its own and inside its callers.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([json.loads(to_json(r)) for r in results], f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


def scan_single(scanner, code: str, args):
    result = scanner.scan(code)
    if args.json:
        print(to_json(result))
    else:
        print(format_report(result, code_snippet=code, verbose=args.verbose))
    return result


def scan_file(scanner, path: str, args):
    with open(path) as f:
        code = f.read()
    return scan_single(scanner, code, args)


def scan_jsonl(scanner, path: str, args):
    results = []
    with open(path) as f:
        lines = f.readlines()

    if args.limit:
        lines = lines[:args.limit]

    for i, line in enumerate(lines):
        data = json.loads(line)
        code = data.get("func", data.get("code", ""))
        label = data.get("target", data.get("label", -1))

        if not code.strip():
            continue

        print(f"\n--- Sample {i+1}/{len(lines)} (label={label}) ---")
        result = scanner.scan(code)

        if args.json:
            print(to_json(result))
        else:
            print(format_report(result, verbose=args.verbose))

        results.append({
            "index": i,
            "label": label,
            **json.loads(to_json(result)),
        })

    print("\n" + batch_summary([
        ScanResult(**{k: v for k, v in r.items()
                      if k in ScanResult.__dataclass_fields__})
        for r in results
    ]))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


def triage_only_mode(args):
    """Run LLM triage directly on code without GNN detection or sandbox."""
    from triage import TriageHead, TriageConfig

    logger = logging.getLogger(__name__)

    code = args.code
    source_name = "<inline>"
    if not code and args.file:
        with open(args.file) as f:
            code = f.read()
        source_name = args.file

    if not code:
        print("Error: --code or --file required with --triage-only")
        sys.exit(1)

    logger.info(
        "[triage-only] Starting: source=%s, code=%d lines, %d chars",
        source_name, code.count("\n") + 1, len(code),
    )

    # Optional: run taint analysis for Joern context
    taint_context = ""
    if args.taint_analysis:
        from analysis.taint import TaintAnalyzer
        logger.info("[triage-only] Running taint analysis...")
        analyzer = TaintAnalyzer(joern_path=getattr(args, "joern_path", None))
        try:
            taint_result = analyzer.analyze(code)
            taint_context = taint_result.format_for_llm()
            logger.info(
                "[triage-only] Taint analysis done: method=%s, "
                "paths=%d, context=%d chars",
                taint_result.analysis_method,
                len(taint_result.taint_paths),
                len(taint_context),
            )
            if taint_context:
                print(f"Joern context: {taint_result.analysis_method} "
                      f"({len(taint_result.taint_paths)} taint paths)")
        except Exception as e:
            logger.error("[triage-only] Taint analysis FAILED: %s", e)
            print(f"Taint analysis failed: {e}")
    else:
        logger.debug("[triage-only] Taint analysis: skipped (not enabled)")

    config = TriageConfig(
        enabled=True,
        model=args.triage_model,
    )
    logger.info(
        "[triage-only] TriageHead config: model=%s", config.model,
    )
    head = TriageHead(config)

    print("Running LLM triage...")
    logger.info("[triage-only] Calling LLM for eval triage...")
    verdict = head.triage_eval_sync(
        source_code=code,
        taint_context=taint_context,
    )

    logger.info(
        "[triage-only] Result: verdict=%s, confidence=%.2f",
        verdict.verdict, verdict.confidence,
    )
    logger.debug("[triage-only] Reasoning: %s", verdict.reasoning[:300])

    print(f"\nVerdict:      {verdict.verdict}")
    print(f"Confidence:   {verdict.confidence:.2%}")
    print(f"Reasoning:    {verdict.reasoning}")

    if args.json:
        import json as json_mod
        print("\n" + json_mod.dumps({
            "triage_verdict": verdict.verdict,
            "triage_confidence": verdict.confidence,
            "triage_reasoning": verdict.reasoning,
        }, indent=2))


def sandbox_only_mode(args):
    """Run sandbox verification directly on code without GNN detection."""
    import re
    from sandbox.executor import SandboxExecutor, SandboxConfig
    from exploit_db.exploit_adapter import ExploitAdapter

    logger = logging.getLogger(__name__)

    code = args.code
    source_name = "<inline>"
    if not code and args.file:
        with open(args.file) as f:
            code = f.read()
        source_name = args.file

    if not code:
        print("Error: --code or --file required with --sandbox-only")
        sys.exit(1)

    has_main = bool(re.search(r'\bint\s+main\s*\(', code))
    logger.info(
        "[sandbox-only] Starting: source=%s, code=%d lines, "
        "has_main=%s, timeout=%ds",
        source_name, code.count("\n") + 1, has_main,
        args.sandbox_timeout,
    )

    executor = SandboxExecutor(SandboxConfig(
        timeout=args.sandbox_timeout,
        use_valgrind=args.use_valgrind,
        use_multi_sanitizer=args.multi_sanitizer,
        use_libfuzzer=args.use_libfuzzer,
        fuzz_time=args.fuzz_time,
        prefer_clang=args.prefer_clang,
    ))
    logger.info(
        "[sandbox-only] Executor config: valgrind=%s, multi_sanitizer=%s, "
        "libfuzzer=%s, prefer_clang=%s",
        args.use_valgrind, args.multi_sanitizer,
        args.use_libfuzzer, args.prefer_clang,
    )

    if has_main:
        logger.info(
            "[sandbox-only] Code has main() — compiling directly "
            "with ASan+UBSan"
        )
        result = executor.execute(
            harness_code=code,
            payload="",
            compilation_cmd="gcc -fsanitize=address,undefined -g",
            expected_signals=["AddressSanitizer", "runtime error", "SEGV"],
        )
    else:
        logger.info(
            "[sandbox-only] No main() — generating harness via "
            "ExploitAdapter"
        )
        adapter = ExploitAdapter()
        harness = adapter.adapt(
            vuln_code=code,
            exploit_template={},
            vuln_type="buffer_overflow",
        )
        logger.info(
            "[sandbox-only] Harness generated: %d lines, "
            "payload=%d chars, cmd=%s",
            harness.harness_code.count("\n") + 1,
            len(harness.payload),
            harness.compilation_cmd[:80],
        )
        result = executor.execute(
            harness_code=harness.harness_code,
            payload=harness.payload,
            compilation_cmd=harness.compilation_cmd,
            expected_signals=harness.expected_signals,
        )

    logger.info(
        "[sandbox-only] Execution done: verdict=%s, confidence=%.2f, "
        "exit_code=%s, compiled=%s, time=%.2fs",
        result.evidence.level.value, result.evidence.confidence,
        result.exit_code, result.harness_compiled,
        result.execution_time,
    )
    if result.evidence.asan_report:
        logger.info(
            "[sandbox-only] ASan report: error_type=%s",
            result.evidence.asan_report.error_type,
        )
    if result.evidence.cwe_id:
        logger.info("[sandbox-only] CWE: %s", result.evidence.cwe_id)

    print(f"Verdict:    {result.evidence.level.value}")
    print(f"Reason:     {result.evidence.reason}")
    print(f"Confidence: {result.evidence.confidence:.2%}")
    print(f"Exit code:  {result.exit_code}")
    print(f"Compiled:   {result.harness_compiled}")
    print(f"Time:       {result.execution_time:.2f}s")

    if result.evidence.asan_report:
        print(f"ASan error: {result.evidence.asan_report.error_type}")
    if result.evidence.cwe_id:
        print(f"CWE:        {result.evidence.cwe_id}")

    if args.verbose:
        if result.stdout:
            print(f"\nstdout:\n{result.stdout[:500]}")
        if result.stderr:
            print(f"\nstderr:\n{result.stderr[:500]}")


def main():
    args = parse_args()

    if not any([args.code, args.file, args.jsonl, args.project,
                args.sandbox_only, args.triage_only]):
        print("Error: one of --code, --file, --jsonl, --project, "
              "--sandbox-only, or --triage-only is required")
        sys.exit(1)

    if args.build_units_only and not args.project:
        print("Error: --build-units-only requires --project")
        sys.exit(1)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger(__name__)

    # Log active modes
    modes = []
    if args.build_units_only:
        modes.append("build-units-only")
    if args.project:
        modes.append(f"project(depth={args.inline_depth})")
    if args.sandbox_only:
        modes.append("sandbox-only")
    if getattr(args, "triage_only", False):
        modes.append("triage-only")
    if getattr(args, "dry_run", False):
        modes.append("DRY-RUN")
    if getattr(args, "triage", False):
        modes.append("triage")
    if getattr(args, "taint_analysis", False):
        modes.append("taint-analysis")
    if getattr(args, "multi_sanitizer", False):
        modes.append("multi-sanitizer")
    if getattr(args, "use_libfuzzer", False):
        modes.append("libfuzzer")
    if getattr(args, "use_valgrind", False):
        modes.append("valgrind")

    source = (args.code and "<inline>") or args.file or args.jsonl \
        or args.project or "N/A"
    logger.info(
        "[CLI] GNN-ReGVD scanner: source=%s, modes=[%s], verbose=%s",
        source, ", ".join(modes) if modes else "full-pipeline",
        args.verbose,
    )

    if args.build_units_only:
        build_units_only_mode(args)
        return

    if args.sandbox_only:
        sandbox_only_mode(args)
        return

    if args.triage_only:
        triage_only_mode(args)
        return

    config = build_config(args)
    logger.info(
        "[CLI] Full pipeline: threshold=%.2f, max_exploits=%d, "
        "triage=%s, taint=%s",
        config.detection_threshold, config.max_exploits_per_sample,
        config.use_triage, config.use_taint_analysis,
    )
    scanner = VulnerabilityScanner(config)

    if args.code:
        scan_single(scanner, args.code, args)
    elif args.file:
        scan_file(scanner, args.file, args)
    elif args.jsonl:
        scan_jsonl(scanner, args.jsonl, args)
    elif args.project:
        scan_project(scanner, args, config)


if __name__ == "__main__":
    main()
