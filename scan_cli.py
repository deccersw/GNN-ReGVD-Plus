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

    p.add_argument("--sandbox-only", action="store_true",
                   help="Run sandbox verification only (skip GNN)")

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
    )


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

    if not any([args.code, args.file, args.jsonl,
                args.sandbox_only, args.triage_only]):
        print("Error: one of --code, --file, --jsonl, "
              "--sandbox-only, or --triage-only is required")
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

    source = args.code and "<inline>" or args.file or args.jsonl or "N/A"
    logger.info(
        "[CLI] GNN-ReGVD scanner: source=%s, modes=[%s], verbose=%s",
        source, ", ".join(modes) if modes else "full-pipeline",
        args.verbose,
    )

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


if __name__ == "__main__":
    main()
