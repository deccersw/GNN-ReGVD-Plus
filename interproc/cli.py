#!/usr/bin/env python3
"""
Standalone CLI for the interprocedural inlining module (decision D-012).

Runs stages 1-8 without touching the detector, so the project-to-units split
can be inspected on its own before it is wired into the scanner.

    python -m interproc.cli --project ./test_samples/projects/p01_simple \
        --inline-depth 2 --out units.jsonl --stats stats.json --show 1
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from interproc.config import InliningConfig            # noqa: E402
from interproc.units import InterprocPipeline, write_units_jsonl  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m interproc.cli",
        description="Split a C/C++ project into depth-limited inlined "
                    "analysis units.",
    )
    p.add_argument("--project", required=True,
                   help="Path to the project directory to analyse")

    p.add_argument("--inline-depth", "--depth", dest="depth", type=int,
                   default=2,
                   help="How deep to inline callees: 0 = no inlining "
                        "(current behaviour), 1 = direct callees, 2 = also "
                        "their callees (default: 2)")
    p.add_argument("--strategy", default="priority",
                   choices=["priority", "dfs", "bfs"],
                   help="Order in which call sites spend the token budget")
    p.add_argument("--max-tokens", type=int, default=398,
                   help="Token budget for the GNN view (default: 398 = "
                        "block_size - 2)")
    p.add_argument("--max-callee-tokens", type=int, default=200)
    p.add_argument("--max-expansions", type=int, default=2,
                   help="Max times one callee may be inlined into a unit")
    p.add_argument("--parser", default="auto",
                   choices=["auto", "treesitter", "lexer"])

    p.add_argument("--out", default=None,
                   help="Write units to this JSONL file")
    p.add_argument("--stats", default=None,
                   help="Write build statistics to this JSON file")
    p.add_argument("--emit-dot", default=None,
                   help="Write the call graph in Graphviz DOT format")

    p.add_argument("--max-roots", type=int, default=0,
                   help="Analyse at most N functions (0 = all)")
    p.add_argument("--only", default=None,
                   help="Only build units for functions with this name")
    p.add_argument("--show", type=int, default=0,
                   help="Print the first N units to stdout")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-tokenizer", action="store_true",
                   help="Skip loading GraphCodeBERT; token counts become "
                        "estimates")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def build_config(args) -> InliningConfig:
    return InliningConfig(
        max_depth=args.depth,
        strategy=args.strategy,
        parser_backend=args.parser,
        max_tokens=args.max_tokens,
        max_callee_tokens=args.max_callee_tokens,
        max_expansions_per_callee=args.max_expansions,
        max_roots=args.max_roots,
        use_cache=not args.no_cache,
        use_real_tokenizer=not args.no_tokenizer,
    )


def print_unit(unit, index: int) -> None:
    print("=" * 70)
    print(f"UNIT #{index + 1}  {unit.unit_id}")
    print(f"  root:       {unit.root}")
    print(f"  tokens:     {unit.tokens_gnn} (gnn) / "
          f"{unit.tokens_sandbox} (sandbox)"
          f"{'  [TRUNCATED]' if unit.truncated else ''}")
    print(f"  depth used: {unit.depth_used}")
    print(f"  files:      {', '.join(unit.provenance_files)}")

    if unit.inlined:
        print("  inlined:")
        for rec in unit.inlined:
            print(f"    + {rec.callee_name:<24} depth={rec.depth} "
                  f"score={rec.score:<6} +{rec.tokens_added} tok  "
                  f"(line {rec.line})")
    if unit.skipped:
        print("  skipped:")
        for rec in unit.skipped[:12]:
            detail = f" [{rec.detail}]" if rec.detail else ""
            print(f"    - {rec.callee_name:<24} {rec.reason}{detail}")
        if len(unit.skipped) > 12:
            print(f"    ... and {len(unit.skipped) - 12} more")

    print("\n  --- code_for_gnn ---")
    print(_indent(unit.code_for_gnn))
    print("\n  --- code_for_sandbox ---")
    print(_indent(unit.code_for_sandbox))
    print()


def _indent(text: str, prefix: str = "  | ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("interproc.cli")

    config = build_config(args)
    logger.info("[CLI] project=%s depth=%d strategy=%s budget=%d parser=%s",
                args.project, config.max_depth, config.strategy,
                config.max_tokens, config.parser_backend)

    pipeline = InterprocPipeline(config)
    try:
        pipeline.analyze(args.project)
    except (NotADirectoryError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not pipeline.index.functions:
        print("Error: no functions found — is this a C/C++ project?",
              file=sys.stderr)
        return 3

    if args.emit_dot:
        with open(args.emit_dot, "w", encoding="utf-8") as fh:
            fh.write(pipeline.graph.to_dot())
        logger.info("[CLI] call graph written to %s", args.emit_dot)

    units = pipeline.build(args.project)
    if args.only:
        units = [u for u in units if u.root.name == args.only]

    for i, unit in enumerate(units[:args.show]):
        print_unit(unit, i)

    if args.out:
        write_units_jsonl(units, args.out)
        logger.info("[CLI] %d units written to %s", len(units), args.out)

    stats = pipeline.stats.to_dict()
    if args.stats:
        with open(args.stats, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        logger.info("[CLI] statistics written to %s", args.stats)

    print(_summary(stats, len(units)))
    return 0


def _summary(stats: dict, n_units: int) -> str:
    lines = [
        "",
        "=" * 60,
        "INTERPROC BUILD SUMMARY",
        "=" * 60,
        f"  files parsed:        {stats['files_parsed']}",
        f"  functions found:     {stats['functions_found']}",
        f"  roots selected:      {stats['roots_selected']} "
        f"(trivial {stats['roots_skipped_trivial']}, "
        f"dup {stats['roots_skipped_duplicate']}, "
        f"test {stats['roots_skipped_test']})",
        f"  units built:         {n_units}",
        f"  with inlining:       {stats['units_with_inlining']}",
        f"  inlined call sites:  {stats['total_inlined_calls']}",
        f"  truncated units:     {stats['units_truncated']}",
        f"  tokens avg:          {stats['avg_tokens_before']} -> "
        f"{stats['avg_tokens_after']} (x{stats['avg_growth']})",
        f"  parser / tokenizer:  {stats['parser_backend']} / "
        f"{'exact' if stats['tokenizer_exact'] else 'ESTIMATED'}",
        f"  elapsed:             {stats['elapsed_sec']}s",
    ]
    if stats["call_resolution"]:
        lines.append("  call resolution:")
        for k, v in sorted(stats["call_resolution"].items(),
                           key=lambda kv: -kv[1]):
            lines.append(f"    {k:<20} {v}")
    if stats["skip_reasons"]:
        lines.append("  inline skip reasons:")
        for k, v in sorted(stats["skip_reasons"].items(),
                           key=lambda kv: -kv[1]):
            lines.append(f"    {k:<20} {v}")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
