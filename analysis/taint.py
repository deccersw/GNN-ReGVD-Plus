"""
Taint Analysis via Joern — extracts source-to-sink data flow paths
from C/C++ code using Joern's Code Property Graph (CPG).

Provides inter-procedural context that improves LLM exploit generation:
  - Taint paths: which user inputs reach which dangerous sinks
  - Call graphs: how functions call each other
  - Vulnerable regions: specific lines/functions where the vuln lives

Falls back to regex-based heuristic analysis when Joern is unavailable.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

DANGEROUS_SINKS = {
    "memory": [
        "strcpy", "strcat", "sprintf", "vsprintf", "gets",
        "memcpy", "memmove", "bcopy",
    ],
    "format": [
        "printf", "fprintf", "sprintf", "snprintf", "syslog",
        "vprintf", "vfprintf", "vsprintf", "vsnprintf",
    ],
    "exec": [
        "system", "popen", "exec", "execl", "execle", "execlp",
        "execv", "execve", "execvp", "execvpe",
    ],
    "file": [
        "fopen", "open", "freopen", "tmpnam", "mktemp",
    ],
    "alloc": [
        "malloc", "calloc", "realloc", "alloca", "free",
    ],
}

INPUT_SOURCES = [
    "argv", "argc", "getenv", "gets", "fgets", "scanf", "fscanf",
    "sscanf", "read", "recv", "recvfrom", "recvmsg", "fread",
    "getchar", "getc", "fgetc", "getline", "getdelim",
    "stdin",
]

JOERN_TAINT_QUERY = """
importCode("{source_path}")

val sources = cpg.call.name("({source_pattern})").l
val sinks = cpg.call.name("({sink_pattern})").l

println("===TAINT_SOURCES===")
sources.foreach {{ s =>
    println(s"${{s.name}}|${{s.lineNumber.getOrElse(-1)}}|${{s.file.name.headOption.getOrElse("")}}")
}}

println("===TAINT_SINKS===")
sinks.foreach {{ s =>
    println(s"${{s.name}}|${{s.lineNumber.getOrElse(-1)}}|${{s.file.name.headOption.getOrElse("")}}")
}}

println("===CALL_GRAPH===")
cpg.method.l.foreach {{ m =>
    m.callOut.l.foreach {{ c =>
        println(s"${{m.name}}|${{c.name}}|${{c.lineNumber.getOrElse(-1)}}")
    }}
}}

println("===FUNCTIONS===")
cpg.method.isExternal(false).l.foreach {{ m =>
    println(s"${{m.name}}|${{m.lineNumber.getOrElse(-1)}}|${{m.lineNumberEnd.getOrElse(-1)}}")
}}
"""


@dataclass
class TaintPath:
    source_func: str
    source_line: int
    sink_func: str
    sink_line: int
    sink_category: str
    intermediates: List[str] = field(default_factory=list)


@dataclass
class CallEdge:
    caller: str
    callee: str
    line: int


@dataclass
class FunctionInfo:
    name: str
    start_line: int
    end_line: int


@dataclass
class TaintResult:
    taint_paths: List[TaintPath]
    call_graph: List[CallEdge]
    functions: List[FunctionInfo]
    source_file: str = ""
    analysis_method: str = ""

    @property
    def has_taint(self) -> bool:
        return len(self.taint_paths) > 0

    def get_dangerous_functions(self) -> List[str]:
        return list(set(p.sink_func for p in self.taint_paths))

    def get_caller_chain(self, func_name: str) -> List[str]:
        """Get the chain of callers for a given function."""
        callers = []
        visited = set()
        self._find_callers(func_name, callers, visited)
        return callers

    def _find_callers(self, func: str, chain: list, visited: set):
        if func in visited:
            return
        visited.add(func)
        for edge in self.call_graph:
            if edge.callee == func and edge.caller not in visited:
                chain.append(edge.caller)
                self._find_callers(edge.caller, chain, visited)

    def format_for_llm(self) -> str:
        """Format taint analysis results as context for LLM prompts."""
        parts = []

        if self.taint_paths:
            parts.append("Taint Analysis Results:")
            for i, path in enumerate(self.taint_paths[:5], 1):
                chain = " -> ".join(path.intermediates) if path.intermediates else "direct"
                parts.append(
                    f"  {i}. {path.source_func}(line {path.source_line}) "
                    f"-[{chain}]-> {path.sink_func}(line {path.sink_line}) "
                    f"[{path.sink_category}]"
                )

        if self.call_graph:
            parts.append("\nRelevant call graph:")
            seen = set()
            for edge in self.call_graph[:10]:
                key = f"{edge.caller}->{edge.callee}"
                if key not in seen:
                    seen.add(key)
                    parts.append(f"  {edge.caller}() -> {edge.callee}()")

        return "\n".join(parts) if parts else ""


class TaintAnalyzer:
    """
    Extracts taint paths from C/C++ source code.

    Uses Joern when available, falls back to regex-based heuristic analysis.
    """

    def __init__(self, joern_path: Optional[str] = None):
        path = joern_path or self._find_joern()
        # Convert to absolute so it works from any cwd (e.g. temp dirs)
        self.joern_path = os.path.abspath(path) if path else None
        self._joern_available = None

    @property
    def joern_available(self) -> bool:
        if self._joern_available is None:
            if not self.joern_path:
                self._joern_available = False
            elif not os.path.isfile(self.joern_path):
                logger.warning(
                    "[TaintAnalyzer] Joern binary not found: %s",
                    self.joern_path,
                )
                self._joern_available = False
            else:
                # Joern CLI is a REPL — --version hangs.
                # Verify by running a trivial script that exits immediately.
                try:
                    result = subprocess.run(
                        [self.joern_path, "--script", "/dev/null"],
                        capture_output=True, timeout=30,
                    )
                    self._joern_available = True
                    logger.info(
                        "[TaintAnalyzer] Joern available: %s",
                        self.joern_path,
                    )
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "[TaintAnalyzer] Joern check timed out, "
                        "falling back to heuristic"
                    )
                    self._joern_available = False
                except (FileNotFoundError, PermissionError, OSError) as e:
                    logger.warning(
                        "[TaintAnalyzer] Joern check failed: %s", e,
                    )
                    self._joern_available = False
        return self._joern_available

    def analyze(self, source_code: str, filename: str = "input.c") -> TaintResult:
        """
        Analyze source code for taint paths.

        Uses Joern if available, otherwise falls back to regex heuristics.
        """
        if self.joern_available:
            return self._analyze_joern(source_code, filename)
        return self._analyze_heuristic(source_code, filename)

    def _analyze_joern(self, source_code: str, filename: str) -> TaintResult:
        """Full Joern CPG analysis."""
        workdir = tempfile.mkdtemp(prefix="joern_")
        try:
            source_path = os.path.join(workdir, filename)
            with open(source_path, "w") as f:
                f.write(source_code)

            source_pattern = "|".join(INPUT_SOURCES[:10])
            all_sinks = []
            for sinks in DANGEROUS_SINKS.values():
                all_sinks.extend(sinks)
            sink_pattern = "|".join(all_sinks[:20])

            query = JOERN_TAINT_QUERY.format(
                source_path=source_path.replace("\\", "\\\\"),
                source_pattern=source_pattern,
                sink_pattern=sink_pattern,
            )

            query_path = os.path.join(workdir, "query.sc")
            with open(query_path, "w") as f:
                f.write(query)

            result = subprocess.run(
                [self.joern_path, "--script", query_path],
                capture_output=True, text=True,
                timeout=120, cwd=workdir,
            )

            return self._parse_joern_output(result.stdout, source_path)
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"Joern analysis failed: {e}, falling back to heuristic")
            return self._analyze_heuristic(source_code, filename)
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    def _parse_joern_output(self, output: str, source_file: str) -> TaintResult:
        """Parse Joern query output into TaintResult."""
        sources = []
        sinks = []
        call_graph = []
        functions = []

        section = None
        for line in output.strip().split("\n"):
            line = line.strip()
            if line == "===TAINT_SOURCES===":
                section = "sources"
                continue
            elif line == "===TAINT_SINKS===":
                section = "sinks"
                continue
            elif line == "===CALL_GRAPH===":
                section = "callgraph"
                continue
            elif line == "===FUNCTIONS===":
                section = "functions"
                continue

            parts = line.split("|")
            if section == "sources" and len(parts) >= 2:
                sources.append({"name": parts[0], "line": int(parts[1])})
            elif section == "sinks" and len(parts) >= 2:
                category = self._categorize_sink(parts[0])
                sinks.append({
                    "name": parts[0], "line": int(parts[1]),
                    "category": category,
                })
            elif section == "callgraph" and len(parts) >= 3:
                call_graph.append(CallEdge(
                    caller=parts[0], callee=parts[1],
                    line=int(parts[2]) if parts[2] != "-1" else -1,
                ))
            elif section == "functions" and len(parts) >= 3:
                functions.append(FunctionInfo(
                    name=parts[0],
                    start_line=int(parts[1]) if parts[1] != "-1" else -1,
                    end_line=int(parts[2]) if parts[2] != "-1" else -1,
                ))

        taint_paths = []
        for src in sources:
            for sink in sinks:
                taint_paths.append(TaintPath(
                    source_func=src["name"],
                    source_line=src["line"],
                    sink_func=sink["name"],
                    sink_line=sink["line"],
                    sink_category=sink["category"],
                ))

        return TaintResult(
            taint_paths=taint_paths,
            call_graph=call_graph,
            functions=functions,
            source_file=source_file,
            analysis_method="joern",
        )

    def _analyze_heuristic(self, source_code: str, filename: str) -> TaintResult:
        """Regex-based heuristic taint analysis (fallback when Joern unavailable)."""
        taint_paths = []
        call_graph = []
        functions = []

        func_pattern = re.compile(
            r'(?:(?:static|void|int|char|long|unsigned|short|float|double|'
            r'size_t|ssize_t)\s*\*?\s+)+(\w+)\s*\(([^)]*)\)\s*\{',
        )

        lines = source_code.split("\n")
        skip_names = {"if", "for", "while", "switch", "return", "sizeof"}

        for match in func_pattern.finditer(source_code):
            name = match.group(1)
            if name in skip_names:
                continue
            start_line = source_code[:match.start()].count("\n") + 1
            brace_count = 0
            end_line = start_line
            for i, line in enumerate(lines[start_line - 1:], start_line):
                brace_count += line.count("{") - line.count("}")
                if brace_count <= 0 and i > start_line:
                    end_line = i
                    break
            functions.append(FunctionInfo(name, start_line, end_line))

        for line_num, line in enumerate(lines, 1):
            for source in INPUT_SOURCES:
                if re.search(rf'\b{re.escape(source)}\b', line):
                    for category, sinks in DANGEROUS_SINKS.items():
                        for sink in sinks:
                            sink_pattern = re.compile(
                                rf'\b{re.escape(sink)}\s*\('
                            )
                            for sink_line_num, sink_line in enumerate(lines, 1):
                                if sink_pattern.search(sink_line):
                                    taint_paths.append(TaintPath(
                                        source_func=source,
                                        source_line=line_num,
                                        sink_func=sink,
                                        sink_line=sink_line_num,
                                        sink_category=category,
                                    ))

            call_pattern = re.compile(r'\b(\w+)\s*\(')
            for func in functions:
                func_body_start = func.start_line - 1
                func_body_end = min(func.end_line, len(lines))
                for i in range(func_body_start, func_body_end):
                    for m in call_pattern.finditer(lines[i]):
                        callee = m.group(1)
                        if callee not in skip_names and callee != func.name:
                            call_graph.append(CallEdge(
                                caller=func.name,
                                callee=callee,
                                line=i + 1,
                            ))

        seen_paths = set()
        unique_paths = []
        for p in taint_paths:
            key = (p.source_func, p.sink_func, p.sink_line)
            if key not in seen_paths:
                seen_paths.add(key)
                unique_paths.append(p)

        return TaintResult(
            taint_paths=unique_paths[:20],
            call_graph=call_graph,
            functions=functions,
            source_file=filename,
            analysis_method="heuristic",
        )

    @staticmethod
    def _categorize_sink(func_name: str) -> str:
        for category, funcs in DANGEROUS_SINKS.items():
            if func_name in funcs:
                return category
        return "unknown"

    @staticmethod
    def _find_joern() -> Optional[str]:
        for path in [
            "joern",
            "/usr/local/bin/joern",
            os.path.expanduser("~/bin/joern"),
            os.path.expanduser("~/.local/bin/joern"),
        ]:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                logger.debug("[TaintAnalyzer] Found Joern at: %s", path)
                return path
        return None
