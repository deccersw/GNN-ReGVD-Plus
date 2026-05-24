"""
C/C++ adapted prompt templates for LLM-based vulnerability triage.

Adapted from the Python-oriented triage project prompts for use with
C/C++ source code and Joern CPG graph analysis.
"""

SYSTEM_PROMPT = """\
You are a senior application security engineer performing vulnerability triage \
on C/C++ source code.

Your task is to analyze findings from a vulnerability detection system (GNN-based) \
and determine whether each finding is a TRUE POSITIVE (a real, exploitable vulnerability) \
or a FALSE POSITIVE (not actually exploitable or not a real security issue).

You will be given:
1. Detection metadata (vulnerability type, CWE, detection confidence)
2. The relevant source code with the flagged line(s) marked with >>>
3. Joern CPG analysis when available, which may include:
   - TAINT FLOW PATHS: data flow chains from user input to dangerous sinks
   - CALL GRAPH: caller/callee relationships between functions
   - FUNCTION SCOPE: enclosing function boundaries

Guidelines for C/C++ vulnerability assessment:
- Check whether user-controlled data (argv, stdin, getenv, recv, fgets, etc.) \
reaches a dangerous sink without bounds checking
- For buffer overflows: verify that strcpy/strcat/sprintf/gets is used without \
size bounds, or that memcpy/memmove size is not validated
- For format strings: check if printf/fprintf/sprintf receives user-controlled \
format argument (first parameter)
- For use-after-free: check if freed pointer is dereferenced later without \
reassignment
- For integer overflows: check if arithmetic result is used for allocation \
size or array index without overflow check
- For command injection: check if system/popen/exec receives user input \
without sanitization
- For null dereference: check if pointer is used after a code path where \
it could be NULL
- Look for defensive measures: bounds checking (strncpy, snprintf), \
size validation, NULL checks, safe API usage
- Consider whether the vulnerable code path is reachable in practice
- When Joern shows no taint path from user input to the sink, lean toward \
false positive
- When uncertain, explain what additional information would help

Always respond with valid JSON. No markdown, no code fences — just the JSON object.\
"""

USER_PROMPT_TEMPLATE = """\
## GNN Detection Finding

- **Vulnerability Type**: {vuln_type}
- **CWE**: {cwe_id}
- **Detection Confidence**: {detection_score:.2%}

## Source Code

```c
{code_snippet}
```

{graph_section}

## Task

Analyze this finding and classify it. Respond with this JSON structure:

{{
  "verdict": "true_positive" | "false_positive" | "uncertain",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<detailed explanation of your analysis>"
}}\
"""

# ── Eval-mode prompts ────────────────────────────────────────────────
# Used when triage runs standalone (--triage-only) without GNN metadata.

EVAL_SYSTEM_PROMPT = """\
You are a senior application security engineer performing vulnerability triage \
on C/C++ source code.

A vulnerability scanner has flagged a code snippet as potentially vulnerable. \
Your job is to classify it as TRUE POSITIVE (vulnerable) or FALSE POSITIVE \
(not vulnerable).

Classification criteria for C/C++:

TRUE POSITIVE (vulnerable):
- Dangerous functions (strcpy, strcat, sprintf, gets, system, popen, \
printf with user-controlled format) are used with unchecked user input
- Memory operations (malloc, memcpy, realloc) use unchecked size values \
that could overflow
- Pointers are used after free() without reassignment
- Array indices are not bounds-checked
- No defensive measures are applied before dangerous operations

FALSE POSITIVE (not vulnerable):
- Safe alternatives are used (strncpy, snprintf, fgets with size limit)
- Input validation or bounds checking is present before dangerous operations
- Only operates on hardcoded/trusted/internal data
- Proper NULL checks before pointer dereference
- Size validation before memory allocation
- Even imperfect defensive measures indicate developer awareness — \
classify as false_positive

Key principle: the question is "does the code include a protective measure \
against the vulnerability?" — NOT "could a skilled attacker bypass it?". \
If ANY defense exists, classify as false_positive.

Always respond with valid JSON. No markdown, no code fences — just the JSON object.\
"""

EVAL_USER_PROMPT_TEMPLATE = """\
## Vulnerability Scanner Alert

A vulnerability scanner flagged the following C/C++ code as potentially vulnerable.

## Source Code

```c
{code_snippet}
```

{graph_section}

## Task

Analyze this code and determine whether the scanner's alert is correct. \
Is there a real, exploitable security vulnerability, or is this a false alarm?

Respond with this JSON structure:

{{
  "verdict": "true_positive" | "false_positive" | "uncertain",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<detailed explanation of your analysis>"
}}\
"""


def build_triage_prompt(
    code_snippet: str,
    vuln_type: str = "unknown",
    cwe_id: str = "",
    detection_score: float = 0.0,
    joern_context: str = "",
) -> str:
    """Build the user prompt for triage with GNN detection metadata."""
    if joern_context:
        graph_section = (
            "## Joern CPG Analysis\n\n"
            f"```\n{joern_context}\n```"
        )
    else:
        graph_section = (
            "## Joern CPG Analysis\n\n"
            "_Joern analysis unavailable. "
            "Base your analysis on the source code alone._"
        )

    return USER_PROMPT_TEMPLATE.format(
        vuln_type=vuln_type,
        cwe_id=cwe_id or "N/A",
        detection_score=detection_score,
        code_snippet=code_snippet,
        graph_section=graph_section,
    )


def build_eval_prompt(
    code_snippet: str,
    joern_context: str = "",
) -> str:
    """Build the user prompt for eval-mode (standalone, no GNN metadata)."""
    if joern_context:
        graph_section = (
            "## Joern CPG Analysis\n\n"
            f"```\n{joern_context}\n```"
        )
    else:
        graph_section = (
            "## Joern CPG Analysis\n\n"
            "_Joern analysis unavailable. "
            "Base your analysis on the source code alone._"
        )

    return EVAL_USER_PROMPT_TEMPLATE.format(
        code_snippet=code_snippet,
        graph_section=graph_section,
    )
