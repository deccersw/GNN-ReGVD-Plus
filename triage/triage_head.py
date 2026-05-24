"""
Triage Head — LLM-based vulnerability classification for C/C++ code.

Wraps litellm to send code + Joern context to an LLM and get a
true_positive / false_positive / uncertain verdict.

Does NOT require SARIF — constructs analysis context directly from
GNN detection output and Joern taint results.
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TriageVerdict:
    """LLM triage verdict for a code snippet."""

    verdict: str   # "true_positive" | "false_positive" | "uncertain"
    confidence: float  # 0.0 — 1.0
    reasoning: str


class TriageHead:
    """
    LLM-based triage head for the vulnerability scanner pipeline.

    Sends code + detection metadata + Joern graph context to an LLM
    and returns a TriageVerdict.

    Usage:
        from triage import TriageHead, TriageConfig

        head = TriageHead(TriageConfig(
            enabled=True,
            model="openrouter/anthropic/claude-sonnet-4-6",
        ))
        verdict = head.triage_sync(
            source_code="void f(char *s){char b[8];strcpy(b,s);}",
            vuln_type="buffer_overflow",
            detection_score=0.87,
        )
        print(verdict.verdict, verdict.confidence)
    """

    def __init__(self, config):
        """
        Args:
            config: TriageConfig with model, temperature, max_tokens, api_key.
        """
        self.config = config
        self._check_litellm()

        # Set API key if provided (otherwise litellm reads from env)
        api_key = config.api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            os.environ.setdefault("OPENROUTER_API_KEY", api_key)

        logger.info(
            "[TriageHead] Initialized: model=%s, temperature=%.2f, "
            "max_tokens=%d, api_key=%s",
            config.model, config.temperature, config.max_tokens,
            "set" if api_key else "not set (will use env)",
        )

    @staticmethod
    def _check_litellm():
        """Verify litellm is importable."""
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError(
                "litellm is required for the triage head. "
                "Install it with: pip install litellm"
            )

    async def triage(
        self,
        source_code: str,
        vuln_type: str = "unknown",
        detection_score: float = 0.0,
        taint_context: str = "",
        cwe_id: str = "",
    ) -> TriageVerdict:
        """
        Triage a code snippet using LLM reasoning.

        Args:
            source_code: The C/C++ source code to analyze.
            vuln_type: Vulnerability type from GNN detection.
            detection_score: Hybrid score from GNN detection (0.0-1.0).
            taint_context: Formatted Joern/heuristic taint analysis output.
            cwe_id: CWE identifier if known.

        Returns:
            TriageVerdict with verdict, confidence, and reasoning.
        """
        import litellm
        import time as _time

        from .prompts_c import SYSTEM_PROMPT, build_triage_prompt

        logger.info(
            "[Triage] Starting triage: vuln_type=%s, detection_score=%.4f, "
            "cwe=%s, taint_context=%d chars",
            vuln_type, detection_score, cwe_id or "N/A",
            len(taint_context),
        )

        code_snippet = self._format_code(source_code)
        prompt = build_triage_prompt(
            code_snippet=code_snippet,
            vuln_type=vuln_type,
            cwe_id=cwe_id,
            detection_score=detection_score,
            joern_context=taint_context,
        )

        logger.debug(
            "[Triage] Prompt built: system=%d chars, user=%d chars, "
            "code=%d lines",
            len(SYSTEM_PROMPT), len(prompt),
            source_code.count("\n") + 1,
        )

        t0 = _time.time()
        try:
            logger.info(
                "[Triage] Calling LLM: model=%s, temp=%.2f",
                self.config.model, self.config.temperature,
            )
            response = await litellm.acompletion(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            elapsed = _time.time() - t0
            logger.error(
                "[Triage] LLM call FAILED after %.2fs: %s", elapsed, e,
            )
            return TriageVerdict(
                verdict="uncertain",
                confidence=0.0,
                reasoning=f"LLM call failed: {e}",
            )

        elapsed = _time.time() - t0
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        logger.info(
            "[Triage] LLM responded in %.2fs, tokens=%d, "
            "response=%d chars",
            elapsed, tokens, len(content) if content else 0,
        )
        logger.debug("[Triage] Raw LLM response: %s", content[:500])

        verdict = self._parse_response(content)
        logger.info(
            "[Triage] Parsed verdict: %s (confidence=%.2f)",
            verdict.verdict, verdict.confidence,
        )
        logger.debug("[Triage] Reasoning: %s", verdict.reasoning[:300])
        return verdict

    async def triage_eval(
        self,
        source_code: str,
        taint_context: str = "",
    ) -> TriageVerdict:
        """
        Triage in eval mode (standalone, no GNN metadata).

        Used with --triage-only flag when there's no GNN detection info.
        """
        import litellm
        import time as _time

        from .prompts_c import EVAL_SYSTEM_PROMPT, build_eval_prompt

        logger.info(
            "[Triage-Eval] Starting eval triage: code=%d lines, "
            "taint_context=%d chars",
            source_code.count("\n") + 1, len(taint_context),
        )

        code_snippet = self._format_code(source_code)
        prompt = build_eval_prompt(
            code_snippet=code_snippet,
            joern_context=taint_context,
        )

        logger.debug(
            "[Triage-Eval] Prompt built: system=%d chars, user=%d chars",
            len(EVAL_SYSTEM_PROMPT), len(prompt),
        )

        t0 = _time.time()
        try:
            logger.info(
                "[Triage-Eval] Calling LLM: model=%s, temp=%.2f",
                self.config.model, self.config.temperature,
            )
            response = await litellm.acompletion(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            elapsed = _time.time() - t0
            logger.error(
                "[Triage-Eval] LLM call FAILED after %.2fs: %s", elapsed, e,
            )
            return TriageVerdict(
                verdict="uncertain",
                confidence=0.0,
                reasoning=f"LLM call failed: {e}",
            )

        elapsed = _time.time() - t0
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        logger.info(
            "[Triage-Eval] LLM responded in %.2fs, tokens=%d, "
            "response=%d chars",
            elapsed, tokens, len(content) if content else 0,
        )
        logger.debug("[Triage-Eval] Raw LLM response: %s", content[:500])

        verdict = self._parse_response(content)
        logger.info(
            "[Triage-Eval] Parsed verdict: %s (confidence=%.2f)",
            verdict.verdict, verdict.confidence,
        )
        logger.debug("[Triage-Eval] Reasoning: %s", verdict.reasoning[:300])
        return verdict

    def triage_sync(
        self,
        source_code: str,
        vuln_type: str = "unknown",
        detection_score: float = 0.0,
        taint_context: str = "",
        cwe_id: str = "",
    ) -> TriageVerdict:
        """Synchronous wrapper around triage()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop — create a new one in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.triage(
                        source_code, vuln_type, detection_score,
                        taint_context, cwe_id,
                    ),
                )
                return future.result()
        else:
            return asyncio.run(
                self.triage(
                    source_code, vuln_type, detection_score,
                    taint_context, cwe_id,
                )
            )

    def triage_eval_sync(
        self,
        source_code: str,
        taint_context: str = "",
    ) -> TriageVerdict:
        """Synchronous wrapper around triage_eval()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.triage_eval(source_code, taint_context),
                )
                return future.result()
        else:
            return asyncio.run(
                self.triage_eval(source_code, taint_context)
            )

    @staticmethod
    def _format_code(source_code: str) -> str:
        """Add line numbers and >>> markers to source code."""
        lines = source_code.split("\n")
        numbered = []
        for i, line in enumerate(lines, 1):
            numbered.append(f"  {i:4d}  {line}")
        return "\n".join(numbered)

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract a JSON object from arbitrary LLM response text.

        Handles markdown fences, prose around JSON, etc.
        """
        stripped = text.strip()

        # Strip markdown fences
        if stripped.startswith("```"):
            first_nl = stripped.find("\n")
            if first_nl != -1:
                stripped = stripped[first_nl + 1:]
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3].rstrip()
            try:
                json.loads(stripped)
                return stripped
            except json.JSONDecodeError:
                pass

        # Try raw text
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

        # Find JSON with "verdict" key
        for match in re.finditer(r'\{[^{}]*"verdict"[^{}]*\}', text):
            candidate = match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

        # Greedy: largest { ... } block
        brace_start = text.find("{")
        if brace_start != -1:
            brace_end = text.rfind("}")
            if brace_end > brace_start:
                candidate = text[brace_start:brace_end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass

        return None

    def _parse_response(self, content: str) -> TriageVerdict:
        """Parse the LLM's JSON response into a TriageVerdict."""
        logger.debug("[Triage-Parse] Parsing response (%d chars)", len(content))

        extracted = self._extract_json(content)
        if not extracted:
            logger.error(
                "[Triage-Parse] FAILED to extract JSON from response: %s",
                content[:300],
            )
            return TriageVerdict(
                verdict="uncertain",
                confidence=0.0,
                reasoning=f"Failed to parse LLM response: {content[:500]}",
            )

        logger.debug("[Triage-Parse] Extracted JSON: %s", extracted[:300])

        try:
            data = json.loads(extracted)
        except json.JSONDecodeError as e:
            logger.error(
                "[Triage-Parse] JSON decode FAILED: %s | extracted: %s",
                e, extracted[:200],
            )
            return TriageVerdict(
                verdict="uncertain",
                confidence=0.0,
                reasoning=f"JSON parse error: {e}. Raw: {content[:500]}",
            )

        verdict_str = data.get("verdict", "uncertain")
        if verdict_str not in ("true_positive", "false_positive", "uncertain"):
            logger.warning(
                "[Triage-Parse] Unknown verdict '%s', defaulting to uncertain",
                verdict_str,
            )
            verdict_str = "uncertain"

        confidence = data.get("confidence", 0.5)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            logger.warning(
                "[Triage-Parse] Invalid confidence value '%s', "
                "defaulting to 0.5", data.get("confidence"),
            )
            confidence = 0.5

        reasoning = data.get("reasoning", "No reasoning provided")

        logger.debug(
            "[Triage-Parse] Result: verdict=%s, confidence=%.2f, "
            "reasoning=%d chars",
            verdict_str, confidence, len(reasoning),
        )

        return TriageVerdict(
            verdict=verdict_str,
            confidence=confidence,
            reasoning=reasoning,
        )
