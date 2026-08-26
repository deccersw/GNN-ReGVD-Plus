"""
Vulnerability Scanner Pipeline — orchestrates all 4 modules:
  1. GNN-based detection (existing)
  2. Exploit retrieval from FAISS DB
  3. LLM-based exploit adaptation (with multi-harness + iterative refinement)
  4. Sandbox verification (with batch execution + early stop)

Follows the architecture from PLAN.md, synthesizing ideas from
VBF (Gajjar 2026) and SAST-Genius (Agrawal & Ahi 2025).
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .config import ScannerConfig

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    verdict: str  # CONFIRMED, SUGGESTIVE, NEUTRAL, SAFE
    confidence: float
    cls_prob: float = 0.0
    faiss_score: float = 0.0
    hybrid_score: float = 0.0
    vuln_type: str = "unknown"
    vuln_type_confidence: float = 0.0
    exploits_tried: int = 0
    exploits_confirmed: int = 0
    evidence_details: List[dict] = field(default_factory=list)
    total_time: float = 0.0
    cwe_id: str = ""
    sanitizer_source: str = ""
    taint_context: str = ""
    # Triage head results
    triage_verdict: str = ""          # "true_positive" / "false_positive" / "uncertain"
    triage_confidence: float = 0.0
    triage_reasoning: str = ""
    # Module 0 provenance (project-level scanning). All defaulted so that
    # existing callers -- including ScanResult(**filtered) in scan_cli -- keep
    # working unchanged.
    unit_id: str = ""
    file: str = ""
    function: str = ""
    start_line: int = 0
    end_line: int = 0
    inline_depth_used: int = 0
    inlined_functions: List[str] = field(default_factory=list)
    provenance_files: List[str] = field(default_factory=list)
    inline_truncated: bool = False


class VulnerabilityScanner:
    """
    End-to-end vulnerability scanning pipeline.

    Usage:
        config = ScannerConfig(model_path="...", exploit_db_dir="...")
        scanner = VulnerabilityScanner(config)
        result = scanner.scan(source_code)
    """

    def __init__(self, config: ScannerConfig):
        self.config = config
        self._detector = None
        self._exploit_db = None
        self._adapter = None
        self._sandbox = None
        self._taint_analyzer = None
        self._triage_head = None

    @property
    def detector(self):
        if self._detector is None:
            self._detector = self._init_detector()
        return self._detector

    @property
    def exploit_db(self):
        if self._exploit_db is None:
            self._exploit_db = self._init_exploit_db()
        return self._exploit_db

    @property
    def adapter(self):
        if self._adapter is None:
            self._adapter = self._init_adapter()
        return self._adapter

    @property
    def sandbox(self):
        if self._sandbox is None:
            self._sandbox = self._init_sandbox()
        return self._sandbox

    @property
    def taint_analyzer(self):
        if self._taint_analyzer is None and self.config.use_taint_analysis:
            self._taint_analyzer = self._init_taint_analyzer()
        return self._taint_analyzer

    @property
    def triage_head(self):
        if self._triage_head is None and self.config.use_triage:
            self._triage_head = self._init_triage_head()
        return self._triage_head

    def scan(self, source_code: str) -> ScanResult:
        """
        Full pipeline scan of a code snippet.

        Steps:
          0. (Optional) Taint analysis via Joern/heuristic
          1. Detect vulnerability with GNN + FAISS
          2. If flagged: retrieve matching exploits
          3. Adapt exploits (multi-harness for asymmetric validation)
          4a. Sandbox verification (batch execution + early stop)
          4b. (Optional) Triage Head (LLM-based reasoning)
          5. Verdict aggregation (merge sandbox + triage)
        """
        start = time.time()
        code_lines = source_code.count("\n") + 1
        logger.info(
            "[Pipeline] === SCAN START === code=%d lines, %d chars",
            code_lines, len(source_code),
        )

        # --- Step 0: Taint Analysis ---
        taint_context = ""
        if self.config.use_taint_analysis and self.taint_analyzer:
            t0 = time.time()
            try:
                taint_result = self.taint_analyzer.analyze(source_code)
                taint_context = taint_result.format_for_llm()
                logger.info(
                    "[Pipeline] Step 0 — Taint analysis: method=%s, "
                    "paths=%d, call_edges=%d, time=%.2fs",
                    taint_result.analysis_method,
                    len(taint_result.taint_paths),
                    len(taint_result.call_graph),
                    time.time() - t0,
                )
            except Exception as e:
                logger.warning(
                    "[Pipeline] Step 0 — Taint analysis FAILED (%.2fs): %s",
                    time.time() - t0, e,
                )
        else:
            logger.debug("[Pipeline] Step 0 — Taint analysis: skipped")

        # --- Step 1: GNN Detection ---
        t0 = time.time()
        detection = self._detect(source_code)
        logger.info(
            "[Pipeline] Step 1 — GNN detection: hybrid=%.4f, "
            "cls_prob=%.4f, faiss=%.4f, threshold=%.2f, time=%.2fs",
            detection["hybrid_score"], detection["cls_prob"],
            detection["faiss_score"], self.config.detection_threshold,
            time.time() - t0,
        )

        if detection["hybrid_score"] < self.config.detection_threshold:
            elapsed = time.time() - start
            logger.info(
                "[Pipeline] Step 1 — EARLY EXIT: SAFE "
                "(hybrid=%.4f < threshold=%.2f), total=%.2fs",
                detection["hybrid_score"],
                self.config.detection_threshold, elapsed,
            )
            return ScanResult(
                verdict="SAFE",
                confidence=1 - detection["hybrid_score"],
                cls_prob=detection["cls_prob"],
                faiss_score=detection["faiss_score"],
                hybrid_score=detection["hybrid_score"],
                total_time=elapsed,
                taint_context=taint_context,
            )

        vuln_type = detection.get("vuln_type", "unknown")
        vuln_type_conf = detection.get("vuln_type_confidence", 0.0)
        logger.info(
            "[Pipeline] Step 1 — Vulnerability flagged: type=%s "
            "(confidence=%.2f)",
            vuln_type, vuln_type_conf,
        )

        # --- Step 2: Exploit Retrieval ---
        t0 = time.time()
        exploits = self._retrieve_exploits(
            detection.get("embedding"), vuln_type
        )
        logger.info(
            "[Pipeline] Step 2 — Exploit retrieval: found=%d, time=%.2fs",
            len(exploits), time.time() - t0,
        )
        for i, ex in enumerate(exploits):
            logger.debug(
                "[Pipeline] Step 2 — Exploit #%d: id=%s, type=%s, sim=%.4f",
                i + 1, ex.get("exploit_id", "?"),
                ex.get("vuln_type", "?"), ex.get("similarity", 0),
            )

        # --- Step 3: Harness Adaptation ---
        t0 = time.time()
        harnesses = self._adapt_exploits(
            source_code, exploits, vuln_type, taint_context
        )
        logger.info(
            "[Pipeline] Step 3 — Harness adaptation: "
            "generated=%d (from %d exploits), time=%.2fs",
            len(harnesses), len(exploits), time.time() - t0,
        )

        if not harnesses:
            elapsed = time.time() - start
            logger.warning(
                "[Pipeline] Step 3 — No valid harnesses generated, "
                "returning SUGGESTIVE, total=%.2fs", elapsed,
            )
            return ScanResult(
                verdict="SUGGESTIVE",
                confidence=detection["hybrid_score"],
                cls_prob=detection["cls_prob"],
                faiss_score=detection["faiss_score"],
                hybrid_score=detection["hybrid_score"],
                vuln_type=vuln_type,
                vuln_type_confidence=vuln_type_conf,
                total_time=elapsed,
                taint_context=taint_context,
            )

        # --- Step 4a: Sandbox Verification ---
        t0 = time.time()
        if self.config.dry_run:
            evidence_results = [{
                "exploit_id": h.source_exploit_id,
                "evidence_level": "NEUTRAL",
                "reason": "dry-run: sandbox skipped",
                "exit_code": 0,
                "execution_time": 0.0,
                "compiled": True,
                "confidence": 0.0,
                "cwe_id": "",
                "sanitizer_source": "",
                "sanitizer_profile": "",
                "fuzz_mode": False,
            } for h in harnesses]
            confirmed_count = 0
            logger.info(
                "[Pipeline] Step 4a — Sandbox: DRY-RUN, "
                "generated %d stub evidence entries",
                len(evidence_results),
            )
        else:
            evidence_results, confirmed_count = self._verify_in_sandbox(
                harnesses, exploits, vuln_type
            )
        sandbox_time = time.time() - t0
        sandbox_levels = {}
        for e in evidence_results:
            lvl = e.get("evidence_level", "?")
            sandbox_levels[lvl] = sandbox_levels.get(lvl, 0) + 1
        if not self.config.dry_run:
            logger.info(
                "[Pipeline] Step 4a — Sandbox: tried=%d, confirmed=%d, "
                "levels=%s, time=%.2fs",
                len(evidence_results), confirmed_count,
                sandbox_levels, sandbox_time,
            )
            for i, e in enumerate(evidence_results):
                logger.debug(
                    "[Pipeline] Step 4a — Evidence #%d: exploit=%s, "
                    "level=%s, confidence=%.2f, exit=%s, cwe=%s, "
                    "reason=%s",
                    i + 1, e.get("exploit_id", "?"),
                    e.get("evidence_level", "?"),
                    e.get("confidence", 0),
                    e.get("exit_code", "?"),
                    e.get("cwe_id", ""),
                    e.get("reason", "")[:100],
                )

        # --- Step 4b: Triage Head (LLM-based reasoning) ---
        t0 = time.time()
        if self.config.dry_run and self.config.use_triage:
            triage_result = {
                "triage_verdict": "uncertain",
                "triage_confidence": 0.5,
                "triage_reasoning": "dry-run: triage API skipped",
            }
            logger.info(
                "[Pipeline] Step 4b — Triage: DRY-RUN, "
                "returning stub verdict=uncertain"
            )
        else:
            triage_result = self._triage(
                source_code, vuln_type,
                detection["hybrid_score"], taint_context,
            )
        triage_time = time.time() - t0
        if not self.config.dry_run:
            if triage_result:
                logger.info(
                    "[Pipeline] Step 4b — Triage: verdict=%s, "
                    "confidence=%.2f, time=%.2fs",
                    triage_result.get("triage_verdict", "N/A"),
                    triage_result.get("triage_confidence", 0),
                    triage_time,
                )
                logger.debug(
                    "[Pipeline] Step 4b — Triage reasoning: %s",
                    triage_result.get("triage_reasoning", "")[:200],
                )
            else:
                logger.debug(
                    "[Pipeline] Step 4b — Triage: skipped (not enabled)"
                )

        # --- Step 5: Verdict Aggregation ---
        verdict = self._aggregate_verdict(
            detection["hybrid_score"], evidence_results,
            confirmed_count, triage_result,
        )

        cwe_id = ""
        sanitizer_source = ""
        for e in evidence_results:
            if e.get("evidence_level") == "CONFIRMED":
                cwe_id = e.get("cwe_id", "")
                sanitizer_source = e.get("sanitizer_source", "")
                break

        final_confidence = self._compute_confidence(
            detection["hybrid_score"], evidence_results,
            confirmed_count, triage_result,
        )
        elapsed = time.time() - start

        logger.info(
            "[Pipeline] Step 5 — Aggregation: verdict=%s, "
            "confidence=%.2f, cwe=%s, sandbox=%s, triage=%s",
            verdict, final_confidence,
            cwe_id or "N/A", sandbox_levels,
            triage_result.get("triage_verdict", "N/A"),
        )
        logger.info(
            "[Pipeline] === SCAN DONE === verdict=%s (%.2f), "
            "total=%.2fs (sandbox=%.2fs, triage=%.2fs)",
            verdict, final_confidence, elapsed,
            sandbox_time, triage_time,
        )

        return ScanResult(
            verdict=verdict,
            confidence=final_confidence,
            cls_prob=detection["cls_prob"],
            faiss_score=detection["faiss_score"],
            hybrid_score=detection["hybrid_score"],
            vuln_type=vuln_type,
            vuln_type_confidence=vuln_type_conf,
            exploits_tried=len(evidence_results),
            exploits_confirmed=confirmed_count,
            evidence_details=evidence_results,
            total_time=elapsed,
            cwe_id=cwe_id,
            sanitizer_source=sanitizer_source,
            taint_context=taint_context,
            triage_verdict=triage_result.get("triage_verdict", ""),
            triage_confidence=triage_result.get("triage_confidence", 0.0),
            triage_reasoning=triage_result.get("triage_reasoning", ""),
        )

    def scan_batch(self, code_snippets: List[str],
                   labels: Optional[List[int]] = None) -> List[ScanResult]:
        """Scan multiple code snippets."""
        results = []
        for i, code in enumerate(code_snippets):
            logger.info(f"Scanning sample {i+1}/{len(code_snippets)}")
            result = self.scan(code)
            results.append(result)
        return results

    def _detect(self, source_code: str) -> dict:
        """Run Module 1: GNN detection + FAISS scoring."""
        result = self.detector.predict_single(source_code)

        embedding = result.get("embedding")
        if embedding is not None and self.detector.faiss_manager is not None:
            vuln_type, vt_conf = self.detector.faiss_manager.compute_vuln_type(
                embedding, top_k=self.config.top_k
            )
            result["vuln_type"] = vuln_type
            result["vuln_type_confidence"] = vt_conf

        return result

    def _retrieve_exploits(self, embedding, vuln_type: str) -> List[dict]:
        """Run Module 2: search exploit DB."""
        if self.exploit_db is None or embedding is None:
            return []

        import numpy as np
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)

        return self.exploit_db.search_exploits(
            query_embedding=embedding,
            vuln_type=vuln_type if vuln_type != "unknown" else None,
            top_k=self.config.exploit_top_k,
        )

    def _adapt_exploits(self, source_code: str, exploits: List[dict],
                        vuln_type: str,
                        taint_context: str = "") -> List:
        """Run Module 3: adapt exploit templates to harnesses."""
        code_with_context = source_code
        if taint_context:
            code_with_context = (
                f"/* === Taint Analysis Context ===\n{taint_context}\n*/\n\n"
                + source_code
            )

        if self.config.use_multi_harness and exploits:
            return self.adapter.adapt_multi(
                vuln_code=code_with_context,
                exploit_templates=exploits,
                vuln_type=vuln_type,
                language=self.config.language,
                max_harnesses=self.config.max_exploits_per_sample,
            )

        harnesses = []
        templates = exploits or [{}]
        for template in templates[:self.config.max_exploits_per_sample]:
            harness = self.adapter.adapt(
                vuln_code=code_with_context,
                exploit_template=template,
                vuln_type=vuln_type,
                language=self.config.language,
            )
            if harness.is_valid:
                harnesses.append(harness)
        return harnesses

    def _verify_in_sandbox(self, harnesses: list,
                           exploits: List[dict],
                           vuln_type: str = "") -> tuple:
        """Run Module 4: sandbox verification with batch execution."""
        evidence_results = []
        confirmed_count = 0

        use_multi = self.config.sandbox_use_multi_sanitizer

        for i, harness in enumerate(harnesses):
            exploit_id = harness.source_exploit_id

            if use_multi:
                exec_results = self.sandbox.execute_multi_sanitizer(
                    harness_code=harness.harness_code,
                    payload=harness.payload,
                    vuln_type=vuln_type,
                    expected_signals=harness.expected_signals,
                )
                for exec_result in exec_results:
                    detail = self._build_evidence_detail(
                        exec_result, exploit_id
                    )
                    evidence_results.append(detail)
                    if exec_result.evidence.level.value == "CONFIRMED":
                        confirmed_count += 1

                if confirmed_count > 0 and self.config.early_stop_on_confirmed:
                    break
            else:
                exec_result = self.sandbox.execute(
                    harness_code=harness.harness_code,
                    payload=harness.payload,
                    compilation_cmd=harness.compilation_cmd,
                    expected_signals=harness.expected_signals,
                )
                detail = self._build_evidence_detail(exec_result, exploit_id)
                evidence_results.append(detail)

                if exec_result.evidence.level.value == "CONFIRMED":
                    confirmed_count += 1
                    if self.config.early_stop_on_confirmed:
                        break

        return evidence_results, confirmed_count

    @staticmethod
    def _build_evidence_detail(exec_result, exploit_id: str) -> dict:
        return {
            "exploit_id": exploit_id,
            "evidence_level": exec_result.evidence.level.value,
            "reason": exec_result.evidence.reason,
            "exit_code": exec_result.exit_code,
            "execution_time": exec_result.execution_time,
            "compiled": exec_result.harness_compiled,
            "confidence": exec_result.evidence.confidence,
            "cwe_id": exec_result.evidence.cwe_id,
            "sanitizer_source": exec_result.evidence.sanitizer_source,
            "sanitizer_profile": getattr(exec_result, "sanitizer_profile", ""),
            "fuzz_mode": getattr(exec_result, "fuzz_mode", False),
        }

    def _triage(self, source_code: str, vuln_type: str,
                hybrid_score: float, taint_context: str = "") -> dict:
        """Run triage head: LLM-based verdict on the code snippet."""
        if not self.triage_head:
            logger.debug("[Pipeline._triage] Triage head not enabled, skipping")
            return {}
        try:
            logger.info(
                "[Pipeline._triage] Invoking triage head: "
                "vuln_type=%s, hybrid_score=%.4f, taint=%d chars",
                vuln_type, hybrid_score, len(taint_context),
            )
            result = self.triage_head.triage_sync(
                source_code=source_code,
                vuln_type=vuln_type,
                detection_score=hybrid_score,
                taint_context=taint_context,
            )
            logger.info(
                "[Pipeline._triage] Triage result: verdict=%s, "
                "confidence=%.2f",
                result.verdict, result.confidence,
            )
            return {
                "triage_verdict": result.verdict,
                "triage_confidence": result.confidence,
                "triage_reasoning": result.reasoning,
            }
        except Exception as e:
            logger.error(
                "[Pipeline._triage] Triage head FAILED: %s", e,
                exc_info=True,
            )
            return {}

    def _aggregate_verdict(self, hybrid_score: float,
                           evidence_results: List[dict],
                           confirmed: int,
                           triage: Optional[dict] = None) -> str:
        """
        Merge sandbox evidence + triage verdict.

        Priority: CONFIRMED (runtime proof) > triage signal > heuristics.
        """
        triage = triage or {}
        tv = triage.get("triage_verdict", "")

        suggestive = sum(1 for e in evidence_results
                         if e["evidence_level"] == "SUGGESTIVE")

        logger.debug(
            "[Aggregation] Inputs: confirmed=%d, suggestive=%d, "
            "triage=%s, hybrid=%.4f",
            confirmed, suggestive, tv or "N/A", hybrid_score,
        )

        # Runtime proof is always definitive
        if confirmed > 0:
            logger.debug(
                "[Aggregation] Rule: sandbox CONFIRMED -> CONFIRMED "
                "(runtime proof, triage=%s ignored)", tv,
            )
            return "CONFIRMED"

        if suggestive > 0:
            # Sandbox says SUGGESTIVE — triage can downgrade
            if tv == "false_positive":
                logger.info(
                    "[Aggregation] Rule: sandbox SUGGESTIVE + "
                    "triage false_positive -> NEUTRAL (triage downgraded)"
                )
                return "NEUTRAL"
            logger.debug(
                "[Aggregation] Rule: sandbox SUGGESTIVE + triage=%s "
                "-> SUGGESTIVE", tv or "N/A",
            )
            return "SUGGESTIVE"

        # Sandbox says NEUTRAL (or no evidence)
        if tv == "true_positive":
            logger.info(
                "[Aggregation] Rule: sandbox NEUTRAL + "
                "triage true_positive -> SUGGESTIVE (triage upgraded)"
            )
            return "SUGGESTIVE"
        if tv == "false_positive":
            logger.info(
                "[Aggregation] Rule: sandbox NEUTRAL + "
                "triage false_positive -> SAFE (both heads agree)"
            )
            return "SAFE"

        # No triage or uncertain — fallback to heuristic
        if not evidence_results:
            verdict = "SUGGESTIVE" if hybrid_score > 0.7 else "NEUTRAL"
            logger.debug(
                "[Aggregation] Rule: no evidence, no triage, "
                "hybrid=%.4f -> %s (heuristic)",
                hybrid_score, verdict,
            )
            return verdict

        logger.debug(
            "[Aggregation] Rule: sandbox NEUTRAL + triage=%s -> NEUTRAL "
            "(fallback)", tv or "N/A",
        )
        return "NEUTRAL"

    def _compute_confidence(self, hybrid_score: float,
                            evidence_results: List[dict],
                            confirmed: int,
                            triage: Optional[dict] = None) -> float:
        """
        Compute final confidence, factoring in triage agreement.

        When sandbox and triage agree: boost confidence.
        When they disagree: lower confidence.
        """
        triage = triage or {}
        tv = triage.get("triage_verdict", "")
        tc = triage.get("triage_confidence", 0.0)

        if confirmed > 0:
            max_evidence_conf = max(
                (e.get("confidence", 0.9) for e in evidence_results
                 if e["evidence_level"] == "CONFIRMED"),
                default=0.9,
            )
            base = min(0.99, hybrid_score * 0.5 + max_evidence_conf * 0.5)
            if tv == "true_positive":
                result = min(0.99, base * 0.5 + tc * 0.5)
                logger.debug(
                    "[Confidence] CONFIRMED + triage TP: base=%.2f, "
                    "triage_conf=%.2f -> %.2f (boosted)",
                    base, tc, result,
                )
                return result
            logger.debug(
                "[Confidence] CONFIRMED (no triage boost): "
                "hybrid=%.2f, evidence=%.2f -> %.2f",
                hybrid_score, max_evidence_conf, base,
            )
            return base

        if not evidence_results and not tv:
            logger.debug(
                "[Confidence] No evidence, no triage -> hybrid=%.2f",
                hybrid_score,
            )
            return hybrid_score

        suggestive = sum(1 for e in evidence_results
                         if e["evidence_level"] == "SUGGESTIVE")

        if suggestive > 0:
            base = hybrid_score * 0.9
            if tv == "true_positive":
                result = min(0.99, base * 0.6 + tc * 0.4)
                logger.debug(
                    "[Confidence] SUGGESTIVE + triage TP: base=%.2f, "
                    "triage_conf=%.2f -> %.2f (agreement boost)",
                    base, tc, result,
                )
                return result
            if tv == "false_positive":
                result = base * 0.6
                logger.debug(
                    "[Confidence] SUGGESTIVE + triage FP: base=%.2f "
                    "-> %.2f (disagreement penalty)",
                    base, result,
                )
                return result
            logger.debug(
                "[Confidence] SUGGESTIVE (no triage): base=%.2f", base,
            )
            return base

        # NEUTRAL sandbox
        if tv == "true_positive":
            result = hybrid_score * 0.5 + tc * 0.5
            logger.debug(
                "[Confidence] NEUTRAL + triage TP: hybrid=%.2f, "
                "triage_conf=%.2f -> %.2f",
                hybrid_score, tc, result,
            )
            return result
        if tv == "false_positive":
            result = max(0.5, 1.0 - tc)
            logger.debug(
                "[Confidence] NEUTRAL + triage FP: triage_conf=%.2f "
                "-> %.2f (inverted)",
                tc, result,
            )
            return result

        result = hybrid_score * 0.7
        logger.debug(
            "[Confidence] NEUTRAL fallback: hybrid=%.2f -> %.2f",
            hybrid_score, result,
        )
        return result

    def _init_detector(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

        import torch
        from transformers import (RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer)
        from model import (GNNReGVD, apply_model_config, load_model_config,
                           load_checkpoint_weights)
        from faiss_index import FAISSIndexManager
        from inference import HybridPredictor

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            and not self.config.no_cuda else "cpu"
        )

        tokenizer = RobertaTokenizer.from_pretrained(
            self.config.tokenizer_name
        )
        roberta_config = RobertaConfig.from_pretrained(
            self.config.model_name_or_path
        )
        roberta_config.num_labels = 1
        encoder = RobertaForSequenceClassification.from_pretrained(
            self.config.model_name_or_path, config=roberta_config
        )

        args = type('Args', (), {
            **vars(self.config), 'device': device,
            'n_gpu': 0, 'local_rank': -1,
        })()

        # A checkpoint's state dict is identical in both encoder modes, so
        # loading one under the wrong mode fails silently and produces wrong
        # scores. The sidecar written at training time settles it.
        checkpoint_config = None
        if self.config.model_path and os.path.exists(self.config.model_path):
            checkpoint_config = load_model_config(self.config.model_path)
            if checkpoint_config:
                apply_model_config(args, checkpoint_config, override=True)
            else:
                logger.warning(
                    "[Init] No %s next to %s — assuming encoder_mode=%r. "
                    "If this checkpoint was trained before the sidecar "
                    "existed, it used encoder_mode='static'.",
                    "model_config.json", self.config.model_path,
                    getattr(args, "encoder_mode", "auto"),
                )

        model = GNNReGVD(encoder, roberta_config, tokenizer, args)

        if self.config.model_path and os.path.exists(self.config.model_path):
            # Handles both the slim format (trainable tensors only, base
            # weights rebuilt from the pretrained model) and old full ones.
            load_checkpoint_weights(self.config.model_path, model, device, args)
            logger.info("[Init] Loaded checkpoint %s (encoder_mode=%s)",
                        self.config.model_path, model.encoder_mode)

        faiss_manager = None
        if self.config.faiss_dir and os.path.exists(self.config.faiss_dir):
            faiss_manager = FAISSIndexManager.load(self.config.faiss_dir)

        return HybridPredictor(
            model, tokenizer, faiss_manager, args,
            beta=self.config.beta, top_k=self.config.top_k,
        )

    def _init_exploit_db(self):
        if self.config.exploit_db_dir and os.path.exists(
            self.config.exploit_db_dir
        ):
            from exploit_db.exploit_index import ExploitDBManager
            return ExploitDBManager.load(self.config.exploit_db_dir)

        from exploit_db.exploit_loader import build_exploit_db
        return build_exploit_db(embed_dim=self.config.embed_dim)

    def _init_adapter(self):
        from exploit_db.exploit_adapter import ExploitAdapter

        return ExploitAdapter(
            llm_backend=self.config.llm_backend,
            model_name=self.config.llm_model_name,
            api_url=self.config.llm_api_url,
            api_key=self.config.llm_api_key,
            max_retries=self.config.llm_max_retries,
            temperature=self.config.llm_temperature,
            cache_results=self.config.llm_cache_results,
        )

    def _init_sandbox(self):
        from sandbox.executor import SandboxExecutor, SandboxConfig

        return SandboxExecutor(SandboxConfig(
            memory_limit=self.config.sandbox_memory,
            cpu_quota=self.config.sandbox_cpu,
            timeout=self.config.sandbox_timeout,
            network=self.config.sandbox_network,
            docker_image=self.config.sandbox_docker_image,
            use_valgrind=self.config.sandbox_use_valgrind,
            use_seccomp=self.config.sandbox_use_seccomp,
            use_multi_sanitizer=self.config.sandbox_use_multi_sanitizer,
            use_libfuzzer=self.config.sandbox_use_libfuzzer,
            fuzz_time=self.config.sandbox_fuzz_time,
            prefer_clang=self.config.sandbox_prefer_clang,
        ))

    def _init_taint_analyzer(self):
        from analysis.taint import TaintAnalyzer
        logger.info("[Init] Initializing TaintAnalyzer (joern=%s)",
                     self.config.joern_path or "auto-detect")
        return TaintAnalyzer(joern_path=self.config.joern_path)

    def _init_triage_head(self):
        from triage import TriageHead, TriageConfig
        logger.info(
            "[Init] Initializing TriageHead: model=%s, temp=%.2f",
            self.config.triage_model, self.config.triage_temperature,
        )
        return TriageHead(TriageConfig(
            enabled=True,
            model=self.config.triage_model,
            temperature=self.config.triage_temperature,
            max_tokens=self.config.triage_max_tokens,
            api_key=self.config.triage_api_key,
        ))
