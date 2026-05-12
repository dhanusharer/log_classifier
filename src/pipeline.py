"""
Pipeline orchestrator — wires Stage A (regex) and Stage B (ML) together.

Data flow
─────────
  raw log message
       │
       ▼
  ┌─────────────────────┐
  │   Stage A: Regex    │  score ≥ high_conf_gate → accept
  │   Classifier        │  low_gate ≤ score < high_gate → Stage B (double-check)
  └─────────────────────┘  score < low_gate → Stage B
       │ None / low-conf
       ▼
  ┌─────────────────────┐
  │   Stage B: ML       │  confidence ≥ ml_min_conf → accept
  │   Classifier        │  confidence <  ml_min_conf → LLM fallback
  └─────────────────────┘
       │
       ▼
  ClassificationResult  →  CSV row  +  audit log
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, load_config
from .ml_classifier import MLClassifier, build_ml_classifier
from .regex_classifier import RegexClassifier
from .schemas import ClassificationResult, TrainingStatus

logger = logging.getLogger(__name__)


class LogClassificationPipeline:
    def __init__(
        self,
        regex_clf: RegexClassifier,
        ml_clf: MLClassifier,
        config: PipelineConfig,
    ) -> None:
        self._regex = regex_clf
        self._ml = ml_clf
        self._cfg = config

    # ── Core classification ──────────────────────────────────────────────────

    def classify_one(self, message: str, source: Optional[str] = None) -> ClassificationResult:
        """Run one log message through the two-stage pipeline."""
        message = message.strip()

        # Stage A
        result = self._regex.classify(message, source=source)

        if result is not None:
            high_gate = self._cfg.thresholds.regex_high_confidence
            if result.confidence_score >= high_gate:
                self._audit(result)
                return result
            # Score is in the ambiguous zone → let Stage B confirm or override
            logger.debug("Regex ambiguous (%.3f); passing to Stage B", result.confidence_score)

        # Stage B
        result = self._ml.classify(message, source=source)
        self._audit(result)
        return result

    def classify_batch(
        self,
        messages: list[str],
        sources: Optional[list[Optional[str]]] = None,
    ) -> list[ClassificationResult]:
        if sources is None:
            sources = [None] * len(messages)
        return [self.classify_one(m, s) for m, s in zip(messages, sources)]

    # ── Audit trail ──────────────────────────────────────────────────────────

    def _audit(self, result: ClassificationResult) -> None:
        audit_logger = logging.getLogger("audit")
        audit_logger.info(
            "label=%s confidence=%.4f method=%s training_status=%s source=%s message=%r",
            result.predicted_label,
            result.confidence_score,
            result.method_used.value,
            result.training_status.value,
            result.source or "—",
            result.log_message[:120],
        )

    # ── CSV export ───────────────────────────────────────────────────────────

    def export_csv(
        self,
        results: list[ClassificationResult],
        path: Optional[Path] = None,
        append: bool = False,
    ) -> Path:
        out_path = path or self._cfg.output_csv
        mode = "a" if append else "w"
        fieldnames = [
            "log_message", "predicted_label", "confidence_score",
            "method_used", "training_status", "source", "timestamp",
        ]
        write_header = not (append and out_path.exists())
        with open(out_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for r in results:
                writer.writerow(r.to_csv_row())
        logger.info("Exported %d rows → %s", len(results), out_path)
        return out_path

    def flagged_for_labelling(
        self, results: list[ClassificationResult]
    ) -> list[ClassificationResult]:
        """Return samples that need human review, such as genuinely new labels."""
        return [
            r for r in results
            if r.training_status in (TrainingStatus.UNKNOWN, TrainingStatus.NEW)
        ]


# ── Factory ────────────────────────────────────────────────────────────────────

def build_pipeline(
    training_texts: list[str],
    training_labels: list[str],
    config: Optional[PipelineConfig] = None,
    llm_api_key: str = "",
) -> LogClassificationPipeline:
    """
    End-to-end factory.  Call once at application startup, reuse the
    returned pipeline object for all classification requests.
    """
    cfg = config or load_config()

    regex_clf = RegexClassifier(
        high_conf_gate=cfg.thresholds.regex_high_confidence,
        low_conf_gate=cfg.thresholds.regex_low_confidence,
    )

    ml_clf = build_ml_classifier(
        texts=training_texts,
        labels=training_labels,
        model_dir=cfg.model_backend == "auto" and Path("models") or Path(cfg.model_backend),
        small_corpus_threshold=cfg.thresholds.small_corpus_threshold,
        offline_mode=cfg.offline_mode,
        min_samples_per_label=cfg.thresholds.min_samples_per_label,
        min_confidence=cfg.thresholds.ml_min_confidence,
        llm_api_key=llm_api_key or cfg.llm_api_key,
        llm_provider=cfg.llm_provider,
        llm_model=cfg.llm_model,
    )

    return LogClassificationPipeline(regex_clf, ml_clf, cfg)
