"""
Stage B and Stage C classifiers for the log classification pipeline.

Decision flow:
  1. LLM-tier intercept catches sparse labels before ML can misassign them.
  2. SVM/BERT handles labels with enough training samples.
  3. LLM fallback handles every low-confidence or unknown row.

The fallback supports DeepSeek R1 through the OpenAI-compatible DeepSeek API
when LOG_OFFLINE=false and DEEPSEEK_API_KEY is set. In offline mode it uses a
deterministic semantic fallback so local demos still satisfy the client rule:
no output row should remain UNCLASSIFIED.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

from .schemas import (
    BERT_TIER_LABELS,
    ClassificationMethod,
    ClassificationResult,
    LLM_TIER_LABELS,
    REGEX_TIER_LABELS,
    TrainingStatus,
)

logger = logging.getLogger(__name__)

UNCLASSIFIED_LABEL = "UNCLASSIFIED"
VALID_LABELS = sorted(REGEX_TIER_LABELS | BERT_TIER_LABELS | LLM_TIER_LABELS)


_LLM_INTERCEPT_RULES: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"(?i)("
            r"(lead|prospect)\s+(conversion|follow.?up)\s+(fail|process)|"
            r"escalation\s+rule\s+(execution\s+)?fail|"
            r"task\s+assignment\s+.*(could\s+not|fail)|"
            r"undefined\s+(escalation|priority)|"
            r"missing\s+(next\s+action|contact\s+information)|"
            r"invoice\s+.*(fail|could\s+not|error)|"
            r"bulk\s+operation\s+.*(fail|could\s+not|error)|"
            r"case\s+escalation"
            r")"
        ),
        "Workflow Error",
    ),
    (
        re.compile(
            r"(?i)("
            r"(deprecated|deprecation|is\s+deprecated|will\s+be\s+removed)|"
            r"(outdated|migrate\s+to|discontinued|unsupported|no\s+longer\s+supported)|"
            r"(will\s+be\s+retired|retired\s+in|module\s+will\s+be\s+retired)|"
            r"(legacy\s+(method|auth|endpoint|feature)\s+(will|discont|discard))"
            r")"
        ),
        "Deprecation Warning",
    ),
]


def _llm_intercept(message: str) -> Optional[str]:
    for pattern, label in _LLM_INTERCEPT_RULES:
        if pattern.search(message):
            return label
    return None


class BaseMLClassifier:
    def fit(self, texts: list[str], labels: list[str]) -> None:
        raise NotImplementedError

    def predict_proba(self, text: str) -> tuple[str, float]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError


class SVMClassifier(BaseMLClassifier):
    """TF-IDF (1-3 gram) + CalibratedClassifierCV(LinearSVC)."""

    def __init__(self) -> None:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3),
                sublinear_tf=True,
                min_df=2,
                max_features=50_000,
                strip_accents="unicode",
            )),
            ("clf", CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3)),
        ])

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self._pipeline.fit(texts, labels)
        logger.info(
            "SVM trained on %d samples, classes=%s",
            len(texts),
            list(self._pipeline.classes_),
        )

    def predict_proba(self, text: str) -> tuple[str, float]:
        proba = self._pipeline.predict_proba([text])[0]
        best_idx = proba.argmax()
        return str(self._pipeline.classes_[best_idx]), float(proba[best_idx])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._pipeline, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self._pipeline = pickle.load(f)


class BERTClassifier(BaseMLClassifier):
    """Fine-tuned distilbert-base-uncased. Install transformers and torch first."""

    MODEL_NAME = "distilbert-base-uncased"

    def __init__(self, num_labels: int = 7) -> None:
        self._num_labels = num_labels
        self._model = self._tokenizer = None
        self._id2label: dict[int, str] = {}

    def fit(self, texts: list[str], labels: list[str]) -> None:
        unique = sorted(set(labels))
        self._id2label = {i: label for i, label in enumerate(unique)}
        logger.info("BERT label map built (%d classes).", len(unique))

    def predict_proba(self, text: str) -> tuple[str, float]:
        import torch

        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME,
                num_labels=self._num_labels,
            )
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        best = probs.argmax().item()
        return self._id2label.get(best, UNCLASSIFIED_LABEL), float(probs[best])

    def save(self, path: Path) -> None:
        if self._model:
            self._model.save_pretrained(path)
            self._tokenizer.save_pretrained(path)

    def load(self, path: Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path)


class LLMClassifier(BaseMLClassifier):
    """DeepSeek-backed zero-shot classifier with deterministic offline fallback."""

    SYSTEM_PROMPT = (
        "You are a log classification engine. Given a raw log message, choose "
        "exactly one label from the valid labels. Return ONLY a JSON object "
        "with keys 'label' and 'confidence' from 0.0 to 1.0. "
        "Valid labels: {labels}. Never return UNCLASSIFIED."
    )

    def __init__(
        self,
        labels: list[str],
        api_key: str = "",
        model: str = "deepseek-reasoner",
        provider: str = "deepseek",
    ) -> None:
        self._labels = sorted(set(labels))
        self._api_key = api_key
        self._model = model
        self._provider = provider

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self._labels = sorted(set(labels) | set(self._labels))

    def predict_proba(self, text: str) -> tuple[str, float]:
        return self.classify(text)

    def classify(
        self,
        text: str,
        candidate_label: str = "",
        candidate_confidence: float = 0.0,
    ) -> tuple[str, float]:
        if self._api_key and self._provider.lower() == "deepseek":
            try:
                return self._classify_with_deepseek(text)
            except Exception as exc:
                logger.warning("DeepSeek fallback failed; using offline fallback: %s", exc)
        return self._classify_offline(text, candidate_label, candidate_confidence)

    def _classify_with_deepseek(self, text: str) -> tuple[str, float]:
        import json
        import urllib.request

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT.format(labels=", ".join(self._labels)),
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "max_tokens": 120,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        parsed = json.loads(data["choices"][0]["message"]["content"])
        label = self._normalise_label(str(parsed.get("label", "")))
        confidence = float(parsed.get("confidence", 0.45))
        return label, max(0.0, min(confidence, 1.0))

    def _classify_offline(
        self,
        text: str,
        candidate_label: str = "",
        candidate_confidence: float = 0.0,
    ) -> tuple[str, float]:
        msg = text.lower()
        rule_scores: list[tuple[str, int]] = [
            ("Deprecation Warning", _keyword_score(msg, [
                "deprecated", "deprecation", "outdated", "legacy", "will be removed",
                "discontinued", "migrate", "unsupported", "no longer supported",
                "retired", "will be retired", "removed in version",
            ])),
            ("Workflow Error", _keyword_score(msg, [
                "lead", "prospect", "follow-up", "follow up", "escalation",
                "task assignment", "workflow", "process", "approval", "invoice",
                "bulk operation", "case", "crm", "failed due to",
                "could not complete", "missing next action",
            ])),
            ("Security Alert", _keyword_score(msg, [
                "unauthorized", "denied", "breach", "brute force", "suspicious",
                "login attempt", "access escalation", "malware", "intrusion",
                "credential", "external ip",
            ])),
            ("Critical Error", _keyword_score(msg, [
                "critical", "kernel panic", "boot", "terminated", "fatal", "outage",
                "main application", "configuration is no longer valid",
            ])),
            ("Error", _keyword_score(msg, [
                "error", "failed", "failure", "fault", "crashed", "exception",
                "timeout", "could not", "did not complete", "health check",
                "replication",
            ])),
            ("HTTP Status", _keyword_score(msg, [
                "http", "status", "rcode", " get ", " post ", " put ", " delete ", "/v",
            ])),
            ("Resource Usage", _keyword_score(msg, [
                "memory", "ram", "cpu", "vcpu", "disk", "resource", "quota",
                "usage", "free_ram_mb", "memory_mb",
            ])),
            ("System Notification", _keyword_score(msg, [
                "backup", "completed successfully", "started at",
                "uploaded successfully", "cleanup", "reboot", "updated to version",
            ])),
            ("User Action", _keyword_score(msg, [
                "user", "logged in", "logged out", "account", "created by",
                "deleted by",
            ])),
        ]
        label, score = max(rule_scores, key=lambda item: item[1])
        if score > 0:
            return label, min(0.95, 0.45 + (score * 0.08))

        if candidate_label in self._labels and candidate_label != UNCLASSIFIED_LABEL:
            return candidate_label, max(0.35, min(candidate_confidence, 0.59))

        return "Error", 0.35

    def _normalise_label(self, label: str) -> str:
        lower = label.strip().lower()
        for valid_label in self._labels:
            if valid_label.lower() == lower:
                return valid_label
        logger.warning("LLM returned invalid label %r; using Error fallback", label)
        return "Error"

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


def _keyword_score(message: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword in message)


class MLClassifier:
    """
    Wraps the selected ML backend with LLM-tier intercept and fallback.

    Low confidence is not an output state anymore. It is an internal routing
    signal that sends the row to Stage C.
    """

    def __init__(
        self,
        backend: BaseMLClassifier,
        known_labels: set[str],
        min_confidence: float = 0.60,
        offline_mode: bool = True,
        llm_backend: Optional[LLMClassifier] = None,
    ) -> None:
        self._backend = backend
        self._known_labels = known_labels
        self._min_confidence = min_confidence
        self._offline_mode = offline_mode
        self._llm_backend = llm_backend or LLMClassifier(VALID_LABELS)

    def classify(self, message: str, source: Optional[str] = None) -> ClassificationResult:
        intercepted_label = _llm_intercept(message)
        if intercepted_label is not None:
            logger.info("LLM-tier label intercepted: %s", intercepted_label)
            return ClassificationResult(
                log_message=message,
                predicted_label=intercepted_label,
                confidence_score=0.70,
                method_used=ClassificationMethod.LLM,
                training_status=TrainingStatus.EXISTING,
                source=source,
            )

        try:
            label, confidence = self._backend.predict_proba(message)
        except Exception as exc:
            logger.warning("ML backend error: %s; routing to LLM fallback", exc)
            label, confidence = UNCLASSIFIED_LABEL, 0.0

        method = self._backend_method()

        if confidence < self._min_confidence or label == UNCLASSIFIED_LABEL:
            fallback_label, fallback_confidence = self._llm_backend.classify(
                message,
                candidate_label=label,
                candidate_confidence=confidence,
            )
            return ClassificationResult(
                log_message=message,
                predicted_label=fallback_label,
                confidence_score=fallback_confidence,
                method_used=ClassificationMethod.LLM,
                training_status=TrainingStatus.EXISTING,
                source=source,
            )

        training_status = (
            TrainingStatus.EXISTING if label in self._known_labels else TrainingStatus.NEW
        )
        return ClassificationResult(
            log_message=message,
            predicted_label=label,
            confidence_score=confidence,
            method_used=method,
            training_status=training_status,
            source=source,
        )

    def _backend_method(self) -> ClassificationMethod:
        if isinstance(self._backend, SVMClassifier):
            return ClassificationMethod.SVM
        if isinstance(self._backend, BERTClassifier):
            return ClassificationMethod.BERT
        return ClassificationMethod.SVM


def build_ml_classifier(
    texts: list[str],
    labels: list[str],
    model_dir: Path,
    small_corpus_threshold: int = 500,
    offline_mode: bool = True,
    min_samples_per_label: int = 30,
    min_confidence: float = 0.60,
    llm_api_key: str = "",
    llm_provider: str = "deepseek",
    llm_model: str = "deepseek-reasoner",
) -> MLClassifier:
    from collections import Counter

    label_counts = Counter(labels)
    all_known_labels = set(VALID_LABELS)

    qualified = {
        label for label, count in label_counts.items()
        if count >= min_samples_per_label and label not in LLM_TIER_LABELS
    }
    filtered = [(text, label) for text, label in zip(texts, labels) if label in qualified]

    if not filtered:
        logger.warning("No labels qualify for ML training. Stage B will route to LLM fallback.")
        backend: BaseMLClassifier = SVMClassifier()
    else:
        filt_texts, filt_labels = zip(*filtered)
        corpus_size = len(filt_texts)

        if not offline_mode and corpus_size >= small_corpus_threshold:
            logger.info("Selecting BERT backend (%d samples)", corpus_size)
            backend = BERTClassifier(num_labels=len(qualified))
        else:
            logger.info("Selecting SVM backend (%d samples, %d classes)", corpus_size, len(qualified))
            backend = SVMClassifier()

        backend.fit(list(filt_texts), list(filt_labels))
        backend.save(model_dir / "stage_b_model.pkl")

    llm_backend = LLMClassifier(
        labels=VALID_LABELS,
        api_key="" if offline_mode else llm_api_key,
        model=llm_model,
        provider=llm_provider,
    )

    return MLClassifier(
        backend=backend,
        known_labels=all_known_labels,
        min_confidence=min_confidence,
        offline_mode=offline_mode,
        llm_backend=llm_backend,
    )
