"""
Data schemas for the log classification pipeline.
Labels and complexity tiers derived from synthetic_logs.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ClassificationMethod(str, Enum):
    REGEX = "regex"
    SVM = "svm"
    BERT = "bert"
    LLM = "llm"
    UNCLASSIFIED = "unclassified"


class TrainingStatus(str, Enum):
    EXISTING = "existing"
    NEW = "new"
    UNKNOWN = "unknown"


class LogCategory(str, Enum):
    """
    The 9 labels present in synthetic_logs.csv, plus UNCLASSIFIED.

    Complexity tier (from dataset):
      regex → HTTP Status, Resource Usage, System Notification, User Action
      bert  → Security Alert, Critical Error, Error
      llm   → Workflow Error, Deprecation Warning  (< 10 samples)
    """
    HTTP_STATUS         = "HTTP Status"
    RESOURCE_USAGE      = "Resource Usage"
    SYSTEM_NOTIFICATION = "System Notification"
    USER_ACTION         = "User Action"
    SECURITY_ALERT      = "Security Alert"
    CRITICAL_ERROR      = "Critical Error"
    ERROR               = "Error"
    WORKFLOW_ERROR      = "Workflow Error"
    DEPRECATION_WARNING = "Deprecation Warning"
    UNCLASSIFIED        = "UNCLASSIFIED"


REGEX_TIER_LABELS = {"HTTP Status", "Resource Usage", "System Notification", "User Action"}
BERT_TIER_LABELS  = {"Security Alert", "Critical Error", "Error"}
LLM_TIER_LABELS   = {"Workflow Error", "Deprecation Warning"}


class TrainingSample(BaseModel):
    log_message: str = Field(..., min_length=1)
    label: str
    source: Optional[str] = None

    @field_validator("log_message")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class ClassificationRequest(BaseModel):
    log_message: str = Field(..., min_length=1)
    source: Optional[str] = None


class ClassificationResult(BaseModel):
    log_message: str
    predicted_label: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    method_used: ClassificationMethod
    training_status: TrainingStatus
    source: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_csv_row(self) -> dict:
        return {
            "log_message": self.log_message,
            "predicted_label": self.predicted_label,
            "confidence_score": round(self.confidence_score, 4),
            "method_used": self.method_used.value,
            "training_status": self.training_status.value,
            "source": self.source or "",
            "timestamp": self.timestamp.isoformat(),
        }


class BatchClassificationResponse(BaseModel):
    total: int
    classified: int
    unclassified: int
    results: list[ClassificationResult]


helper_functions = {
    "REGEX_TIER_LABELS": REGEX_TIER_LABELS,
    "BERT_TIER_LABELS": BERT_TIER_LABELS,
    "LLM_TIER_LABELS": LLM_TIER_LABELS,
}   