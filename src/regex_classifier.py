"""
Stage A — Regex classifier calibrated to synthetic_logs.csv.

Coverage summary (validated against 2,410 rows):
  Label              Regex cov.   Contam.   Remaining → Stage B
  ─────────────────────────────────────────────────────────────
  HTTP Status          100 %        0        —
  Resource Usage       100 %        0        —
  System Notification  100 %        0        —
  User Action          100 %        0        —
  Security Alert        50 %        0        ~50 % → SVM
  Critical Error        50 %        0        ~50 % → SVM
  Error                 35 %        0        ~65 % → SVM
  Workflow Error         0 %        0        all  → LLM-intercept
  Deprecation Warning    0 %        0        all  → LLM-intercept
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .schemas import ClassificationMethod, ClassificationResult, TrainingStatus

logger = logging.getLogger(__name__)


@dataclass
class RegexRule:
    pattern: re.Pattern
    label: str
    base_confidence: float
    corroborating: list[re.Pattern] = field(default_factory=list)
    CORROBORATION_BONUS: float = 0.04


_RAW_RULES: list[tuple] = [

    # ── 100 % structural — zero contamination ─────────────────────────────────

    # HTTP Status: ALL rows are nova.osapi or nova.metadata wsgi messages
    (
        r"nova\.(osapi_compute|metadata)\.wsgi\.server",
        "HTTP Status", 0.99,
        [r'(?:GET|POST|PUT|DELETE|HEAD|PATCH)\s+/', r'status[:\s]+\d{3}|HTTP/\d'],
    ),

    # Resource Usage: ALL rows are nova.compute.claims or nova.compute.resource_tracker
    (
        r"nova\.compute\.(claims|resource_tracker)",
        "Resource Usage", 0.99,
        [r"(?i)(memory|disk|cpu|vcpu|ram|resource|claim|free|used|total)"],
    ),

    # System Notification: rigid templates (7 canonical patterns, 100 % coverage)
    (
        r"(?i)("
        r"backup\s+(completed|started|ended|initiated|finished)|"
        r"disk\s+cleanup\s+completed|"
        r"File\s+\S+\s+uploaded\s+successfully|"
        r"system\s+(reboot\s+initiated|updated\s+to\s+version)|"
        r"reboot\s+initiated\s+by"
        r")",
        "System Notification", 0.98,
        [],
    ),

    # User Action: rigid templates (3 canonical patterns, 100 % coverage)
    (
        r"(?i)("
        r"User\s+\w+\s+logged\s+(in|out)|"
        r"Account\s+with\s+ID\s+\d+\s+created\s+by"
        r")",
        "User Action", 0.98,
        [],
    ),

    # ── Partial-coverage rules — zero contamination ───────────────────────────
    # Unmatched rows from these labels fall through to Stage B (SVM).

    # Security Alert: covers ~50 % of messages; 0 contamination confirmed
    (
        r"(?i)("
        r"unauthorized\s+(access|login|entry|credentials|attempt)|"
        r"brute.force\s+(login|attack)|"
        r"suspicious\s+(login|activity|parameters)|"
        r"intrusion\s+(detection|detected)|"
        r"privilege.escal|admin.access.escal|"
        r"security\s+(breach|threat|violation)|"
        r"(access|login)\s+(denied|blocked|flagged)|"
        r"invalid\s+credentials\s+for\s+user"
        r")",
        "Security Alert", 0.93,
        [r"(?i)(user\s+\w+|account\s+\w+|IP\s+\d|from\s+\d)"],
    ),

    # Critical Error: covers ~50 % of messages; 0 contamination confirmed
    (
        r"(?i)("
        r"kernel\s+(panic|issue|error|failure|malfunction)|"
        r"boot\s+(process|sequence)\s+(fail|terminat|abort|interrupt|stop)|"
        r"configuration\s+(corrupt|invalid|error|malfunction|integrity|no\s+longer\s+valid)|"
        r"system\s+config.{0,20}(invalid|no\s+longer\s+valid|corrupt)|"
        r"critical\s+(system\s+(crash|fail)|fail|bug|error).{0,30}(core|main|central|application)|"
        r"system\s+component\s+(malfunction|failure|down)|"
        r"disk\s+fault|RAID"
        r")",
        "Critical Error", 0.92,
        [r"(?i)(component|device|element|equipment|unit)\s*(ID|#|\d)"],
    ),

    # Error: covers ~35 % of messages; 0 contamination confirmed
    # Email sub-pattern removed — causes 8 Critical Error false-positives; SVM handles it better
    (
        r"(?i)("
        r"shard\s+\d+.{0,20}(replication|synchronization|sync).{0,40}(fail|not\s+complet|unsuccess|ended\s+in)|"
        r"(data\s+)?(replication|synchronization)\s+(task\s+)?(fail|not\s+complet|unsuccess|issue\s+on)|"
        r"SSL\s+certificate.{0,30}(fail|invalid|prevent|issue|cause)|"
        r"server\s+\d+.{0,30}(crash|restart|shutdown).{0,20}(unexpect|without\s+warn|during)"
        r")",
        "Error", 0.91,
        [r"(?i)(shard|replication|SSL|server\s+\d)"],
    ),
]


def _compile(raw: list[tuple]) -> list[RegexRule]:
    return [
        RegexRule(
            pattern=re.compile(p),
            label=label,
            base_confidence=conf,
            corroborating=[re.compile(c) for c in corr],
        )
        for p, label, conf, corr in raw
    ]


class RegexClassifier:
    def __init__(
        self,
        rules: Optional[list[RegexRule]] = None,
        high_conf_gate: float = 0.90,
        low_conf_gate: float = 0.50,
    ) -> None:
        self.rules = rules or _compile(_RAW_RULES)
        self.high_conf_gate = high_conf_gate
        self.low_conf_gate = low_conf_gate

    def _score(self, message: str, rule: RegexRule) -> float:
        if not rule.pattern.search(message):
            return 0.0
        score = rule.base_confidence
        for corr in rule.corroborating:
            if corr.search(message):
                score = min(1.0, score + RegexRule.CORROBORATION_BONUS)
        return score

    def classify(
        self, message: str, source: Optional[str] = None
    ) -> Optional[ClassificationResult]:
        best_label: Optional[str] = None
        best_score: float = 0.0
        for rule in self.rules:
            score = self._score(message, rule)
            if score > best_score:
                best_score = score
                best_label = rule.label
        if best_score < self.low_conf_gate or best_label is None:
            logger.debug("Regex: no match (score=%.3f)", best_score)
            return None
        return ClassificationResult(
            log_message=message,
            predicted_label=best_label,
            confidence_score=best_score,
            method_used=ClassificationMethod.REGEX,
            training_status=TrainingStatus.EXISTING,
            source=source,
        )

    def add_rule(
        self,
        pattern: str,
        label: str,
        base_confidence: float,
        corroborating: Optional[list[str]] = None,
    ) -> None:
        self.rules.insert(0, RegexRule(
            pattern=re.compile(pattern),
            label=label,
            base_confidence=base_confidence,
            corroborating=[re.compile(c) for c in (corroborating or [])],
        ))
