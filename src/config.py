"""
Pipeline configuration — all thresholds calibrated to synthetic_logs.csv.

Key findings from data analysis:
  - HTTP Status + Resource Usage: 100% nova.* → regex confidence 0.99
  - System Notification + User Action: rigid templates → regex confidence 0.98
  - Security Alert (371), Critical Error (161), Error (177): SVM/BERT territory
  - Workflow Error (4), Deprecation Warning (3): below min_samples → LLM fallback
  - Total training corpus: ~2000 rows → SVM by default; upgrade to BERT at 500+
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR  = BASE_DIR / "models"

for _d in (DATA_DIR, OUTPUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class ThresholdConfig:
    # Stage A
    regex_high_confidence: float = 0.90   # accept immediately; structural rules hit 0.98–0.99
    regex_low_confidence:  float = 0.50   # below this → Stage B

    # Stage B
    ml_min_confidence:      float = 0.60  # below this → LLM fallback
    min_samples_per_label:  int   = 30    # Workflow Error(4) + Deprecation Warning(3) excluded
                                          # Override: LOG_MIN_SAMPLES env var

    # Model backend selection
    # Dataset has ~2000 training rows → SVM is default (fast, offline-safe)
    # Switch to bert when corpus exceeds this:
    small_corpus_threshold: int = 500


@dataclass
class PipelineConfig:
    thresholds:   ThresholdConfig = field(default_factory=ThresholdConfig)
    training_csv: Path = DATA_DIR / "training_data.csv"
    output_csv:   Path = OUTPUT_DIR / "classifications.csv"
    log_file:     Path = OUTPUT_DIR / "audit.log"
    model_backend: str = "auto"       # "svm" | "bert" | "llm" | "auto"
    offline_mode:  bool = True
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-reasoner"
    llm_api_key: str = ""


def load_config() -> PipelineConfig:
    cfg = PipelineConfig()
    if (v := os.getenv("LOG_MIN_SAMPLES")):
        cfg.thresholds.min_samples_per_label = int(v)
    if (v := os.getenv("LOG_MODEL_BACKEND")):
        cfg.model_backend = v
    if (v := os.getenv("LOG_OFFLINE")):
        cfg.offline_mode = v.lower() in ("1", "true", "yes")
    if (v := os.getenv("LOG_LLM_PROVIDER")):
        cfg.llm_provider = v
    if (v := os.getenv("LOG_LLM_MODEL")):
        cfg.llm_model = v
    if (v := os.getenv("DEEPSEEK_API_KEY")):
        cfg.llm_api_key = v
    return cfg
