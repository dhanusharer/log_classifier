"""
data_loader.py — handles all CSV I/O for the pipeline.

The canonical file schema (synthetic_logs.csv):
  timestamp     – log event time (various formats; may be missing in inference files)
  source        – originating service (ModernCRM, BillingSystem, etc.)
  log_message   – raw log text to classify
  target_label  – ground-truth label (present in labelled files; absent in inference)
  complexity    – tier hint (regex/bert/llm; present in labelled files; optional)

Inference files sent by the client:
  Required : log_message
  Optional : timestamp, source  (passed through to output unchanged)
  Absent   : target_label, complexity  (these are what we produce)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# All column names we recognise as the log text column
_MESSAGE_ALIASES = {"log_message", "message", "log", "text", "raw_log", "msg"}


def _find_message_col(columns: list[str]) -> str:
    for col in columns:
        if col.lower() in _MESSAGE_ALIASES:
            return col
    raise ValueError(
        f"Cannot find a log message column. Expected one of {_MESSAGE_ALIASES}. "
        f"Got: {columns}"
    )


def load_labelled_csv(path: Path) -> tuple[list[str], list[str], list[str]]:
    """
    Load a labelled CSV (training or eval).
    Returns (log_messages, labels, sources).
    Rows with missing log_message or target_label are dropped.
    """
    df = pd.read_csv(path)
    msg_col = _find_message_col(df.columns.tolist())

    if "target_label" not in df.columns:
        raise ValueError(f"{path} has no 'target_label' column — use load_inference_csv instead.")

    before = len(df)
    df = df.dropna(subset=[msg_col, "target_label"])
    df[msg_col] = df[msg_col].astype(str).str.strip()
    df = df[df[msg_col].str.len() > 0]
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with missing log_message or target_label", dropped)

    sources = df["source"].fillna("").astype(str).tolist() if "source" in df.columns else [""] * len(df)
    return df[msg_col].tolist(), df["target_label"].tolist(), sources


def load_inference_csv(path: Path) -> pd.DataFrame:
    """
    Load an inference CSV (no target_label required).
    Returns the full DataFrame; caller classifies the log_message column.
    Missing rows are dropped; original columns are preserved for pass-through.
    """
    df = pd.read_csv(path)
    msg_col = _find_message_col(df.columns.tolist())

    # Normalise to 'log_message' internally
    if msg_col != "log_message":
        df = df.rename(columns={msg_col: "log_message"})

    before = len(df)
    df = df.dropna(subset=["log_message"])
    df["log_message"] = df["log_message"].astype(str).str.strip()
    df = df[df["log_message"].str.len() > 0].reset_index(drop=True)
    if before - len(df):
        logger.warning("Dropped %d empty rows", before - len(df))

    # Ensure standard passthrough columns exist
    for col in ("timestamp", "source"):
        if col not in df.columns:
            df[col] = ""

    return df


def save_classified_csv(df: pd.DataFrame, results: list, path: Path) -> Path:
    """
    Merge classification results back onto the original DataFrame rows
    and write to a CSV with the full output schema.

    Output columns:
      timestamp | source | log_message | predicted_label | confidence_score
      | method_used | training_status | classified_at
    """
    from datetime import datetime, timezone

    output_rows = []
    for i, result in enumerate(results):
        row = {
            "timestamp"       : df.at[i, "timestamp"] if "timestamp" in df.columns else "",
            "source"          : result.source or "",
            "log_message"     : result.log_message,
            "predicted_label" : result.predicted_label,
            "confidence_score": round(result.confidence_score, 4),
            "method_used"     : result.method_used.value,
            "training_status" : result.training_status.value,
            "classified_at"   : datetime.now(timezone.utc).isoformat(),
        }
        output_rows.append(row)

    out_df = pd.DataFrame(output_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(path, index=False)
    logger.info("Saved %d classified rows → %s", len(out_df), path)
    return path




