"""
REST API — calibrated to synthetic_logs.csv schema.

Endpoints
─────────
  POST /classify              — classify a single log message (JSON)
  POST /classify/batch        — classify a list of log messages (JSON)
  POST /classify/upload       — classify a full CSV file (multipart upload)
  GET  /classify/export       — download the last batch output as CSV
  GET  /health                — pipeline readiness check

Postman quick-start
───────────────────
Single:
  POST http://localhost:8000/classify
  Body (raw JSON):
  { "log_message": "nova.osapi_compute.wsgi.server ... status: 200", "source": "ModernCRM" }

Batch JSON:
  POST http://localhost:8000/classify/batch
  Body: { "messages": [{"log_message": "...", "source": "..."}, ...] }

CSV Upload (Postman):
  POST http://localhost:8000/classify/upload
  Body → form-data → Key: file | Type: File | Value: <your CSV>
  Returns a downloadable classified CSV.
"""

from __future__ import annotations

import io
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import load_config
from .data_loader import load_inference_csv, load_labelled_csv, save_classified_csv
from .pipeline import LogClassificationPipeline, build_pipeline
from .schemas import BatchClassificationResponse, ClassificationRequest, ClassificationResult

logger = logging.getLogger(__name__)

_pipeline: Optional[LogClassificationPipeline] = None
_last_upload_results: list[ClassificationResult] = []
_last_upload_df: Optional[pd.DataFrame] = None


def _bootstrap_pipeline(cfg) -> LogClassificationPipeline:
    """Load training data and build the pipeline."""
    if not cfg.training_csv.exists():
        raise RuntimeError(
            f"Training CSV not found at {cfg.training_csv}. "
            "Run: python -m src.run_batch --mode train --input data/synthetic_logs.csv"
        )
    texts, labels, _ = load_labelled_csv(cfg.training_csv)
    logger.info("Loaded %d training samples from %s", len(texts), cfg.training_csv)
    return build_pipeline(texts, labels, config=cfg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    cfg = load_config()
    _pipeline = _bootstrap_pipeline(cfg)
    logger.info("Pipeline ready — %d labels in model", len(_pipeline._ml._known_labels))
    yield


app = FastAPI(
    title="Log Classifier API",
    description=(
        "Two-stage log classification pipeline for synthetic_logs.csv schema.\n\n"
        "Labels: HTTP Status · Resource Usage · System Notification · User Action · "
        "Security Alert · Critical Error · Error · Workflow Error · Deprecation Warning"
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Request/Response models ───────────────────────────────────────────────────

class BatchRequest(BaseModel):
    messages: list[ClassificationRequest]


class UploadSummary(BaseModel):
    total_rows: int
    classified: int
    unclassified: int
    label_distribution: dict[str, int]
    method_distribution: dict[str, int]
    download_url: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/classify",
    response_model=ClassificationResult,
    summary="Classify a single log message",
    tags=["Classification"],
)
async def classify_one(request: ClassificationRequest) -> ClassificationResult:
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    return _pipeline.classify_one(request.log_message, source=request.source)


@app.post(
    "/classify/batch",
    response_model=BatchClassificationResponse,
    summary="Classify a JSON list of log messages",
    tags=["Classification"],
)
async def classify_batch(request: BatchRequest) -> BatchClassificationResponse:
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    messages = [r.log_message for r in request.messages]
    sources  = [r.source      for r in request.messages]
    results  = _pipeline.classify_batch(messages, sources)
    classified = sum(1 for r in results if r.predicted_label != "UNCLASSIFIED")
    return BatchClassificationResponse(
        total=len(results),
        classified=classified,
        unclassified=len(results) - classified,
        results=results,
    )


@app.post(
    "/classify/upload",
    response_model=UploadSummary,
    summary="Upload a CSV file and classify every log_message row",
    tags=["Classification"],
)
async def classify_upload(file: UploadFile = File(...)) -> UploadSummary:
    """
    Accepts a CSV with the synthetic_logs.csv schema:
      timestamp, source, log_message  (target_label and complexity are optional)

    Returns a summary JSON. Download the enriched CSV from GET /classify/export.

    Postman setup:
      Body → form-data
      Key: file   Type: File   Value: <select your CSV>
    """
    global _last_upload_results, _last_upload_df

    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready")

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    # Read upload into a temp file then use data_loader
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        df = load_inference_csv(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if len(df) == 0:
        raise HTTPException(400, "CSV has no valid rows with a log_message column.")

    # Classify
    results = _pipeline.classify_batch(
        df["log_message"].tolist(),
        df["source"].fillna("").tolist(),
    )

    # Store for /classify/export
    _last_upload_results = results
    _last_upload_df = df

    # Save enriched CSV to output dir
    out_path = load_config().output_csv
    save_classified_csv(df, results, out_path)

    # Build summary
    from collections import Counter
    labels  = Counter(r.predicted_label  for r in results)
    methods = Counter(r.method_used.value for r in results)
    classified = sum(1 for r in results if r.predicted_label != "UNCLASSIFIED")

    return UploadSummary(
        total_rows=len(results),
        classified=classified,
        unclassified=len(results) - classified,
        label_distribution=dict(labels.most_common()),
        method_distribution=dict(methods.most_common()),
        download_url="/classify/export",
    )


@app.get(
    "/classify/export",
    summary="Download the last classified CSV",
    tags=["Export"],
)
async def export_csv(limit: int = Query(default=5000, le=50000)):
    """
    Returns the enriched CSV from the last /classify/upload call.
    Columns: timestamp | source | log_message | predicted_label |
             confidence_score | method_used | training_status | classified_at
    """
    out_path = load_config().output_csv
    if not out_path.exists():
        raise HTTPException(404, "No classified output found. Run POST /classify/upload first.")

    lines: list[str] = []
    with open(out_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > limit:
                break
            lines.append(line)

    return StreamingResponse(
        iter(lines),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=classified_{out_path.name}"},
    )


@app.get("/health", summary="Pipeline health check", tags=["Ops"])
async def health():
    return {
        "status": "ok",
        "pipeline_ready": _pipeline is not None,
        "known_labels": sorted(_pipeline._ml._known_labels) if _pipeline else [],
    }
