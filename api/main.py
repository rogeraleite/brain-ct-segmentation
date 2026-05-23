"""
FastAPI application for brain CT lesion segmentation.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health   → service + model status
    POST /segment  → upload NIfTI file → segmentation mask + derived metrics
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.inference import load_model, predict_from_bytes
from api.schemas import HealthResponse, SegmentationResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB — a full brain CT is ~50–200 MB

# Shared state: populated at startup, cleared at shutdown
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, release at shutdown."""
    logger.info("Loading model checkpoint...")
    try:
        model, device = load_model()
        _state["model"] = model
        _state["device"] = device
        logger.info(f"Model loaded successfully on {device}")
    except FileNotFoundError as e:
        logger.warning(f"Model file not found: {e}. /segment will return 503.")
    yield
    _state.clear()
    logger.info("Model unloaded.")


app = FastAPI(
    title="Brain CT Lesion Segmentation API",
    description=(
        "Upload a brain CT scan in NIfTI format (.nii or .nii.gz) "
        "and receive a binary lesion segmentation mask along with "
        "derived clinical metrics: lesion volume (mL) and hemisphere."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, summary="Service health check")
async def health() -> HealthResponse:
    model_loaded = "model" in _state
    device_str = str(_state.get("device", "unavailable"))
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        device=device_str,
    )


@app.post(
    "/segment",
    response_model=SegmentationResponse,
    summary="Segment brain lesion in a CT scan",
    response_description="Binary mask + volume (mL) + hemisphere lateralization",
)
async def segment(file: UploadFile = File(..., description="NIfTI CT scan (.nii or .nii.gz)")) -> SegmentationResponse:
    if "model" not in _state:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check that models/best_model.pth exists and restart.",
        )

    filename = file.filename or ""
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{filename}'. Upload a .nii or .nii.gz NIfTI file.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes) / 1e6:.1f} MB). Maximum is 500 MB.",
        )

    try:
        result = predict_from_bytes(file_bytes, _state["model"], _state["device"])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    return SegmentationResponse(**result)
