"""
FastAPI application for brain CT lesion segmentation.

Run locally:
    PYTHONPATH=. uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health        → service + model status
    POST /segment       → Small3DUNet with global resize (64×128×128)
    POST /segment/sw    → Small3DUNet with sliding window (native H×W resolution)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from api.inference import load_model, predict_from_bytes
from api.inference_sw import load_model_sw, predict_from_bytes_sw
from api.schemas import HealthResponse, SegmentationResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both models at startup; release at shutdown."""
    logger.info("Loading Small3DUNet (resize) checkpoint...")
    try:
        model, device = load_model()
        _state["model"] = model
        _state["device"] = device
        logger.info(f"Resize model loaded on {device}")
    except FileNotFoundError as e:
        logger.warning(f"Resize model not found: {e}. /segment will return 503.")

    logger.info("Loading Small3DUNet (sliding window) checkpoint...")
    try:
        model_sw, device_sw = load_model_sw()
        _state["model_sw"] = model_sw
        _state["device_sw"] = device_sw
        logger.info(f"Sliding window model loaded on {device_sw}")
    except FileNotFoundError as e:
        logger.warning(f"Sliding window model not found: {e}. /segment/sw will return 503.")

    yield
    _state.clear()
    logger.info("Models unloaded.")


app = FastAPI(
    title="Brain CT Lesion Segmentation API",
    description=(
        "Upload a brain CT scan in NIfTI format (.nii or .nii.gz) and receive "
        "a binary lesion segmentation mask with derived clinical metrics.\n\n"
        "Two models are available for comparison:\n"
        "- `/segment` — global resize to (64×128×128), fast\n"
        "- `/segment/sw` — sliding window at native H×W resolution (650×650), higher detail"
    ),
    version="1.1.0",
    lifespan=lifespan,
)


def _validate_upload(file: UploadFile, file_bytes: bytes) -> None:
    filename = file.filename or ""
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{filename}'. Upload a .nii or .nii.gz NIfTI file.",
        )
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes) / 1e6:.1f} MB). Maximum is 500 MB.",
        )


@app.get("/health", response_model=HealthResponse, summary="Service health check")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded="model" in _state,
        device=str(_state.get("device", "unavailable")),
        model_sw_loaded="model_sw" in _state,
    )


@app.post(
    "/segment",
    response_model=SegmentationResponse,
    summary="Segment lesion — global resize model",
    response_description="Binary mask + volume (mL) + hemisphere. Input resized to 64×128×128.",
)
async def segment(file: UploadFile = File(...)) -> SegmentationResponse:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Resize model not loaded. Check models/best_model_small3DUNet.pth.")
    file_bytes = await file.read()
    _validate_upload(file, file_bytes)
    try:
        result = predict_from_bytes(file_bytes, _state["model"], _state["device"])
    except Exception as exc:
        logger.exception("Prediction failed (/segment)")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
    return SegmentationResponse(**result)


@app.post(
    "/segment/sw",
    response_model=SegmentationResponse,
    summary="Segment lesion — sliding window model",
    response_description="Binary mask + volume (mL) + hemisphere. Native H×W resolution preserved.",
)
async def segment_sw(file: UploadFile = File(...)) -> SegmentationResponse:
    if "model_sw" not in _state:
        raise HTTPException(status_code=503, detail="Sliding window model not loaded. Check models/best_model_slidingWindow.pth.")
    file_bytes = await file.read()
    _validate_upload(file, file_bytes)
    try:
        result = predict_from_bytes_sw(file_bytes, _state["model_sw"], _state["device_sw"])
    except Exception as exc:
        logger.exception("Prediction failed (/segment/sw)")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
    return SegmentationResponse(**result)
