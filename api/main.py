"""
FastAPI application for brain CT lesion segmentation.

Run locally:
    PYTHONPATH=. uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health        → service + model status
    POST /segment/mresize → Small3DUNet with global resize (64×128×128)
    POST /segment/sw    → Small3DUNet with sliding window (native H×W resolution)
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from api.inference import load_model, predict_from_bytes
from api.inference_sw import load_model_sw, predict_from_bytes_sw
from api.schemas import HealthResponse, SegmentationResponse

SAMPLE_SCAN = Path(__file__).parent.parent / "data" / "sample" / "demo.nii.gz"

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

    logger.info("Loading Small3DUNet (sliding window v5) checkpoint...")
    try:
        model_sw_v5, device_sw_v5 = load_model_sw("models/best_model_slidingWindow_v5.pth")
        _state["model_sw_v5"] = model_sw_v5
        _state["device_sw_v5"] = device_sw_v5
        logger.info(f"Sliding window v5 model loaded on {device_sw_v5}")
    except FileNotFoundError as e:
        logger.warning(f"SW v5 model not found: {e}. /segment/sw_v5 will return 503.")

    logger.info("Loading Small3DUNet (sliding window v6) checkpoint...")
    try:
        model_sw_v6, device_sw_v6 = load_model_sw("models/best_model_slidingWindow_v6.pth")
        _state["model_sw_v6"] = model_sw_v6
        _state["device_sw_v6"] = device_sw_v6
        logger.info(f"Sliding window v6 model loaded on {device_sw_v6}")
    except FileNotFoundError as e:
        logger.warning(f"SW v6 model not found: {e}. /segment/sw_v6 will return 503.")

    logger.info("Loading Small3DUNet (sliding window v7) checkpoint...")
    try:
        model_sw_v7, device_sw_v7 = load_model_sw("models/best_model_slidingWindow_v7.pth")
        _state["model_sw_v7"] = model_sw_v7
        _state["device_sw_v7"] = device_sw_v7
        logger.info(f"Sliding window v7 model loaded on {device_sw_v7}")
    except FileNotFoundError as e:
        logger.warning(f"SW v7 model not found: {e}. /segment/sw_v7 will return 503.")

    logger.info("Loading Small3DUNet (sliding window v8) checkpoint...")
    try:
        model_sw_v8, device_sw_v8 = load_model_sw("models/best_model_slidingWindow_v8_dice.pth")
        _state["model_sw_v8"] = model_sw_v8
        _state["device_sw_v8"] = device_sw_v8
        logger.info(f"Sliding window v8 model loaded on {device_sw_v8}")
    except FileNotFoundError as e:
        logger.warning(f"SW v8 model not found: {e}. /segment/sw_v8 will return 503.")

    logger.info("Loading Small3DUNet (sliding window v9) checkpoint...")
    try:
        model_sw_v9, device_sw_v9 = load_model_sw("models/best_model_slidingWindow_v9_dice.pth")
        _state["model_sw_v9"] = model_sw_v9
        _state["device_sw_v9"] = device_sw_v9
        logger.info(f"Sliding window v9 model loaded on {device_sw_v9}")
    except FileNotFoundError as e:
        logger.warning(f"SW v9 model not found: {e}. /segment/sw_v9 will return 503.")

    logger.info("Loading Small3DUNet (sliding window v10) checkpoint...")
    try:
        model_sw_v10, device_sw_v10 = load_model_sw("models/best_model_slidingWindow_v10_dice.pth")
        _state["model_sw_v10"] = model_sw_v10
        _state["device_sw_v10"] = device_sw_v10
        logger.info(f"Sliding window v10 model loaded on {device_sw_v10}")
    except FileNotFoundError as e:
        logger.warning(f"SW v10 model not found: {e}. /segment/sw_v10 will return 503.")

    logger.info("Loading Small3DUNet (sliding window v11) checkpoint...")
    try:
        model_sw_v11, device_sw_v11 = load_model_sw("models/best_model_slidingWindow_v11.pth")
        _state["model_sw_v11"] = model_sw_v11
        _state["device_sw_v11"] = device_sw_v11
        logger.info(f"Sliding window v11 model loaded on {device_sw_v11}")
    except FileNotFoundError as e:
        logger.warning(f"SW v11 model not found: {e}. /segment/sw_v11 will return 503.")

    yield
    _state.clear()
    logger.info("Models unloaded.")


app = FastAPI(
    title="Brain CT Lesion Segmentation API",
    description=(
        "Upload a brain CT scan in NIfTI format (.nii or .nii.gz) and receive "
        "a binary lesion segmentation mask with derived clinical metrics.\n\n"
        "Two models are available for comparison:\n"
        "- `/segment/mresize` — global resize to (64×128×128), fast\n"
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
        model_sw_v5_loaded="model_sw_v5" in _state,
        model_sw_v6_loaded="model_sw_v6" in _state,
        model_sw_v7_loaded="model_sw_v7" in _state,
        model_sw_v8_loaded="model_sw_v8" in _state,
        model_sw_v9_loaded="model_sw_v9" in _state,
        model_sw_v10_loaded="model_sw_v10" in _state,
        model_sw_v11_loaded="model_sw_v11" in _state,
    )


@app.get(
    "/segment/sample",
    response_model=SegmentationResponse,
    summary="Segment built-in demo scan (no upload needed)",
    response_description="Segmentation of the bundled sample CT (patient 085, 12 mL lesion).",
)
async def segment_sample() -> SegmentationResponse:
    """Run both models on the built-in demo scan and return the resize-model result.
    Use this to verify the API works without needing your own NIfTI file."""
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Resize model not loaded.")
    if not SAMPLE_SCAN.exists():
        raise HTTPException(status_code=404, detail=f"Sample scan not found at {SAMPLE_SCAN}.")
    try:
        file_bytes = SAMPLE_SCAN.read_bytes()
        result = predict_from_bytes(file_bytes, _state["model"], _state["device"])
    except Exception as exc:
        logger.exception("Prediction failed (/segment/sample)")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
    return SegmentationResponse(**result)


@app.post(
    "/segment/mresize",
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
    summary="Segment lesion — sliding window model (v4)",
    response_description="Binary mask + volume (mL) + hemisphere. Native H×W resolution preserved.",
)
async def segment_sw(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
) -> SegmentationResponse:
    if "model_sw" not in _state:
        raise HTTPException(status_code=503, detail="Sliding window model not loaded. Check models/best_model_slidingWindow.pth.")
    file_bytes = await file.read()
    _validate_upload(file, file_bytes)
    try:
        result = predict_from_bytes_sw(
            file_bytes, _state["model_sw"], _state["device_sw"],
            threshold=threshold,
        )
    except Exception as exc:
        logger.exception("Prediction failed (/segment/sw)")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
    return SegmentationResponse(**result)


@app.post(
    "/segment/sw_v5",
    response_model=SegmentationResponse,
    summary="Segment lesion — sliding window model (v5, bone suppression + augmentation)",
    response_description="Binary mask + volume (mL) + hemisphere. Native H×W resolution, reduced skull false positives.",
)
async def segment_sw_v5(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
) -> SegmentationResponse:
    if "model_sw_v5" not in _state:
        raise HTTPException(status_code=503, detail="SW v5 model not loaded. Check models/best_model_slidingWindow_v5.pth.")
    file_bytes = await file.read()
    _validate_upload(file, file_bytes)
    try:
        result = predict_from_bytes_sw(
            file_bytes, _state["model_sw_v5"], _state["device_sw_v5"],
            model_version="sw-v5",
            threshold=threshold,
        )
    except Exception as exc:
        logger.exception("Prediction failed (/segment/sw_v5)")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
    return SegmentationResponse(**result)


@app.post(
    "/segment/sw_v6",
    response_model=SegmentationResponse,
    summary="Segment lesion — sliding window model (v6, skull-strip + no elastic)",
    response_description="Binary mask + volume (mL) + hemisphere. Skull stripping pre-processing, no elastic augmentation.",
)
async def segment_sw_v6(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
) -> SegmentationResponse:
    if "model_sw_v6" not in _state:
        raise HTTPException(status_code=503, detail="SW v6 model not loaded. Check models/best_model_slidingWindow_v6.pth.")
    file_bytes = await file.read()
    _validate_upload(file, file_bytes)
    try:
        result = predict_from_bytes_sw(
            file_bytes, _state["model_sw_v6"], _state["device_sw_v6"],
            model_version="sw-v6",
            threshold=threshold,
        )
    except Exception as exc:
        logger.exception("Prediction failed (/segment/sw_v6)")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
    return SegmentationResponse(**result)


@app.post(
    "/segment/sw_v7",
    response_model=SegmentationResponse,
    summary="Segment lesion — sliding window model (v7)",
    response_description="Binary mask + volume (mL) + hemisphere. Native H×W resolution.",
)
async def segment_sw_v7(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
) -> SegmentationResponse:
    if "model_sw_v7" not in _state:
        raise HTTPException(status_code=503, detail="SW v7 model not loaded. Check models/best_model_slidingWindow_v7.pth.")
    file_bytes = await file.read()
    _validate_upload(file, file_bytes)
    try:
        result = predict_from_bytes_sw(
            file_bytes, _state["model_sw_v7"], _state["device_sw_v7"],
            model_version="sw-v7",
            threshold=threshold,
        )
    except Exception as exc:
        logger.exception("Prediction failed (/segment/sw_v7)")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
    return SegmentationResponse(**result)


@app.post(
    "/segment/sw_v8",
    response_model=SegmentationResponse,
    summary="Segment lesion — sliding window model (v8, 3D skull exclusion + dual checkpoint)",
    response_description="Binary mask + volume (mL) + hemisphere. Native H×W resolution.",
)
async def segment_sw_v8(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
) -> SegmentationResponse:
    if "model_sw_v8" not in _state:
        raise HTTPException(status_code=503, detail="SW v8 model not loaded. Check models/best_model_slidingWindow_v8_dice.pth.")
    file_bytes = await file.read()
    _validate_upload(file, file_bytes)
    try:
        result = predict_from_bytes_sw(
            file_bytes, _state["model_sw_v8"], _state["device_sw_v8"],
            model_version="sw-v8",
            threshold=threshold,
            skull_strip=True,
        )
    except Exception as exc:
        logger.exception("Prediction failed (/segment/sw_v8)")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
    return SegmentationResponse(**result)


@app.post(
    "/segment/sw_v9",
    response_model=SegmentationResponse,
    summary="Segment lesion — sliding window model (v9, SAM skull masks + largest-CC inference)",
    response_description="Binary mask + volume (mL) + hemisphere. Native H×W resolution.",
)
async def segment_sw_v9(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.3, ge=0.0, le=1.0),
) -> SegmentationResponse:
    if "model_sw_v9" not in _state:
        raise HTTPException(status_code=503, detail="SW v9 model not loaded. Check models/best_model_slidingWindow_v9_dice.pth.")
    file_bytes = await file.read()
    _validate_upload(file, file_bytes)
    try:
        result = predict_from_bytes_sw(
            file_bytes, _state["model_sw_v9"], _state["device_sw_v9"],
            model_version="sw-v9",
            threshold=threshold,
            skull_strip=True,
        )
    except Exception as exc:
        logger.exception("Prediction failed (/segment/sw_v9)")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
    return SegmentationResponse(**result)


@app.post(
    "/segment/sw_v10",
    response_model=SegmentationResponse,
    summary="Segment lesion — sliding window model (v10, SAM skull masks + pos_weight=5)",
    response_description="Binary mask + volume (mL) + hemisphere. Native H×W resolution.",
)
async def segment_sw_v10(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.3, ge=0.0, le=1.0),
) -> SegmentationResponse:
    if "model_sw_v10" not in _state:
        raise HTTPException(status_code=503, detail="SW v10 model not loaded. Check models/best_model_slidingWindow_v10_dice.pth.")
    file_bytes = await file.read()
    _validate_upload(file, file_bytes)
    try:
        result = predict_from_bytes_sw(
            file_bytes, _state["model_sw_v10"], _state["device_sw_v10"],
            model_version="sw-v10",
            threshold=threshold,
            skull_strip=True,
        )
    except Exception as exc:
        logger.exception("Prediction failed (/segment/sw_v10)")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
    return SegmentationResponse(**result)
