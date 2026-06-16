# Add Sliding Window Model to Dashboard

Wire up a new sliding window model checkpoint so it appears fully in the dashboard (overlays, legend, bone exclusion, threshold reset, comparison table).

**Usage:** `/add-sw-model v7` — where `v7` is the version tag for the new model.

The checkpoint must already exist at `models/best_model_slidingWindow_$ARGUMENTS.pth`.

---

## Steps

### 1 — Verify checkpoint

Check that `models/best_model_slidingWindow_$ARGUMENTS.pth` exists and is a valid PyTorch file:

```bash
ls -lh models/best_model_slidingWindow_$ARGUMENTS.pth
.venv/bin/python -c "
import torch
ck = torch.load('models/best_model_slidingWindow_$ARGUMENTS.pth', map_location='cpu')
print('keys:', list(ck.keys()))
print('epoch:', ck.get('epoch'))
print('val_dice:', ck.get('val_dice'))
"
```

If the file doesn't exist or fails to load, stop and report the error to the user.

### 2 — Inspect current versions already registered

Read `api/main.py` and `app_viewer.py` to understand what versions exist (e.g., v4, v5, v6) so you copy the exact same pattern without conflicts.

### 3 — Update `api/main.py`

Three additions, each following the same pattern as the previous version:

**a) In the `lifespan` function** — add a loading block after the last existing SW model block:
```python
    logger.info("Loading Small3DUNet (sliding window $ARGUMENTS) checkpoint...")
    try:
        model_sw_$ARGUMENTS, device_sw_$ARGUMENTS = load_model_sw("models/best_model_slidingWindow_$ARGUMENTS.pth")
        _state["model_sw_$ARGUMENTS"] = model_sw_$ARGUMENTS
        _state["device_sw_$ARGUMENTS"] = device_sw_$ARGUMENTS
        logger.info(f"Sliding window $ARGUMENTS model loaded on {device_sw_$ARGUMENTS}")
    except FileNotFoundError as e:
        logger.warning(f"SW $ARGUMENTS model not found: {e}. /segment/sw_$ARGUMENTS will return 503.")
```

**b) In the `/health` endpoint** — add `model_sw_$ARGUMENTS_loaded="model_sw_$ARGUMENTS" in _state` to the `HealthResponse(...)` call. Also add the corresponding field to `api/schemas.py` `HealthResponse` if it doesn't already accept arbitrary extra fields.

**c) Add a new POST endpoint** after the last SW endpoint:
```python
@app.post(
    "/segment/sw_$ARGUMENTS",
    response_model=SegmentationResponse,
    summary="Segment lesion — sliding window model ($ARGUMENTS)",
    response_description="Binary mask + volume (mL) + hemisphere. Native H×W resolution.",
)
async def segment_sw_$ARGUMENTS(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
) -> SegmentationResponse:
    if "model_sw_$ARGUMENTS" not in _state:
        raise HTTPException(status_code=503, detail="SW $ARGUMENTS model not loaded. Check models/best_model_slidingWindow_$ARGUMENTS.pth.")
    file_bytes = await file.read()
    _validate_upload(file, file_bytes)
    try:
        result = predict_from_bytes_sw(
            file_bytes, _state["model_sw_$ARGUMENTS"], _state["device_sw_$ARGUMENTS"],
            model_version="sw-$ARGUMENTS",
            threshold=threshold,
        )
    except Exception as exc:
        logger.exception("Prediction failed (/segment/sw_$ARGUMENTS)")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
    return SegmentationResponse(**result)
```

### 4 — Update `api/schemas.py`

Check the `HealthResponse` model. If it lists individual fields for each SW version (e.g., `model_sw_v5_loaded`, `model_sw_v6_loaded`), add the new field:
```python
model_sw_$ARGUMENTS_loaded: bool = False
```

### 5 — Update `app_viewer.py`

**a) `MODEL_OPTIONS`** — insert a new entry at the top (v$ARGUMENTS is the newest, so it goes first). Use the actual Dice and epoch from the checkpoint's `val_dice` key. If not available, mark it as `~?`:
```python
"Sliding Window $ARGUMENTS — Dice ~? · <brief description> · ~20s": "/segment/sw_$ARGUMENTS",
```

**b) `MODEL_DEFAULT_THRESHOLD`** — add a default threshold. Start with `0.30` (same as v6); adjust after testing:
```python
"/segment/sw_$ARGUMENTS": 0.30,
```

**c) Model comparison table** inside the `st.expander("Model comparison")` block — add a row:
```
| SW $ARGUMENTS | ~? | 650×650 native | ~20s | <brief description> |
```

### 6 — Verify syntax

```bash
.venv/bin/python -c "import ast; ast.parse(open('api/main.py').read()); print('main.py OK')"
.venv/bin/python -c "import ast; ast.parse(open('app_viewer.py').read()); print('app_viewer.py OK')"
```

### 7 — Restart the API

```bash
pkill -f "uvicorn api.main" 2>/dev/null; sleep 1
PYTHONPATH=. .venv/bin/uvicorn api.main:app --reload --port 8000 &
sleep 4 && curl -s http://localhost:8000/health
```

Confirm that `model_sw_$ARGUMENTS_loaded: true` appears in the health response.

### 8 — Smoke-test the new endpoint

Pick the first scan file from `data/raw/images/` and run a quick inference:

```bash
.venv/bin/python - << 'EOF'
import requests, base64, numpy as np
fname = sorted(__import__('os').listdir("data/raw/images"))[0]
with open(f"data/raw/images/{fname}", "rb") as f:
    fb = f.read()
r = requests.post(
    "http://localhost:8000/segment/sw_$ARGUMENTS",
    files={"file": (fname, fb, "application/octet-stream")},
    params={"threshold": 0.3}, timeout=180,
)
r.raise_for_status()
res = r.json()
mask = np.frombuffer(base64.b64decode(res["mask_base64"]), dtype=np.uint8).reshape(res["mask_shape"])
slices = np.where(mask.sum(axis=(1,2)) > 0)[0].tolist()
print(f"vol={res['lesion_volume_ml']} mL  voxels={res['lesion_voxel_count']}  slices={slices[:6]}")
EOF
```

### 9 — Report

Tell the user:
- Checkpoint loaded: epoch and val_dice found in the file
- Files changed: `api/main.py`, `api/schemas.py` (if needed), `app_viewer.py`
- Health check result
- Smoke-test result (volume and slices with predictions)
- Default threshold used (suggest adjusting if smoke-test returns 0 voxels)
