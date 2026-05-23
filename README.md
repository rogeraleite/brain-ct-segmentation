# Brain CT Lesion Segmentation

A small end-to-end medical imaging project: a 3D U-Net trained on brain CT scans to segment lesions, exposed as a REST API, and packaged in Docker. Built in 3–4 days as a portfolio project to demonstrate medical imaging fundamentals, PyTorch, and clean ML engineering.

---

## Architecture

```
NIfTI file (.nii.gz)
        │
        ▼
[loader.py]          nibabel.load() → float32 (D,H,W) + voxel spacing (mm)
        │
        ▼
[transforms.py]      brain window [-5, 75] HU → normalize [0,1] → resize (64,128,128)
        │
        ▼
[dataset.py]         BrainCTDataset → (volume_tensor, mask_tensor) pairs
        │
        ▼
[unet.py]            Small3DUNet  encoder(16→32→64) + bottleneck(128)
                                  + decoder with skip connections
                                  + sigmoid output
        │
        ▼
[trainer.py]         Dice + BCE combined loss | AdamW | ReduceLROnPlateau
                     → best_model.pth (saved by val Dice)
        │
        ▼
[FastAPI /segment]   POST .nii.gz → {
                       "lesion_volume_ml": 12.4,
                       "hemisphere": "left",
                       "centroid_voxel": [32, 64, 45],
                       "mask_base64": "..."
                     }
        │
        ▼
[Docker]             docker compose up  →  uvicorn on :8000
```

---

## Quick Start

```bash
git clone <repo-url>
cd medical-imaging-portfolio

# Build and start the API
docker compose up --build

# Health check
curl http://localhost:8000/health

# Run inference on a NIfTI scan
curl -X POST http://localhost:8000/segment \
  -F "file=@/path/to/scan.nii.gz"

# Explore the interactive API docs
open http://localhost:8000/docs
```

> The container expects `models/best_model.pth` to be present.  
> See **Training** below to produce it, or download a pre-trained checkpoint.

---

## Dataset

**GTS.ai Brain CT Segmentation Dataset** — ~1,000 brain CT studies with expert segmentation masks covering 10 pathological categories (tumors, hemorrhages, lesions, and others).

- Format: NIfTI (`.nii.gz`)
- Request access: https://gts.ai/dataset-download/brain-ct-segmentation-dataset-1000-studies/

Expected directory layout after download:

```
data/raw/
├── images/   ← *.nii.gz  CT volumes
└── masks/    ← *.nii.gz  binary segmentation masks (same filename)
```

Split: 80% train / 20% validation, deterministic shuffle with `seed=42`.

---

## Preprocessing

Brain CT values are in **Hounsfield Units (HU)**. The preprocessing pipeline applies:

1. **Brain soft-tissue window** — clip to **[-5, 75] HU**  
   This range captures the clinically relevant structures:  
   - CSF: 0–10 HU | White matter: 25–30 | Gray matter: 35–40 | Hemorrhage: 50–80  
   Bone (>700 HU) and air (~-1000 HU) are excluded — they add no signal for lesion detection.

2. **Min-max normalization** to [0, 1]

3. **Resize to (64, 128, 128)** — trilinear for volumes, nearest-neighbor for masks  
   (Nearest-neighbor on masks is critical: linear interpolation creates fractional boundary  
   values that corrupt binary labels)

---

## Model Architecture

`Small3DUNet` — ~3M parameters.

| Stage | Block | Output shape (B=1) |
|---|---|---|
| Input | — | (1, 1, 64, 128, 128) |
| Encoder 1 | Conv3d 1→16, BN, ReLU × 2 | (1, 16, 64, 128, 128) |
| Pool 1 | MaxPool3d(2) | (1, 16, 32, 64, 64) |
| Encoder 2 | Conv3d 16→32, BN, ReLU × 2 | (1, 32, 32, 64, 64) |
| Pool 2 | MaxPool3d(2) | (1, 32, 16, 32, 32) |
| Encoder 3 | Conv3d 32→64, BN, ReLU × 2 | (1, 64, 16, 32, 32) |
| Pool 3 | MaxPool3d(2) | (1, 64, 8, 16, 16) |
| Bottleneck | Conv3d 64→128, BN, ReLU × 2 | (1, 128, 8, 16, 16) |
| Up 3 + skip | ConvTranspose3d + cat(enc3) + conv | (1, 64, 16, 32, 32) |
| Up 2 + skip | ConvTranspose3d + cat(enc2) + conv | (1, 32, 32, 64, 64) |
| Up 1 + skip | ConvTranspose3d + cat(enc1) + conv | (1, 16, 64, 128, 128) |
| Output | Conv3d(16→1, 1×1×1) + Sigmoid | (1, 1, 64, 128, 128) |

---

## Training

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/train.py \
  --data-root data/raw \
  --save-path models/best_model.pth \
  --epochs 50 \
  --batch-size 2 \
  --lr 1e-3
```

Device is auto-selected: MPS (Apple Silicon) → CUDA → CPU.  
Expected time: ~40–60 min on M-series Mac | ~2–4h on CPU.

---

## Results

| Metric | Value |
|---|---|
| Val Dice | _fill after training_ |
| Val Loss | _fill after training_ |
| Epochs | 50 |
| Best checkpoint | models/best_model.pth |

Training curves and sample predictions are in [`notebooks/03_model_training.ipynb`](notebooks/03_model_training.ipynb).

---

## API Reference

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/health` | GET | — | `{"status": "ok", "model_loaded": true, "device": "mps"}` |
| `/segment` | POST | `.nii` or `.nii.gz` file | `SegmentationResponse` (see below) |
| `/docs` | GET | — | Swagger UI |

**SegmentationResponse fields:**

| Field | Type | Description |
|---|---|---|
| `lesion_volume_ml` | float | Lesion volume in mL (voxel count × voxel size) |
| `hemisphere` | string | `"left"`, `"right"`, `"bilateral"`, or `"none"` |
| `centroid_voxel` | int[3] | `[D, H, W]` centroid of the lesion in the prediction volume |
| `lesion_voxel_count` | int | Raw lesion voxel count |
| `mask_shape` | int[3] | Shape of the returned mask |
| `mask_base64` | string | Base64-encoded binary mask (uint8, row-major) |

---

## Design Decisions

**Why 3D convolutions instead of 2D slice-by-slice?**  
Brain lesions are volumetric. A 2D model sees each axial slice independently, losing the inter-slice continuity that helps distinguish a small lesion from noise. 3D conv captures that context at the cost of more memory and slower training — a trade-off worth making for segmentation accuracy.

**Why Dice + BCE combined loss?**  
BCE alone diverges on class-imbalanced masks: brain lesions typically occupy <5% of voxels, so a model predicting all-zeros gets low BCE with zero learning signal. Dice loss directly optimises the overlap metric we care about, but is unstable early in training when all predictions are near 0.5. Combining them gives stable gradients throughout.

**Why the [-5, 75] HU window?**  
This is the standard brain soft-tissue window (center 35, width 80). It preserves the HU range where pathology lives (hemorrhage: 50–80, edema: 15–30) while discarding bone and air, which would otherwise dominate the histogram and cause the normalizer to waste dynamic range on irrelevant structures.

**Why hemisphere lateralization from the centroid?**  
A simple centroid-vs-midline comparison is clinically meaningful and fully deterministic — no additional model needed. The 10% midline margin avoids over-committing on lesions that straddle the midline. More sophisticated approaches (atlas registration, midsagittal plane detection) exist but are disproportionate for this scope.

**Why NIfTI as the primary format with a separate DICOM demo?**  
The training dataset ships as NIfTI. Adding DICOM ingestion to the main pipeline would require building a DICOM series stacker (sort by InstanceNumber, handle multi-series studies, etc.) — meaningful work, but orthogonal to demonstrating the ML pipeline. `dicom_demo.py` shows that capability in isolation, which is cleaner than mixing two I/O paths into one pipeline.

---

## Limitations & Next Steps

- **Volume calibration**: The model is trained at (64, 128, 128) voxels; volume is back-calculated using scaled voxel spacing. For clinical use, inference should run at original resolution or with a more careful resampling.
- **Multi-class masks collapsed to binary**: The GTS.ai dataset has 10 pathology classes. This project collapses them to lesion/background to keep training tractable in 2–4 days. A natural extension is multi-class segmentation with a softmax output head.
- **No test set evaluation**: The results above are validation-set metrics. A held-out test set would give a fairer estimate of generalisation.
- **Lateralization assumes standard axial orientation**: The hemisphere detection relies on the W axis being the L/R axis. Oblique acquisitions would require affine-aware orientation normalisation (e.g., `nibabel.as_closest_canonical`).

---

## Project Structure

```
medical-imaging-portfolio/
├── src/data/          loader.py · dataset.py · dicom_demo.py
├── src/preprocessing/ transforms.py
├── src/models/        unet.py
├── src/training/      trainer.py
├── src/visualization/ plots.py
├── api/               main.py · schemas.py · inference.py
├── notebooks/         01_data_exploration · 02_preprocessing · 03_model_training
├── scripts/           train.py
├── Dockerfile
└── docker-compose.yml
```
