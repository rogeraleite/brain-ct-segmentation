"""
Inference for the v15 detect-then-segment cascade.

Stage 1: an ENSEMBLE of the 5 fold detectors scores each axial slice; the mean
sigmoid over folds is the per-slice probability, and the case probability is the
max over slices. Ensembling the folds is the robust choice for an unseen upload
(no single fold trained on it).

Stage 2: if the case probability clears the gate, the v14 sliding-window
segmenter runs; otherwise the mask is all-zeros — the whole point of the gate is
to stop the segmenter from inventing false positives on lesion-free scans.
"""

import base64
import os
import tempfile

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom

from api.inference import compute_lateralization, compute_volume_ml, get_device
from src.inference.sliding_window import sliding_window_predict
from src.models.detector import build_detector
from src.preprocessing.transforms import normalize

DETECTOR_PREFIX = "models/detector_v15"
SEGMENTER_PATH = "models/best_model_slidingWindow_v14_fold0_f1.pth"
DETECTOR_OUT_HW = 224


def load_detectors(prefix: str = DETECTOR_PREFIX, n_folds: int = 5) -> tuple[list, torch.device]:
    """Load the fold detectors present on disk as an ensemble. Raises
    FileNotFoundError if none are found so the API can report 503."""
    device = get_device()
    detectors = []
    for k in range(n_folds):
        path = f"{prefix}_fold{k}.pth"
        if not os.path.exists(path):
            continue
        model = build_detector(pretrained=False).to(device)
        model.load_state_dict(torch.load(path, map_location=device)["model_state_dict"])
        model.eval()
        detectors.append(model)
    if not detectors:
        raise FileNotFoundError(f"No detector checkpoints found at {prefix}_fold*.pth")
    return detectors, device


@torch.no_grad()
def _slice_probabilities(volume: np.ndarray, detectors: list, device: torch.device,
                         out_hw: int = DETECTOR_OUT_HW) -> np.ndarray:
    """Ensemble mean P(hemorrhage) per axial slice. volume is (D, H, W) in HU."""
    vol = normalize(volume).astype(np.float32)          # window + [0, 1]
    D, H, W = vol.shape
    fy, fx = out_hw / H, out_hw / W
    batch = np.stack([zoom(vol[d], (fy, fx), order=1) for d in range(D)])  # (D, out, out)
    x = torch.from_numpy(batch).unsqueeze(1).to(device)                    # (D, 1, out, out)
    probs = torch.zeros(D, device=device)
    for det in detectors:
        probs += torch.sigmoid(det(x)).squeeze(1)
    probs /= len(detectors)
    return probs.cpu().numpy()


def predict_cascade_from_bytes(
    file_bytes: bytes,
    detectors: list,
    det_device: torch.device,
    seg_model,
    seg_device: torch.device,
    case_threshold: float = 0.5,
    threshold: float = 0.3,
    stride: int = 64,
    model_version: str = "sw-v15",
) -> dict:
    suffix = ".nii.gz" if file_bytes[:2] == b"\x1f\x8b" else ".nii"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, file_bytes)
        os.close(tmp_fd)
        img = nib.load(tmp_path)
        volume = img.get_fdata().astype(np.float32)
        spacing_mm = np.abs(np.diag(img.affine)[:3])[::-1].copy()
    finally:
        os.unlink(tmp_path)

    if volume.ndim == 3:
        volume = np.transpose(volume, (2, 0, 1))  # (H,W,D) → (D,H,W)

    # ── Stage 1: detector ──────────────────────────────────────────────────
    slice_probs = _slice_probabilities(volume, detectors, det_device)
    case_prob = float(slice_probs.max()) if slice_probs.size else 0.0
    detected = case_prob >= case_threshold

    # ── Stage 2: segment only if the gate opens ────────────────────────────
    if detected:
        spacing_hw_mm = float(min(spacing_mm[1], spacing_mm[2]))
        mask = sliding_window_predict(
            volume, seg_model, seg_device,
            stride=stride, threshold=threshold,
            spacing_hw_mm=spacing_hw_mm, skull_strip=True, skull_excl_mm=0.0,
        )
    else:
        mask = np.zeros(volume.shape, dtype=np.uint8)

    lesion_vol = compute_volume_ml(mask, spacing_mm)
    hemisphere, centroid = compute_lateralization(mask)
    mask_b64 = base64.b64encode(mask.astype(np.uint8).tobytes()).decode("utf-8")

    return {
        "lesion_volume_ml": round(lesion_vol, 3),
        "hemisphere": hemisphere,
        "centroid_voxel": centroid,
        "lesion_voxel_count": int(mask.sum()),
        "mask_shape": list(mask.shape),
        "mask_base64": mask_b64,
        "model_version": model_version,
        "hemorrhage_detected": bool(detected),
        "case_probability": round(case_prob, 4),
        "case_threshold": float(case_threshold),
        "slice_probabilities": [round(float(p), 4) for p in slice_probs],
    }
