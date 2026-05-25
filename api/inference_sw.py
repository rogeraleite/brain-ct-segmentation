"""
Inference helpers for the sliding-window model (best_model_slidingWindow.pth).

Mirrors the structure of inference.py but uses sliding_window_predict()
instead of global resize, preserving native H×W resolution.
"""

import base64
import os
import tempfile

import nibabel as nib
import numpy as np
import torch

from src.models.unet import Small3DUNet
from src.inference.sliding_window import sliding_window_predict
from api.inference import get_device, compute_volume_ml, compute_lateralization

MODEL_PATH_SW = "models/best_model_slidingWindow.pth"


def load_model_sw(model_path: str = MODEL_PATH_SW) -> tuple[Small3DUNet, torch.device]:
    """Load sliding-window model checkpoint. Called once at API startup."""
    device = get_device()
    model = Small3DUNet()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def predict_from_bytes_sw(
    file_bytes: bytes,
    model: Small3DUNet,
    device: torch.device,
    stride: int = 64,
    threshold: float = 0.5,
) -> dict:
    """
    End-to-end sliding-window inference from raw NIfTI bytes.

    Differences vs predict_from_bytes (resize model):
      - No global resize — H×W stays at native resolution (650×650)
      - sliding_window_predict() slides (D_MAX, 128, 128) patches across H×W
      - Volume metric uses original voxel spacing directly (no rescaling needed)
    """
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

    # Sliding window — returns mask at native resolution (D, H, W)
    mask = sliding_window_predict(volume, model, device, stride=stride, threshold=threshold)

    # Volume: spacing is for the original resolution — no rescaling needed
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
    }
