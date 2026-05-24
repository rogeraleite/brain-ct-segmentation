"""
Inference helpers: load model, preprocess an uploaded NIfTI, run prediction,
compute derived metrics (volume + hemisphere lateralization).
"""

import base64
import io
import os
import tempfile

import nibabel as nib
import numpy as np
import torch

from src.models.unet import Small3DUNet
from src.preprocessing.transforms import preprocess, TARGET_SHAPE

MODEL_PATH = "models/best_model.pth"


# ── Model loading ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_path: str = MODEL_PATH) -> tuple[Small3DUNet, torch.device]:
    """Load model checkpoint. Called once at API startup via lifespan."""
    device = get_device()
    model = Small3DUNet()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


# ── Derived metrics ────────────────────────────────────────────────────────────

def compute_volume_ml(mask: np.ndarray, spacing_mm: np.ndarray) -> float:
    """
    Estimate lesion volume in millilitres.

    spacing_mm: (D_sp, H_sp, W_sp) in mm — from the NIfTI affine.
    Volume = num_lesion_voxels × voxel_volume_mm³, converted to mL (÷ 1000).
    """
    voxel_vol_mm3 = float(np.prod(spacing_mm))
    return float(mask.sum()) * voxel_vol_mm3 / 1000.0


def compute_lateralization(mask: np.ndarray) -> tuple[str, list[int]]:
    """
    Determine which brain hemisphere contains the lesion centroid.

    The W axis (last axis) is the left-right axis in axial brain CT.
    Radiological convention: image-left = patient-right.

    A 10% midline margin avoids over-committing on central lesions.

    Returns:
        hemisphere: "left" | "right" | "bilateral" | "none"
        centroid:   [D, H, W] index of the lesion centroid
    """
    if mask.sum() == 0:
        return "none", [0, 0, 0]

    coords = np.argwhere(mask > 0)
    centroid = coords.mean(axis=0).astype(int).tolist()  # [D, H, W]

    w_size = mask.shape[2]
    midline = w_size // 2
    margin = w_size * 0.10  # 10% tolerance zone around midline

    w_centroid = centroid[2]
    if w_centroid < midline - margin:
        hemisphere = "right"   # image-left = patient-right (radiological)
    elif w_centroid > midline + margin:
        hemisphere = "left"
    else:
        hemisphere = "bilateral"

    return hemisphere, centroid


# ── Full prediction pipeline ───────────────────────────────────────────────────

def predict_from_bytes(
    file_bytes: bytes,
    model: Small3DUNet,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    End-to-end inference from raw file bytes to structured result.

    Steps:
      1. Parse NIfTI from bytes
      2. Preprocess (brain window + normalize + resize)
      3. Model forward pass
      4. Threshold → binary mask
      5. Compute volume + hemisphere
      6. Base64-encode mask for API transport

    Returns dict matching SegmentationResponse schema.
    """
    # 1. Load NIfTI — nibabel's FileHolder doesn't handle gzip in-memory,
    #    so we write to a temp file (works for both .nii and .nii.gz)
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

    # 2. Preprocess
    volume_pp, _ = preprocess(volume, np.zeros_like(volume, dtype=np.uint8))

    # 3. Forward pass
    x = torch.from_numpy(volume_pp).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
    with torch.no_grad():
        prob_map = model(x)[0, 0].cpu().numpy()  # (D, H, W)

    # 4. Threshold
    mask = (prob_map >= threshold).astype(np.uint8)

    # 5. Derived metrics
    # Volume uses the resized voxel spacing (spacing scaled to TARGET_SHAPE)
    scale = np.array(TARGET_SHAPE, dtype=np.float32) / np.array(volume.shape, dtype=np.float32)
    resized_spacing = spacing_mm / scale  # mm per resized voxel
    lesion_vol = compute_volume_ml(mask, resized_spacing)
    hemisphere, centroid = compute_lateralization(mask)

    # 6. Encode mask
    mask_bytes = mask.astype(np.uint8).tobytes()
    mask_b64 = base64.b64encode(mask_bytes).decode("utf-8")

    return {
        "lesion_volume_ml": round(lesion_vol, 3),
        "hemisphere": hemisphere,
        "centroid_voxel": centroid,
        "lesion_voxel_count": int(mask.sum()),
        "mask_shape": list(mask.shape),
        "mask_base64": mask_b64,
    }
