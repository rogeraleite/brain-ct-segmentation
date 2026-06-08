"""
Sliding-window inference for the patch-based model.

The model was trained on patches of (D_MAX, PATCH_HW, PATCH_HW) from
native-resolution volumes. Inference slides the same patch across H and W,
averages overlapping probability maps, and returns a mask at native resolution.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_erosion, binary_fill_holes

from src.data.patch_dataset import D_MAX, PATCH_HW, _pad_depth
from src.preprocessing.transforms import normalize, BRAIN_HU_MIN

# HU threshold for bone detection (Path 1 — raw CT only).
_SKULL_HU_THRESH: float = 300.0

# Distance from outer head contour to inner skull surface (Path 2 — pre-windowed data).
# Includes scalp (~5–8 mm) + bone (~7 mm). The user-facing excl_mm margin is
# applied on top of this, measured inward from the inner bone surface.
_SKULL_THICKNESS_MM: float = 15.0


def sliding_window_predict(
    volume: np.ndarray,
    model: nn.Module,
    device: torch.device,
    patch_hw: int = PATCH_HW,
    stride: int = 64,
    threshold: float = 0.5,
    spacing_hw_mm: float = 1.0,
    skull_excl_mm: float = 3.0,
) -> np.ndarray:
    """
    Run sliding-window inference on a native-resolution CT volume.

    The window covers the full depth (padded to D_MAX) and slides across
    H and W with the given stride. Overlapping regions are averaged.

    skull_excl_mm: after inference, predictions within this distance (mm) of
    dense bone (HU > 400) are zeroed out. The inner skull surface (dura mater,
    HU 50-100) is indistinguishable from hemorrhage by intensity; the exclusion
    margin removes these false positives without touching parenchymal lesions.
    Set to 0.0 to disable.

    Args:
        volume:        (D, H, W) float32 array in Hounsfield Units
        model:         trained Small3DUNet, expects (B, 1, D_MAX, patch_hw, patch_hw)
        device:        torch device
        patch_hw:      spatial patch size (must match training patch size)
        stride:        step between patches in H and W (smaller = smoother but slower)
        threshold:     binarisation threshold for the final mask
        spacing_hw_mm: in-plane voxel size in mm (used to convert skull_excl_mm to voxels)
        skull_excl_mm: exclusion margin around dense bone in mm (0 = disabled)

    Returns:
        mask: (D, H, W) uint8 binary mask at native resolution
    """
    D, H, W = volume.shape

    # Normalize (brain window + [0,1]) — no spatial resize
    volume_norm = normalize(volume).astype(np.float32)

    # Pad D to D_MAX; record how much was added at the top so we can trim later
    pad_total  = max(0, D_MAX - D)
    pad_before = pad_total // 2
    volume_padded, _ = _pad_depth(volume_norm, np.zeros_like(volume_norm, dtype=np.uint8), D_MAX)

    # Accumulate probabilities and overlap counts over H×W
    prob_sum = np.zeros((D_MAX, H, W), dtype=np.float32)
    count    = np.zeros((D_MAX, H, W), dtype=np.float32)

    h_starts = _patch_starts(H, patch_hw, stride)
    w_starts = _patch_starts(W, patch_hw, stride)

    model.eval()
    with torch.no_grad():
        for h0 in h_starts:
            for w0 in w_starts:
                patch = volume_padded[:, h0:h0 + patch_hw, w0:w0 + patch_hw]
                x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                prob = model(x)[0, 0].cpu().numpy()  # (D_MAX, patch_hw, patch_hw)
                prob_sum[:, h0:h0 + patch_hw, w0:w0 + patch_hw] += prob
                count[:,   h0:h0 + patch_hw, w0:w0 + patch_hw] += 1.0

    # Average and remove depth padding
    prob_avg = prob_sum / np.maximum(count, 1e-6)
    prob_orig = prob_avg[pad_before:pad_before + D]  # back to original D

    # Skull exclusion: zero out predictions near dense bone
    if skull_excl_mm > 0.0:
        excl = _skull_exclusion_mask(volume, _SKULL_HU_THRESH, skull_excl_mm, spacing_hw_mm)
        prob_orig[excl] = 0.0

    return (prob_orig >= threshold).astype(np.uint8)


def _skull_exclusion_mask(
    volume_hu: np.ndarray,
    bone_thresh: float,
    excl_mm: float,
    spacing_hw: float,
    skull_thickness_mm: float = _SKULL_THICKNESS_MM,
) -> np.ndarray:
    """
    Build a boolean mask covering the skull boundary zone (pre-windowed data).

    Per axial slice: fill the head outline, then erode inward by
    (skull_thickness_mm + excl_mm). The skull_thickness_mm traverses the bone
    from the outer head contour to the inner skull surface; excl_mm is the
    additional safety margin measured from that inner surface into the brain.

    Everything between the outer contour and (inner surface + excl_mm) is
    masked out — skull FPs and dura FPs are suppressed, parenchymal lesions
    are preserved.
    """
    D, H, W = volume_hu.shape

    total_mm = skull_thickness_mm + excl_mm
    radius_px = max(1, int(round(total_mm / spacing_hw)))
    d = 2 * radius_px + 1
    y_g, x_g = np.ogrid[:d, :d]
    circle2d = (y_g - radius_px) ** 2 + (x_g - radius_px) ** 2 <= radius_px ** 2

    skull_zone = np.zeros((D, H, W), dtype=bool)
    for i in range(D):
        slc = volume_hu[i]
        head_bin = slc > (BRAIN_HU_MIN + 1.0)
        if not head_bin.any():
            continue
        filled = binary_fill_holes(head_bin)
        eroded = binary_erosion(filled, structure=circle2d, border_value=0)
        skull_zone[i] = filled & ~eroded
    return skull_zone


def _patch_starts(dim: int, patch: int, stride: int) -> list[int]:
    """
    Compute patch start positions so patches cover [0, dim) completely.
    Always includes a final patch ending exactly at dim.
    """
    starts = list(range(0, dim - patch + 1, stride))
    if not starts or starts[-1] + patch < dim:
        starts.append(dim - patch)
    return starts
