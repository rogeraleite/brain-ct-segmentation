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
from src.preprocessing.transforms import normalize, extract_intracranial_mask, BRAIN_HU_MIN, BRAIN_HU_MAX

# HU threshold for bone detection in raw-CT volumes.
_SKULL_HU_THRESH: float = 300.0

# For brain-windowed NIfTIs (max HU ≤ 100), bone is clipped to BRAIN_HU_MAX (75 HU).
# Use the same 85%-of-range proxy as extract_intracranial_mask to detect the bone ring.
_SKULL_HU_THRESH_BW: float = BRAIN_HU_MIN + 0.85 * (BRAIN_HU_MAX - BRAIN_HU_MIN)  # ≈ 63 HU


def sliding_window_predict(
    volume: np.ndarray,
    model: nn.Module,
    device: torch.device,
    patch_hw: int = PATCH_HW,
    stride: int = 64,
    threshold: float = 0.5,
    spacing_hw_mm: float = 1.0,
    skull_excl_mm: float = 0.0,
    skull_strip: bool = False,
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

    # Skull stripping must run on raw HU before normalize() loses bone signal.
    if skull_strip:
        intracranial = extract_intracranial_mask(volume)
        volume = np.where(intracranial, volume, BRAIN_HU_MIN)

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

    # Always zero out predictions outside the intracranial region (background + scalp)
    prob_orig[_extracranial_mask(volume, _SKULL_HU_THRESH)] = 0.0

    return (prob_orig >= threshold).astype(np.uint8)


def _skull_exclusion_mask(
    volume_hu: np.ndarray,
    bone_thresh: float,
    excl_mm: float,
    spacing_hw: float,
) -> np.ndarray:
    """
    Build a boolean mask covering the exclusion zone just inside the inner skull boundary.

    For raw-HU volumes (max > 100 HU): detects the bone ring at bone_thresh,
    fills it to find the intracranial space, and returns a band eroded inward by excl_mm.
    For brain-windowed volumes (max ≤ 100 HU): bone is saturated to BRAIN_HU_MAX,
    so a 85%-of-range proxy (~63 HU) detects the bone ring instead — same logic as
    extract_intracranial_mask — ensuring the exclusion zone sits at the inner skull
    surface rather than the outer scalp boundary.
    """
    D, H, W = volume_hu.shape

    radius_px = max(1, int(round(excl_mm / spacing_hw)))

    skull_thresh = _SKULL_HU_THRESH_BW if float(volume_hu.max()) <= 100.0 else bone_thresh

    # Build 3D intracranial volume via per-slice 2D fills — reliable for open skull rings
    # (3D fill fails at foramen magnum and other skull openings).
    intracranial_3d = np.zeros((D, H, W), dtype=bool)
    for i in range(D):
        bone_mask = volume_hu[i] > skull_thresh
        if not bone_mask.any():
            continue
        intracranial_3d[i] = binary_fill_holes(bone_mask) & ~bone_mask

    if not intracranial_3d.any():
        return np.zeros((D, H, W), dtype=bool)

    # 3D ball structuring element — erosion is Z-aware so the exclusion band
    # accounts for bone influence that leaks across slices via 3D convolutions.
    d = 2 * radius_px + 1
    zz, yy, xx = np.ogrid[:d, :d, :d]
    struct_3d = (zz - radius_px)**2 + (yy - radius_px)**2 + (xx - radius_px)**2 <= radius_px**2

    eroded_3d = binary_erosion(intracranial_3d, structure=struct_3d, border_value=0)
    return intracranial_3d & ~eroded_3d


def _extracranial_mask(volume_hu: np.ndarray, bone_thresh: float) -> np.ndarray:
    """
    True wherever we are outside the skull fill (background/scalp/air).

    For raw-HU volumes (max > 100 HU): detects the bone ring at bone_thresh
    and fills it to find the intracranial region.
    For brain-windowed volumes (max ≤ 100 HU, e.g. JPEG-derived NIfTIs):
    bone is saturated to BRAIN_HU_MAX — use the 85%-of-range proxy (~63 HU)
    to detect the bone ring, matching the logic in extract_intracranial_mask.
    """
    D, H, W = volume_hu.shape
    mask = np.ones((D, H, W), dtype=bool)
    brain_windowed = float(volume_hu.max()) <= 100.0
    skull_thresh = _SKULL_HU_THRESH_BW if brain_windowed else bone_thresh
    fallback_thresh = (BRAIN_HU_MIN + 1.0) if brain_windowed else bone_thresh
    for i in range(D):
        slc = volume_hu[i]
        tissue = slc > skull_thresh
        if not tissue.any():
            # No bone-proxy pixels in this slice (edge slices, pure brain).
            # Fall back to non-air tissue detection so predictions are not
            # incorrectly zeroed out for slices without a detectable skull ring.
            tissue = slc > fallback_thresh
            if not tissue.any():
                continue
        mask[i] = ~binary_fill_holes(tissue)
    return mask


def _patch_starts(dim: int, patch: int, stride: int) -> list[int]:
    """
    Compute patch start positions so patches cover [0, dim) completely.
    Always includes a final patch ending exactly at dim.
    """
    starts = list(range(0, dim - patch + 1, stride))
    if not starts or starts[-1] + patch < dim:
        starts.append(dim - patch)
    return starts
