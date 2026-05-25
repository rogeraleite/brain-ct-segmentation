"""
Sliding-window inference for the patch-based model.

The model was trained on patches of (D_MAX, PATCH_HW, PATCH_HW) from
native-resolution volumes. Inference slides the same patch across H and W,
averages overlapping probability maps, and returns a mask at native resolution.
"""

import numpy as np
import torch
import torch.nn as nn

from src.data.patch_dataset import D_MAX, PATCH_HW, _pad_depth
from src.preprocessing.transforms import normalize


def sliding_window_predict(
    volume: np.ndarray,
    model: nn.Module,
    device: torch.device,
    patch_hw: int = PATCH_HW,
    stride: int = 64,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run sliding-window inference on a native-resolution CT volume.

    The window covers the full depth (padded to D_MAX) and slides across
    H and W with the given stride. Overlapping regions are averaged.

    Args:
        volume:   (D, H, W) float32 array in Hounsfield Units
        model:    trained Small3DUNet, expects (B, 1, D_MAX, patch_hw, patch_hw)
        device:   torch device
        patch_hw: spatial patch size (must match training patch size)
        stride:   step between patches in H and W (smaller = smoother but slower)
        threshold: binarisation threshold for the final mask

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

    return (prob_orig >= threshold).astype(np.uint8)


def _patch_starts(dim: int, patch: int, stride: int) -> list[int]:
    """
    Compute patch start positions so patches cover [0, dim) completely.
    Always includes a final patch ending exactly at dim.
    """
    starts = list(range(0, dim - patch + 1, stride))
    if not starts or starts[-1] + patch < dim:
        starts.append(dim - patch)
    return starts
