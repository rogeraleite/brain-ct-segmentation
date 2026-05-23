import numpy as np
from scipy.ndimage import zoom

# Brain soft-tissue window.
# This range captures the clinically relevant structures:
#   CSF: 0–10 HU | White matter: 25–30 | Gray matter: 35–40 | Hemorrhage: 50–80
# Bone (>700 HU) and air (~-1000 HU) are excluded — they add noise without
# contributing to lesion detection.
BRAIN_HU_MIN: float = -5.0
BRAIN_HU_MAX: float = 75.0

# Training volume size: (D, H, W).
# 64×128×128 is a deliberate trade-off: small enough for CPU training in <4h,
# large enough to preserve spatial structure for segmentation.
TARGET_SHAPE: tuple[int, int, int] = (64, 128, 128)


def apply_brain_window(volume: np.ndarray) -> np.ndarray:
    """Clip volume to the brain soft-tissue HU window."""
    return np.clip(volume, BRAIN_HU_MIN, BRAIN_HU_MAX)


def normalize(volume: np.ndarray) -> np.ndarray:
    """Window then min-max normalize to [0, 1]."""
    volume = apply_brain_window(volume)
    return (volume - BRAIN_HU_MIN) / (BRAIN_HU_MAX - BRAIN_HU_MIN)


def resize_volume(volume: np.ndarray, target: tuple[int, int, int] = TARGET_SHAPE) -> np.ndarray:
    """Trilinear resize of a float volume to target shape."""
    factors = [t / s for t, s in zip(target, volume.shape)]
    return zoom(volume, factors, order=1).astype(np.float32)


def resize_mask(mask: np.ndarray, target: tuple[int, int, int] = TARGET_SHAPE) -> np.ndarray:
    """
    Nearest-neighbor resize of a binary mask.
    Must NOT use linear interpolation — that would create fractional values
    (e.g. 0.3 or 0.7) at mask boundaries, corrupting binary labels.
    """
    factors = [t / s for t, s in zip(target, mask.shape)]
    resized = zoom(mask.astype(np.float32), factors, order=0)
    return (resized > 0.5).astype(np.uint8)


def preprocess(
    volume: np.ndarray,
    mask: np.ndarray,
    target: tuple[int, int, int] = TARGET_SHAPE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline:
      1. NaN/inf guard (defensive — should already be clean from loader)
      2. Brain windowing + normalization → [0, 1]
      3. Resize volume (trilinear) and mask (nearest-neighbor) to target shape

    Returns:
        volume_norm: float32 (D, H, W), values in [0, 1]
        mask_resized: uint8  (D, H, W), values in {0, 1}
    """
    volume = np.nan_to_num(volume, nan=-1000.0, posinf=3000.0, neginf=-1000.0)
    mask = (mask > 0.5).astype(np.uint8)

    volume_norm = normalize(resize_volume(volume, target))
    mask_resized = resize_mask(mask, target)

    return volume_norm, mask_resized
