import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter, label as nd_label, map_coordinates, rotate, zoom

# Brain soft-tissue window.
# This range captures the clinically relevant structures:
#   CSF: 0–10 HU | White matter: 25–30 | Gray matter: 35–40 | Hemorrhage: 50–80
# Bone (>700 HU) and air (~-1000 HU) are excluded — they add noise without
# contributing to lesion detection.
BRAIN_HU_MIN: float = -5.0
BRAIN_HU_MAX: float = 75.0

# Threshold for identifying cortical bone — used only for spatial skull masking,
# not for HU windowing (which uses the 100 HU cutoff in apply_brain_window).
_SKULL_BONE_HU: float = 300.0

# Training volume size: (D, H, W).
# 64×128×128 is a deliberate trade-off: small enough for CPU training in <4h,
# large enough to preserve spatial structure for segmentation.
TARGET_SHAPE: tuple[int, int, int] = (64, 128, 128)


def extract_intracranial_mask(volume_hu: np.ndarray) -> np.ndarray:
    """
    Return a bool (D, H, W) mask that is True inside the skull ring.

    Handles two data regimes automatically:
      - Raw CT (max HU > 100): cortical bone at _SKULL_BONE_HU (300 HU)
      - Brain-windowed data (max HU ≤ 100, e.g. JPEG-sourced): bone was
        clipped to BRAIN_HU_MAX (75 HU); use the top 15% of the HU range
        as the skull-ring proxy (≈ 63 HU threshold).

    Per axial slice: detect the skull ring, call binary_fill_holes to capture
    the enclosed intracranial region, and return that as the mask.  Slices
    with no detectable bone are left all-False.
    """
    vol_max = float(volume_hu.max())
    if vol_max > 100.0:
        skull_thresh = _SKULL_BONE_HU
    else:
        # Saturated pixels (bone clipped to BRAIN_HU_MAX) mark the skull ring.
        # Threshold at 85 % of the HU range to catch the ring while avoiding
        # most brain tissue (grey matter tops out at ~40 HU in this window).
        skull_thresh = BRAIN_HU_MIN + 0.85 * (BRAIN_HU_MAX - BRAIN_HU_MIN)

    D, H, W = volume_hu.shape
    mask = np.zeros((D, H, W), dtype=bool)
    for i in range(D):
        bone = volume_hu[i] > skull_thresh
        if bone.any():
            # Fill the skull ring to capture intracranial contents, then subtract
            # the bone pixels themselves so only soft tissue remains.
            mask[i] = binary_fill_holes(bone) & ~bone
    return mask


def extract_intracranial_mask_cc(volume_hu: np.ndarray) -> np.ndarray:
    """
    Return a bool (D, H, W) mask that is True inside the skull ring.

    Handles two failure modes of the naive HU-threshold approach:
      1. Hemorrhage exclusion: bright pixels (75 HU) include both skull bone and
         hemorrhage in brain-windowed data. CCs are used to isolate the skull ring
         (border-touching CCs preferred; largest CC as fallback).
      2. Ring gaps: the skull ring may have small intensity gaps that break
         binary_fill_holes. A 3-pixel dilation bridges these gaps before filling;
         the original (un-dilated) skull ring is subtracted so only soft tissue
         inside the ring is masked True.

    The intracranial region is defined as: filled(dilated_skull_ring) & ~skull_ring.
    """
    vol_max = float(volume_hu.max())
    skull_thresh = _SKULL_BONE_HU if vol_max > 100.0 else BRAIN_HU_MIN + 0.85 * (BRAIN_HU_MAX - BRAIN_HU_MIN)

    D, H, W = volume_hu.shape
    mask = np.zeros((D, H, W), dtype=bool)
    for i in range(D):
        bone = volume_hu[i] > skull_thresh
        if not bone.any():
            continue
        labeled, n_cc = nd_label(bone)
        if n_cc == 0:
            continue

        # Identify the skull ring: prefer CCs that touch the image border (the skull
        # always encircles the head and reaches the field-of-view boundary), fall back
        # to the largest CC when the head is fully contained within the image.
        border_labels = set(
            labeled[0, :].ravel().tolist() +
            labeled[-1, :].ravel().tolist() +
            labeled[:, 0].ravel().tolist() +
            labeled[:, -1].ravel().tolist()
        ) - {0}
        if not border_labels:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            border_labels = {int(np.argmax(counts))}
        skull_ring = np.isin(labeled, list(border_labels))

        # Dilate the skull ring slightly to close compression/threshold gaps before
        # filling, then subtract the ORIGINAL (un-dilated) ring so that interior
        # bright pixels (hemorrhage) are not removed from the mask.
        skull_dilated = binary_dilation(skull_ring, iterations=5)
        filled = binary_fill_holes(skull_dilated)
        candidate = filled & ~skull_ring

        # Fallback for highly fragmented skull rings (skull-base / open ring slices):
        # if the filled region is suspiciously small (<5% of image), derive the
        # intracranial region from brain parenchyma directly.  Brain tissue (0–63 HU)
        # forms one large connected blob; filling it captures hemorrhage holes inside.
        if candidate.sum() < 0.05 * H * W:
            tissue = (volume_hu[i] > (BRAIN_HU_MIN + 1.0)) & ~bone
            if tissue.any():
                lab_t, _ = nd_label(tissue)
                cnt_t = np.bincount(lab_t.ravel())
                cnt_t[0] = 0
                brain_cc = lab_t == int(np.argmax(cnt_t))
                candidate = binary_fill_holes(brain_cc)

        mask[i] = candidate
    return mask


def apply_brain_window(volume: np.ndarray) -> np.ndarray:
    """Clip to brain soft-tissue window, suppressing bone/calcification first."""
    volume = volume.copy()
    volume[volume > 100.0] = BRAIN_HU_MIN   # bone/calcification → background
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


def _rotate_pair(
    volume: np.ndarray, mask: np.ndarray, max_angle: float
) -> tuple[np.ndarray, np.ndarray]:
    angle = np.random.uniform(-max_angle, max_angle)
    vol_r = rotate(volume, angle, axes=(1, 2), reshape=False,
                   order=1, mode='constant', cval=0.0)
    mask_r = rotate(mask.astype(np.float32), angle, axes=(1, 2),
                    reshape=False, order=0, mode='constant', cval=0.0)
    return vol_r.astype(np.float32), (mask_r > 0.5).astype(np.uint8)


def _elastic_pair(
    volume: np.ndarray, mask: np.ndarray, alpha: float, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    shape = volume.shape
    fields = [gaussian_filter(np.random.standard_normal(shape), sigma) * alpha
              for _ in range(3)]
    grids = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
        indexing='ij',
    )
    indices = tuple(np.reshape(g + d, (-1,)) for g, d in zip(grids, fields))
    vol_d = map_coordinates(volume, indices, order=1,
                            mode='constant').reshape(shape)
    mask_d = map_coordinates(mask.astype(np.float32), indices, order=0,
                             mode='constant').reshape(shape)
    return vol_d.astype(np.float32), (mask_d > 0.5).astype(np.uint8)


def _jitter_volume(
    volume: np.ndarray, shift_range: float, noise_sigma: float
) -> np.ndarray:
    shift = float(np.random.uniform(-shift_range, shift_range))
    noise = np.random.normal(0.0, noise_sigma, volume.shape).astype(np.float32)
    return np.clip(volume + shift + noise, 0.0, 1.0)


def augment_pair(
    volume: np.ndarray,
    mask: np.ndarray,
    p_flip: float = 0.5,
    p_rotate: float = 0.5,
    max_angle: float = 10.0,
    p_elastic: float = 0.5,
    elastic_alpha: float = 150.0,
    elastic_sigma: float = 12.0,
    p_jitter: float = 0.7,
    jitter_shift: float = 0.0625,
    jitter_noise_sigma: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    On-the-fly augmentation applied per sample during training.

    volume: float32 (D, H, W) in [0, 1]
    mask:   uint8   (D, H, W) in {0, 1}

    Each transform fires independently at its own probability so combinations
    are possible — e.g. a flipped + rotated + jittered sample in one call.
    Geometric transforms (flip, rotate, elastic) are applied to both arrays;
    intensity jitter is volume-only.
    """
    if np.random.random() < p_flip:
        volume = volume[::-1].copy()
        mask = mask[::-1].copy()
    if np.random.random() < p_rotate:
        volume, mask = _rotate_pair(volume, mask, max_angle)
    if np.random.random() < p_elastic:
        volume, mask = _elastic_pair(volume, mask, elastic_alpha, elastic_sigma)
    if np.random.random() < p_jitter:
        volume = _jitter_volume(volume, jitter_shift, jitter_noise_sigma)
    return volume, mask


def preprocess(
    volume: np.ndarray,
    mask: np.ndarray,
    target: tuple[int, int, int] = TARGET_SHAPE,
    skull_strip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline:
      1. NaN/inf guard (defensive — should already be clean from loader)
      2. Optional skull stripping: zero out extracranial voxels before normalization
      3. Brain windowing + normalization → [0, 1]
      4. Resize volume (trilinear) and mask (nearest-neighbor) to target shape

    Returns:
        volume_norm: float32 (D, H, W), values in [0, 1]
        mask_resized: uint8  (D, H, W), values in {0, 1}
    """
    volume = np.nan_to_num(volume, nan=-1000.0, posinf=3000.0, neginf=-1000.0)
    mask = (mask > 0.5).astype(np.uint8)

    if skull_strip:
        intracranial = extract_intracranial_mask(volume)
        volume = np.where(intracranial, volume, BRAIN_HU_MIN)

    volume_norm = normalize(resize_volume(volume, target))
    mask_resized = resize_mask(mask, target)

    return volume_norm, mask_resized
