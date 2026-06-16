import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.loader import load_nifti, load_skull_mask
from src.preprocessing.transforms import (
    BRAIN_HU_MIN,
    augment_pair,
    extract_intracranial_mask_cc,
    normalize,
)

# All volumes in this dataset share H=W=650.
# D varies from 19 to 40 — we pad shorter volumes to D_MAX so the model
# receives a fixed-size input without downsampling H or W.
D_MAX: int = 40
PATCH_HW: int = 128


class PatchBrainCTDataset(Dataset):
    """
    Patch-based dataset for sliding-window model training.

    Unlike BrainCTDataset, this class does NOT resize the volume.
    It preserves native H×W resolution (650×650) and only:
      1. Applies brain windowing + normalization
      2. Pads D to D_MAX (zero-pad, symmetric)
      3. Crops a random PATCH_HW×PATCH_HW patch in the H×W plane

    Each item: (float32 [1, D_MAX, PATCH_HW, PATCH_HW],
                float32 [1, D_MAX, PATCH_HW, PATCH_HW])
    """

    def __init__(
        self,
        records: list[dict],
        patch_hw: int = PATCH_HW,
        d_max: int = D_MAX,
        augment: bool = False,
        skull_strip: bool = False,
        no_elastic: bool = False,
        skull_mask_dir: str | None = None,
    ) -> None:
        self.records = records
        self.patch_hw = patch_hw
        self.d_max = d_max
        self.augment = augment
        self.skull_strip = skull_strip
        self.no_elastic = no_elastic
        self.skull_mask_dir = Path(skull_mask_dir) if skull_mask_dir else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.records[idx]

        volume, _ = load_nifti(r["image"])   # (D, H, W) native resolution
        mask, _   = load_nifti(r["mask"])

        mask = (mask > 0.5).astype(np.uint8)

        # Skull stripping must run on raw HU before normalization loses bone signal.
        if self.skull_mask_dir is not None:
            mask_path = self.skull_mask_dir / Path(r["image"]).name
            if mask_path.exists():
                intracranial = load_skull_mask(str(mask_path))
                volume = np.where(intracranial, volume, BRAIN_HU_MIN)
        elif self.skull_strip:
            intracranial = extract_intracranial_mask_cc(volume)
            volume = np.where(intracranial, volume, BRAIN_HU_MIN)

        # Brain window + normalize to [0, 1] — no spatial resize
        volume = normalize(volume).astype(np.float32)

        # Pad D to d_max (symmetric, zero-fill)
        volume, mask = _pad_depth(volume, mask, self.d_max)

        # Lesion-centered sampling: 60% of patches are anchored on a lesion voxel.
        # Pure random crops from 650×650 miss the lesion >99% of the time, causing
        # the model to converge to the trivial all-zeros solution.
        _, H, W = volume.shape
        lesion_coords = np.argwhere(mask > 0)  # (N, 3): [d, h, w]
        use_lesion_center = (len(lesion_coords) > 0) and (random.random() < 0.7)
        if use_lesion_center:
            center = lesion_coords[random.randint(0, len(lesion_coords) - 1)]
            h0 = int(np.clip(center[1] - self.patch_hw // 2, 0, H - self.patch_hw))
            w0 = int(np.clip(center[2] - self.patch_hw // 2, 0, W - self.patch_hw))
        else:
            h0 = random.randint(0, H - self.patch_hw)
            w0 = random.randint(0, W - self.patch_hw)
        volume = volume[:, h0:h0 + self.patch_hw, w0:w0 + self.patch_hw]
        mask   = mask[:,   h0:h0 + self.patch_hw, w0:w0 + self.patch_hw]

        if self.augment:
            volume, mask = augment_pair(volume, mask, p_elastic=0.0 if self.no_elastic else 0.5)

        x = torch.from_numpy(volume).unsqueeze(0)        # (1, D_MAX, PH, PW)
        y = torch.from_numpy(mask).float().unsqueeze(0)  # (1, D_MAX, PH, PW)

        assert torch.isfinite(x).all(), f"Non-finite values in volume: {r['image']}"

        return x, y


def _pad_depth(
    volume: np.ndarray,
    mask: np.ndarray,
    d_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrically zero-pad the D axis to d_max. No-op if D >= d_max."""
    D = volume.shape[0]
    if D >= d_max:
        return volume[:d_max], mask[:d_max]
    pad_total  = d_max - D
    pad_before = pad_total // 2
    pad_after  = pad_total - pad_before
    volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)))
    mask   = np.pad(mask,   ((pad_before, pad_after), (0, 0), (0, 0)))
    return volume, mask
