"""
Slice-level dataset for the v15 detect-then-segment Stage-1 detector.

Each axial slice becomes one training sample labelled 1 if its lesion mask
holds any positive voxel, 0 otherwise — weak-but-free supervision derived from
the existing segmentation masks. Slices are windowed/normalized exactly like the
segmenter's input (transforms.normalize) so the two stages share a distribution.

Slices are pre-resized to out_hw×out_hw and cached in memory at construction
(~540 MB for all 82 cases at 224²), so training does not reload NIfTI volumes
per slice and never rebuilds the full-resolution volume that OOM-kills here.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from src.data.loader import load_nifti
from src.preprocessing.transforms import normalize


class SliceHemorrhageDataset(Dataset):
    def __init__(self, records: list[dict], out_hw: int = 224, augment: bool = False) -> None:
        self.out_hw = out_hw
        self.augment = augment
        slices: list[np.ndarray] = []
        labels: list[int] = []
        case_ids: list[str] = []

        for r in records:
            vol, _ = load_nifti(r["image"])
            vol = normalize(vol).astype(np.float32)          # (D, H, W) in [0, 1]
            mask, _ = load_nifti(r["mask"])
            mask = mask > 0.5
            cid = Path(r["image"]).name
            _, H, W = vol.shape
            fy, fx = out_hw / H, out_hw / W
            for d in range(vol.shape[0]):
                sl = zoom(vol[d], (fy, fx), order=1).astype(np.float32)
                slices.append(sl)
                labels.append(int(mask[d].any()))
                case_ids.append(cid)

        self.slices = np.stack(slices)                       # (N, out_hw, out_hw)
        self.labels = np.asarray(labels, dtype=np.float32)   # (N,)
        self.case_ids = case_ids                             # len N

    def __len__(self) -> int:
        return len(self.labels)

    def pos_weight(self) -> float:
        """n_neg / n_pos over this split — for BCEWithLogitsLoss class balancing."""
        n_pos = float(self.labels.sum())
        n_neg = float(len(self.labels) - n_pos)
        return n_neg / max(n_pos, 1.0)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        sl = self.slices[i]
        # Lateral (left-right) flip — H is the L-R axis in these RAS volumes.
        if self.augment and np.random.random() < 0.5:
            sl = sl[:, ::-1].copy()
        x = torch.from_numpy(sl).unsqueeze(0)                # (1, out_hw, out_hw)
        return x, torch.tensor(self.labels[i])
