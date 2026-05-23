import torch
from torch.utils.data import Dataset

from src.data.loader import load_nifti
from src.preprocessing.transforms import preprocess, TARGET_SHAPE


class BrainCTDataset(Dataset):
    """
    PyTorch Dataset for brain CT segmentation.

    Each item is a (volume_tensor, mask_tensor) pair:
      volume_tensor: float32 (1, D, H, W), values in [0, 1]
      mask_tensor:   float32 (1, D, H, W), values in {0.0, 1.0}

    Augmentation (when augment=True):
      - Random depth-axis flip (p=0.5) — flip along the D axis
        Both volume and mask are flipped together to keep them aligned.
        We only flip along depth (not height/width) to avoid laterality confusion
        since hemisphere detection depends on the L/R axis.
    """

    def __init__(
        self,
        records: list[dict],
        target_shape: tuple[int, int, int] = TARGET_SHAPE,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.target_shape = target_shape
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.records[idx]

        volume, _ = load_nifti(r["image"])
        mask, _ = load_nifti(r["mask"])

        volume, mask = preprocess(volume, mask, self.target_shape)

        x = torch.from_numpy(volume).unsqueeze(0)          # (1, D, H, W)
        y = torch.from_numpy(mask).float().unsqueeze(0)    # (1, D, H, W)

        assert torch.isfinite(x).all(), f"Non-finite values in volume: {r['image']}"

        if self.augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])

        return x, y
