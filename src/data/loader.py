import nibabel as nib
import numpy as np
from pathlib import Path


def load_skull_mask(path: str) -> np.ndarray:
    """Load a SAM-generated skull mask NIfTI. Returns bool (D, H, W)."""
    img = nib.load(path)
    arr = img.get_fdata().astype(np.float32)
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))  # nibabel (H,W,D) → (D,H,W)
    return arr > 0.5


def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI file.

    Returns:
        volume:  float32 array, shape (D, H, W) in Hounsfield Units
        spacing: float64 array, voxel size in mm [D_sp, H_sp, W_sp]
    """
    img = nib.load(path)
    volume = img.get_fdata().astype(np.float32)

    # nibabel loads (H, W, D) — transpose to (D, H, W) for PyTorch convention
    if volume.ndim == 3:
        volume = np.transpose(volume, (2, 0, 1))

    # Voxel spacing from the affine diagonal (abs handles flipped axes)
    spacing = np.abs(np.diag(img.affine)[:3])[::-1].copy()  # (D_sp, H_sp, W_sp)

    volume = np.nan_to_num(volume, nan=-1000.0, posinf=3000.0, neginf=-1000.0)
    return volume, spacing


def build_index(data_root: str) -> list[dict]:
    """
    Pair up image and mask NIfTI files by sorted filename.

    Expects:
        <data_root>/images/*.nii or *.nii.gz
        <data_root>/masks/*.nii  or *.nii.gz

    Returns list of {"image": path_str, "mask": path_str} dicts.
    """
    root = Path(data_root)
    images = sorted((root / "images").glob("*.nii*"))
    masks = sorted((root / "masks").glob("*.nii*"))

    if len(images) == 0:
        raise FileNotFoundError(f"No NIfTI files found in {root / 'images'}")
    if len(images) != len(masks):
        raise ValueError(
            f"Image/mask count mismatch: {len(images)} images vs {len(masks)} masks"
        )

    return [{"image": str(i), "mask": str(m)} for i, m in zip(images, masks)]


def train_val_split(
    records: list[dict], val_fraction: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Deterministic shuffle + split. Returns (train_records, val_records)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(records)).tolist()
    n_val = max(1, int(len(records) * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return [records[i] for i in train_idx], [records[i] for i in val_idx]
