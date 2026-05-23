"""
Converts the vbookshelf/computed-tomography-ct-images dataset
(2D JPG brain-window slices per patient) into 3D NIfTI volumes.

What this script does for each patient:
  1. Reads brain-window slices sorted by slice number (1.jpg, 2.jpg, ...)
  2. Reverse-maps pixel values [0,255] back to approximate Hounsfield Units
     (the JPGs were exported using a brain window: [-5, 75] HU → [0, 255])
  3. Stacks slices into a 3D volume (D, H, W)
  4. Builds a 3D mask: slices with a _HGE_Seg.jpg file → mask=1, rest → mask=0
  5. Saves both as .nii.gz in data/raw/images/ and data/raw/masks/

Usage:
    python scripts/convert_jpg_to_nifti.py \
        --dataset-dir /tmp/kaggle_test/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0 \
        --out-dir data/raw
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

# Brain window used during original CT export (standard brain window)
JPG_HU_MIN = -5.0   # black pixel = -5 HU
JPG_HU_MAX = 75.0   # white pixel = 75 HU


def jpg_to_hu(pixel_array: np.ndarray) -> np.ndarray:
    """
    Reverse-map JPG pixel values [0, 255] → Hounsfield Units [-5, 75].

    The original dataset was exported from DICOM using a brain window of
    [-5, 75] HU. This restores the approximate HU values so our standard
    preprocessing pipeline (which operates in HU space) works unchanged.
    """
    return pixel_array.astype(np.float32) * (JPG_HU_MAX - JPG_HU_MIN) / 255.0 + JPG_HU_MIN


def read_sorted_slices(brain_dir: Path) -> list[int]:
    """Return sorted slice numbers (ignoring _HGE_Seg files)."""
    numbers = []
    for f in brain_dir.glob("*.jpg"):
        if "_HGE_Seg" not in f.name:
            try:
                numbers.append(int(f.stem))
            except ValueError:
                pass
    return sorted(numbers)


def build_patient_volume(brain_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack 2D slices into (D, H, W) volume and matching binary mask.

    Returns:
        volume: float32 (D, H, W) in approximate HU
        mask:   uint8  (D, H, W) with values in {0, 1}
    """
    slice_numbers = read_sorted_slices(brain_dir)
    if not slice_numbers:
        raise ValueError(f"No slices found in {brain_dir}")

    vol_slices, mask_slices = [], []

    for n in slice_numbers:
        # CT slice
        img_path = brain_dir / f"{n}.jpg"
        img = np.array(Image.open(img_path).convert("L"))  # grayscale
        vol_slices.append(jpg_to_hu(img))

        # Mask slice (only exists for hemorrhage slices)
        mask_path = brain_dir / f"{n}_HGE_Seg.jpg"
        if mask_path.exists():
            mask_img = np.array(Image.open(mask_path).convert("L"))
            mask_slices.append((mask_img > 128).astype(np.uint8))
        else:
            mask_slices.append(np.zeros_like(img, dtype=np.uint8))

    volume = np.stack(vol_slices,  axis=0)  # (D, H, W)
    mask   = np.stack(mask_slices, axis=0)  # (D, H, W)
    return volume, mask


def make_affine(n_slices: int, slice_thickness_mm: float = 5.0) -> np.ndarray:
    """
    Build a simple NIfTI affine matrix.
    In-plane spacing is unknown (JPGs have no DICOM metadata), so we use 1mm.
    Slice thickness is 5mm as documented in the dataset README.
    """
    return np.diag([1.0, 1.0, slice_thickness_mm, 1.0])


def save_nifti(array: np.ndarray, affine: np.ndarray, out_path: Path) -> None:
    # nibabel convention: (H, W, D), so we transpose from (D, H, W)
    arr_nib = np.transpose(array, (1, 2, 0))
    img = nib.Nifti1Image(arr_nib, affine)
    nib.save(img, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="/tmp/kaggle_test/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0",
        help="Root folder of the downloaded Kaggle dataset",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw",
        help="Output root: images/ and masks/ subdirs will be created here",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    patients_dir = dataset_dir / "Patients_CT"
    out_images = Path(args.out_dir) / "images"
    out_masks  = Path(args.out_dir) / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    patient_ids = sorted(p.name for p in patients_dir.iterdir() if p.is_dir())
    print(f"Found {len(patient_ids)} patients")

    n_with_lesion = 0
    errors = []

    for pid in tqdm(patient_ids, desc="Converting"):
        brain_dir = patients_dir / pid / "brain"
        if not brain_dir.exists():
            errors.append(f"{pid}: no brain/ folder")
            continue

        try:
            volume, mask = build_patient_volume(brain_dir)
        except Exception as e:
            errors.append(f"{pid}: {e}")
            continue

        affine = make_affine(volume.shape[0])
        save_nifti(volume, affine, out_images / f"{pid}.nii.gz")
        save_nifti(mask,   affine, out_masks  / f"{pid}.nii.gz")

        if mask.sum() > 0:
            n_with_lesion += 1

    print(f"\nDone.")
    print(f"  Converted : {len(patient_ids) - len(errors)} patients")
    print(f"  With lesion: {n_with_lesion}")
    print(f"  No lesion  : {len(patient_ids) - len(errors) - n_with_lesion}")
    print(f"  Errors     : {len(errors)}")
    print(f"  Output     : {Path(args.out_dir).resolve()}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
