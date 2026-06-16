"""
Spot-check SAM skull masks against brain-windowed CT and hemorrhage labels.

For each sampled case: shows 3 axial slices (top/mid/bottom of brain).
Each panel: original CT | masked CT (skull removed) | label overlay.

Usage:
    PYTHONPATH=. python scripts/visualize_skull_masks.py
    PYTHONPATH=. python scripts/visualize_skull_masks.py --cases 049 053 067 085
    PYTHONPATH=. python scripts/visualize_skull_masks.py --n-random 10 --out /tmp/skull_check.png
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loader import build_index, load_nifti
from src.preprocessing.transforms import BRAIN_HU_MIN, BRAIN_HU_MAX


def load_skull_mask(case_name: str, mask_dir: Path) -> np.ndarray | None:
    p = mask_dir / case_name
    if not p.exists():
        return None
    nib_img = nib.load(str(p))
    arr = np.transpose(nib_img.get_fdata(), (2, 0, 1))  # (H,W,D) → (D,H,W)
    return arr.astype(bool)


def to_display(slc_hu: np.ndarray) -> np.ndarray:
    clipped = np.clip(slc_hu, BRAIN_HU_MIN, BRAIN_HU_MAX)
    return (clipped - BRAIN_HU_MIN) / (BRAIN_HU_MAX - BRAIN_HU_MIN)


def pick_slices(mask_3d: np.ndarray, n: int = 3) -> list[int]:
    """Pick n evenly-spaced slices from the brain region (where mask is non-empty)."""
    occupied = [i for i in range(mask_3d.shape[0]) if mask_3d[i].any()]
    if not occupied:
        D = mask_3d.shape[0]
        return [D // 4, D // 2, 3 * D // 4]
    step = max(1, len(occupied) // (n + 1))
    return [occupied[min(step * (k + 1), len(occupied) - 1)] for k in range(n)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",   default="data/raw")
    parser.add_argument("--mask-dir",    default="data/skull_masks")
    parser.add_argument("--cases",       nargs="*", default=None,
                        help="Case IDs to visualize, e.g. 049 053 085")
    parser.add_argument("--n-random",    type=int,  default=10)
    parser.add_argument("--out",         default="/tmp/skull_mask_check.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_dir = Path(args.mask_dir)
    records = build_index(args.data_root)

    if args.cases:
        case_set = {c if c.endswith(".nii.gz") else f"{c}.nii.gz" for c in args.cases}
        selected = [r for r in records if Path(r["image"]).name in case_set]
    else:
        random.seed(42)
        selected = random.sample(records, min(args.n_random, len(records)))

    SLICES_PER_CASE = 3
    COLS = 3  # orig | masked | label
    n_cases = len(selected)
    fig, axes = plt.subplots(
        n_cases * SLICES_PER_CASE, COLS,
        figsize=(COLS * 3, n_cases * SLICES_PER_CASE * 3),
        squeeze=False,
    )

    for ci, rec in enumerate(selected):
        case_name = Path(rec["image"]).name
        volume, _  = load_nifti(rec["image"])   # (D, H, W)
        label, _   = load_nifti(rec["mask"])    # (D, H, W)
        label_b    = label > 0.5
        skull_mask = load_skull_mask(case_name, mask_dir)

        n_hem_total = int(label_b.sum())
        n_hem_preserved = int((label_b & skull_mask).sum()) if skull_mask is not None else 0
        pct = 100 * n_hem_preserved / n_hem_total if n_hem_total > 0 else 100.0

        ref_mask = skull_mask if skull_mask is not None else np.ones_like(label_b)
        slice_idxs = pick_slices(ref_mask, SLICES_PER_CASE)

        for si, sidx in enumerate(slice_idxs):
            row = ci * SLICES_PER_CASE + si
            slc   = volume[sidx]
            lbl   = label_b[sidx]
            smask = skull_mask[sidx] if skull_mask is not None else np.ones_like(lbl)

            orig_disp   = to_display(slc)
            masked_disp = to_display(np.where(smask, slc, BRAIN_HU_MIN))

            ax0, ax1, ax2 = axes[row]

            ax0.imshow(orig_disp, cmap="gray", vmin=0, vmax=1)
            ax0.axis("off")

            ax1.imshow(masked_disp, cmap="gray", vmin=0, vmax=1)
            ax1.axis("off")

            # label overlay: red = label, green = mask boundary
            rgb = np.stack([masked_disp] * 3, axis=-1)
            rgb[lbl]  = [1.0, 0.0, 0.0]  # hemorrhage → red
            edge = smask.astype(np.uint8)
            edge_px = (edge - np.minimum(edge, np.roll(edge, 1, 0))
                            - np.minimum(edge, np.roll(edge, -1, 0))
                            - np.minimum(edge, np.roll(edge, 1, 1))
                            - np.minimum(edge, np.roll(edge, -1, 1))).clip(0)
            rgb[edge_px.astype(bool)] = [0.0, 1.0, 0.0]  # mask edge → green
            ax2.imshow(rgb)
            ax2.axis("off")

            if si == 0:
                ax0.set_title(f"{case_name}\nz={sidx}", fontsize=7)
                ax1.set_title(f"masked\n(skull removed)", fontsize=7)
                ax2.set_title(f"label (red)\n{pct:.0f}% hem. preserved", fontsize=7)
            else:
                ax0.set_title(f"z={sidx}", fontsize=7)
                ax1.set_title("")
                ax2.set_title("")

    plt.tight_layout(pad=0.5)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=100, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
