"""
Visualize the skull exclusion ring (Path 2 / pre-windowed) on a few axial slices.

Usage:
    python scripts/check_skull_exclusion.py [nii_path] [excl_mm]

Defaults: data/raw/images/049.nii.gz, excl_mm=20
"""

import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.sliding_window import _skull_exclusion_mask, _SKULL_HU_THRESH

NII_PATH   = sys.argv[1] if len(sys.argv) > 1 else "data/raw/images/049.nii.gz"
EXCL_MM    = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
OUT_PATH   = "scripts/skull_excl_check.png"
SPACING_MM = 1.0   # typical in-plane spacing for this dataset

img    = nib.load(NII_PATH)
volume = img.get_fdata().astype(np.float32)
volume = np.transpose(volume, (2, 0, 1))   # (H,W,D) → (D,H,W)

D = volume.shape[0]
print(f"Volume shape: {volume.shape}  HU range: [{volume.min():.0f}, {volume.max():.0f}]")
print(f"Voxels > {_SKULL_HU_THRESH} HU: {(volume > _SKULL_HU_THRESH).sum()}  (0 = pre-windowed, Path 2 will run)")

excl = _skull_exclusion_mask(volume, _SKULL_HU_THRESH, EXCL_MM, SPACING_MM)
print(f"Exclusion mask: {excl.sum()} voxels masked  ({100*excl.mean():.1f}% of volume)")

# Pick 6 evenly-spaced slices from the middle half of the volume
slice_ids = np.linspace(D // 4, 3 * D // 4, 6, dtype=int)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    f"{Path(NII_PATH).name} — skull exclusion ring  (excl_mm={EXCL_MM})\n"
    f"orange = excluded zone  |  green = kept brain region",
    fontsize=12,
)

for ax, s in zip(axes.flat, slice_ids):
    slc  = volume[s]
    ring = excl[s]

    # Normalise slice to [0,1] for display
    lo, hi = slc.min(), slc.max()
    disp = (slc - lo) / (hi - lo + 1e-6)

    ax.imshow(disp, cmap="gray", origin="upper")
    ax.contour(ring.astype(float), levels=[0.5], colors=["orange"], linewidths=1.5)

    # Shade the excluded zone with a semi-transparent overlay
    overlay = np.zeros((*ring.shape, 4), dtype=np.float32)
    overlay[ring, 0] = 1.0   # R
    overlay[ring, 3] = 0.35  # alpha
    ax.imshow(overlay, origin="upper")

    ax.set_title(f"slice {s}", fontsize=9)
    ax.axis("off")

orange_patch = mpatches.Patch(color="orange", alpha=0.5, label="excluded (skull zone)")
fig.legend(handles=[orange_patch], loc="lower center", ncol=1, fontsize=10)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
print(f"\nSaved → {OUT_PATH}")
