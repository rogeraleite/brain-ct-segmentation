"""
Visualiza opções de remoção do anel residual após CC skull strip.

Estratégia: erosão morfológica da CC mask por K pixels remove a camada
externa (anel ósseo/dura) sem tocar o interior do cérebro.

Colunas:
  Col 1 - CC skull strip (estado atual, com anel)
  Col 2 - Erosão K=3 px (anel em laranja = removido)
  Col 3 - Erosão K=5 px (anel em laranja = removido)
  Col 4 - Ground truth

Usage:
    PYTHONPATH=. .venv/bin/python scripts/visualize_ring_removal.py \
        --case 051 --slices 10 15 20
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion

from src.preprocessing.transforms import (
    BRAIN_HU_MIN, BRAIN_HU_MAX,
    extract_intracranial_mask_cc,
    normalize,
)

DATA_ROOT  = Path("data/raw")
OUTPUT_DIR = Path("reports/figures")


def load_volume(bare_id: str):
    img = nib.load(DATA_ROOT / "images" / f"{bare_id}.nii.gz")
    msk = nib.load(DATA_ROOT / "masks"  / f"{bare_id}.nii.gz")
    vol = np.transpose(img.get_fdata().astype(np.float32), (2, 0, 1))
    lbl = np.transpose(msk.get_fdata().astype(np.float32), (2, 0, 1)) > 0.5
    return vol, lbl


def display(vol: np.ndarray) -> np.ndarray:
    return np.clip((vol - BRAIN_HU_MIN) / (BRAIN_HU_MAX - BRAIN_HU_MIN), 0, 1)


def erode_mask(cc_mask: np.ndarray, k: int) -> np.ndarray:
    """Erode a máscara 2D por slice por K pixels."""
    D = cc_mask.shape[0]
    eroded = np.zeros_like(cc_mask)
    struct = np.ones((2*k+1, 2*k+1), dtype=bool)  # quadrado K px
    for i in range(D):
        if cc_mask[i].any():
            eroded[i] = binary_erosion(cc_mask[i], structure=struct)
    return eroded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="051")
    parser.add_argument("--slices", nargs="+", type=int, default=None)
    args = parser.parse_args()

    bare_id = args.case.replace(".nii.gz", "")
    print(f"Carregando {bare_id}...")
    vol, lbl = load_volume(bare_id)
    D, H, W  = vol.shape

    print("Computando CC skull strip...")
    cc_mask  = extract_intracranial_mask_cc(vol)
    vol_cc   = np.where(cc_mask, vol, BRAIN_HU_MIN)

    print("Computando erosões...")
    eroded3  = erode_mask(cc_mask, k=3)
    eroded5  = erode_mask(cc_mask, k=5)

    vol_e3   = np.where(eroded3, vol, BRAIN_HU_MIN)
    vol_e5   = np.where(eroded5, vol, BRAIN_HU_MIN)

    # Seleção de slices
    if args.slices:
        slices = [s for s in args.slices if 0 <= s < D]
    else:
        lesion_slices = np.where(lbl.any(axis=(1, 2)))[0]
        if len(lesion_slices) == 0:
            slices = [D // 4, D // 2, 3 * D // 4]
        else:
            idxs   = np.linspace(0, len(lesion_slices) - 1, min(3, len(lesion_slices)), dtype=int)
            slices = lesion_slices[idxs].tolist()

    print(f"Slices: {slices}")

    n = len(slices)
    fig, axes = plt.subplots(n, 4, figsize=(17, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = [
        "CC skull strip\n(estado atual, com anel)",
        "Erosão K=3 px\n(~1.5 mm)",
        "Erosão K=5 px\n(~2.5 mm)",
        "Ground truth",
    ]
    for col, t in enumerate(titles):
        axes[0, col].set_title(t, fontsize=11, fontweight="bold", pad=8)

    for row, sl in enumerate(slices):
        ax1, ax2, ax3, ax4 = axes[row]

        d_cc = display(vol_cc[sl])
        d_e3 = display(vol_e3[sl])
        d_e5 = display(vol_e5[sl])
        d_or = display(vol[sl])

        ax1.imshow(d_cc, cmap="gray", vmin=0, vmax=1)
        ax1.set_ylabel(f"z={sl}", fontsize=10)

        # K=3: mostra o que é removido em laranja
        ax2.imshow(d_e3, cmap="gray", vmin=0, vmax=1)
        ring3 = cc_mask[sl] & ~eroded3[sl]
        ov2 = np.zeros((*ring3.shape, 4))
        ov2[ring3] = [1.0, 0.5, 0.0, 0.55]
        ax2.imshow(ov2)

        # K=5: idem
        ax3.imshow(d_e5, cmap="gray", vmin=0, vmax=1)
        ring5 = cc_mask[sl] & ~eroded5[sl]
        ov3 = np.zeros((*ring5.shape, 4))
        ov3[ring5] = [1.0, 0.5, 0.0, 0.55]
        ax3.imshow(ov3)

        # Ground truth
        ax4.imshow(d_or, cmap="gray", vmin=0, vmax=1)
        if lbl[sl].any():
            ov4 = np.zeros((*lbl[sl].shape, 4))
            ov4[lbl[sl]] = [1.0, 0.1, 0.1, 0.65]
            ax4.imshow(ov4)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.axis("off")

    patch_orange = mpatches.Patch(color=(1.0, 0.5, 0.0, 0.7), label="Removido pela erosão (anel proposto)")
    patch_red    = mpatches.Patch(color=(1.0, 0.1, 0.1, 0.8), label="Lesão (ground truth)")
    fig.legend(handles=[patch_orange, patch_red], loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        f"Remoção do anel — erosão morfológica — caso {bare_id}\n"
        "Laranja = pixels removidos pelo passo proposto",
        fontsize=12, fontweight="bold",
    )

    out = OUTPUT_DIR / f"ring_removal_{bare_id}.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Salvo → {out}")
    plt.show()


if __name__ == "__main__":
    main()
