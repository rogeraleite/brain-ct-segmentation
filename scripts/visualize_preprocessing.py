"""
Visualize the preprocessing pipeline for each model version.

Columns per slice:
  Col 1 - Original CT (brain window, HU)
  Col 2 - Após CC skull strip (treino E inferência v11 — mesmo algoritmo)
  Col 3 - Normalizado [0, 1] (o que o modelo realmente vê)
  Col 4 - Ground truth (lesão em vermelho)

Usage:
    PYTHONPATH=. .venv/bin/python scripts/visualize_preprocessing.py \
        --case 051 --slices 10 15 20
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import nibabel as nib
import numpy as np

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


def to_display(vol: np.ndarray) -> np.ndarray:
    """Brain window + normaliza para [0,1] para exibição."""
    return np.clip((vol - BRAIN_HU_MIN) / (BRAIN_HU_MAX - BRAIN_HU_MIN), 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="051")
    parser.add_argument("--slices", nargs="+", type=int, default=None)
    args = parser.parse_args()

    bare_id = args.case.replace(".nii.gz", "")
    print(f"Carregando caso {bare_id}...")
    vol, lbl = load_volume(bare_id)
    D, H, W  = vol.shape

    # CC skull strip (mesmo algoritmo no treino e na inferência — v11)
    print("Computando CC skull strip...")
    cc_mask  = extract_intracranial_mask_cc(vol)
    vol_cc   = np.where(cc_mask, vol, BRAIN_HU_MIN)   # HU com extracraniano zerado
    vol_cc_n = normalize(vol_cc)                        # [0,1] — o que o modelo vê

    vol_disp = to_display(vol)  # original para exibição

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

    col_titles = [
        "Passo 1 — Original CT\n(brain window, HU bruto)",
        "Passo 2 — CC skull strip\n(extracraniano → -5 HU)\ntreino = inferência",
        "Passo 3 — Normalizado [0, 1]\n(o que o modelo vê)",
        "Ground truth\n(lesão em vermelho)",
    ]
    for col, t in enumerate(col_titles):
        axes[0, col].set_title(t, fontsize=10, fontweight="bold", pad=8)

    for row, sl in enumerate(slices):
        ax1, ax2, ax3, ax4 = axes[row]

        ax1.imshow(vol_disp[sl], cmap="gray", vmin=0, vmax=1)
        ax1.set_ylabel(f"z = {sl}", fontsize=10)

        # Passo 2: CC skull strip — azul = removido
        ax2.imshow(to_display(vol_cc[sl]), cmap="gray", vmin=0, vmax=1)
        removed = ~cc_mask[sl]
        ov2 = np.zeros((*removed.shape, 4))
        ov2[removed] = [0.15, 0.35, 0.95, 0.35]
        ax2.imshow(ov2)

        # Passo 3: normalizado — o que o modelo recebe como input
        ax3.imshow(vol_cc_n[sl], cmap="gray", vmin=0, vmax=1)
        # anota os valores HU de referência no colormap
        ax3.text(5, H - 10, "0.0 = -5 HU (fundo)", color="cyan",
                 fontsize=6.5, va="bottom")
        ax3.text(5, H - 22, "0.69 = 50 HU (hemorragia mín.)", color="yellow",
                 fontsize=6.5, va="bottom")
        ax3.text(5, H - 34, "1.0 = 75 HU (hemorragia / osso)", color="red",
                 fontsize=6.5, va="bottom")

        # Passo 4: ground truth overlay
        ax4.imshow(vol_disp[sl], cmap="gray", vmin=0, vmax=1)
        if lbl[sl].any():
            ov4 = np.zeros((*lbl[sl].shape, 4))
            ov4[lbl[sl]] = [1.0, 0.1, 0.1, 0.65]
            ax4.imshow(ov4)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.axis("off")

    patch_blue = mpatches.Patch(color=(0.15, 0.35, 0.95, 0.6),
                                label="Removido pelo CC skull strip (extracraniano → 0)")
    patch_red  = mpatches.Patch(color=(1.0, 0.1, 0.1, 0.8),
                                label="Lesão (ground truth)")
    fig.legend(handles=[patch_blue, patch_red], loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        f"Pipeline v11 — caso {bare_id}\n"
        "CC skull strip idêntico no treino e na inferência — sem mismatch de distribuição",
        fontsize=12, fontweight="bold",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"preprocessing_v11_{bare_id}.png"
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Salvo → {out}")


if __name__ == "__main__":
    main()
