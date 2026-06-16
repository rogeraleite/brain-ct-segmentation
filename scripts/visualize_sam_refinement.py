"""
Visualiza o tratamento sequencial pós-SAM:

  Passo 1 — Original CT (brain window)
  Passo 2 — Após SAM mask (anel ósseo + blobs periféricos ainda presentes)
  Passo 3 — Após CC refinement (só parênquima + hemorragia, sem anel nem blobs)

Usage:
    PYTHONPATH=. .venv/bin/python scripts/visualize_sam_refinement.py \
        --case 051 --slices 10 15 20
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, label as nd_label

from src.data.loader import load_skull_mask
from src.preprocessing.transforms import BRAIN_HU_MIN, BRAIN_HU_MAX

DATA_ROOT  = Path("data/raw")
MASK_DIR   = Path("data/skull_masks")
OUTPUT_DIR = Path("reports/figures")


def load_volume(bare_id: str):
    img  = nib.load(DATA_ROOT / "images" / f"{bare_id}.nii.gz")
    msk  = nib.load(DATA_ROOT / "masks"  / f"{bare_id}.nii.gz")
    vol  = np.transpose(img.get_fdata().astype(np.float32), (2, 0, 1))
    lbl  = np.transpose(msk.get_fdata().astype(np.float32), (2, 0, 1)) > 0.5
    return vol, lbl


def to_display(vol: np.ndarray) -> np.ndarray:
    return np.clip((vol - BRAIN_HU_MIN) / (BRAIN_HU_MAX - BRAIN_HU_MIN), 0, 1)


def refine_with_cc(vol_sam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Para cada slice: encontra o maior CC de pixels não-zerados que não
    toca a borda da imagem. Zera todo o resto (anel ósseo + blobs periféricos).

    Retorna:
        vol_refined  — volume com apenas o cérebro interior
        brain_mask   — bool (D, H, W) True onde o cérebro foi preservado
    """
    D, H, W = vol_sam.shape
    brain_mask = np.zeros((D, H, W), dtype=bool)

    for i in range(D):
        slc = vol_sam[i]
        tissue = slc > (BRAIN_HU_MIN + 1.0)   # qualquer pixel não-zerado
        if not tissue.any():
            continue

        labeled, n_cc = nd_label(tissue)
        if n_cc == 0:
            continue

        # Exclui CCs que tocam a borda (skull ring toca a borda do crânio,
        # blobs periféricos também podem tocar)
        border_labels = set(
            labeled[0, :].tolist() + labeled[-1, :].tolist() +
            labeled[:, 0].tolist() + labeled[:, -1].tolist()
        ) - {0}

        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        # Zera CCs de borda
        for bl in border_labels:
            counts[bl] = 0

        if counts.max() == 0:
            # Todos os CCs tocam a borda — pega o maior como fallback
            counts2 = np.bincount(labeled.ravel())
            counts2[0] = 0
            best = int(np.argmax(counts2))
        else:
            best = int(np.argmax(counts))

        brain_mask[i] = labeled == best

    vol_refined = np.where(brain_mask, vol_sam, BRAIN_HU_MIN)
    return vol_refined, brain_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="051")
    parser.add_argument("--slices", nargs="+", type=int, default=None)
    args = parser.parse_args()

    bare_id = args.case.replace(".nii.gz", "")
    print(f"Carregando caso {bare_id}...")

    vol, lbl = load_volume(bare_id)
    D, H, W  = vol.shape

    # Passo 2: SAM mask
    sam_mask = load_skull_mask(str(MASK_DIR / f"{bare_id}.nii.gz"))
    vol_sam  = np.where(sam_mask, vol, BRAIN_HU_MIN)

    # Passo 3: CC refinement sobre o volume já pós-SAM
    print("Aplicando CC refinement...")
    vol_ref, brain_mask = refine_with_cc(vol_sam)

    # Seleciona slices
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
    fig, axes = plt.subplots(n, 4, figsize=(17, 4.8 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = [
        "Passo 1\nOriginal CT (brain window)",
        "Passo 2\nApós SAM mask\n(anel ósseo + blobs ainda presentes)",
        "Passo 3\nApós CC refinement\n(só cérebro interior)",
        "Ground truth\n(lesão em vermelho)",
    ]
    for col, t in enumerate(titles):
        axes[0, col].set_title(t, fontsize=10, fontweight="bold", pad=8)

    for row, sl in enumerate(slices):
        ax1, ax2, ax3, ax4 = axes[row]

        d1 = to_display(vol[sl])
        d2 = to_display(vol_sam[sl])
        d3 = to_display(vol_ref[sl])

        ax1.imshow(d1, cmap="gray", vmin=0, vmax=1)
        ax1.set_ylabel(f"z = {sl}", fontsize=10)

        # Passo 2: mostra o que foi zerado pelo SAM (azul) e o que sobrou
        ax2.imshow(d2, cmap="gray", vmin=0, vmax=1)
        removed_by_sam = ~sam_mask[sl]
        ov2 = np.zeros((*removed_by_sam.shape, 4))
        ov2[removed_by_sam] = [0.15, 0.35, 0.95, 0.3]
        ax2.imshow(ov2)

        # Passo 3: mostra o que foi removido pelo CC (laranja) em cima do vol_sam
        ax3.imshow(d3, cmap="gray", vmin=0, vmax=1)
        removed_by_cc = sam_mask[sl] & ~brain_mask[sl]   # estava no SAM mas foi removido pelo CC
        ov3 = np.zeros((*removed_by_cc.shape, 4))
        ov3[removed_by_cc] = [1.0, 0.5, 0.0, 0.45]      # laranja = removido pelo CC
        ax3.imshow(ov3)

        # Ground truth
        ax4.imshow(d1, cmap="gray", vmin=0, vmax=1)
        if lbl[sl].any():
            ov4 = np.zeros((*lbl[sl].shape, 4))
            ov4[lbl[sl]] = [1.0, 0.1, 0.1, 0.65]
            ax4.imshow(ov4)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.axis("off")

    patch_sam = mpatches.Patch(color=(0.15, 0.35, 0.95, 0.6), label="Removido pelo SAM (extracraniano)")
    patch_cc  = mpatches.Patch(color=(1.0, 0.5, 0.0, 0.7),   label="Removido pelo CC refinement (anel + blobs)")
    patch_les = mpatches.Patch(color=(1.0, 0.1, 0.1, 0.8),   label="Lesão (ground truth)")
    fig.legend(handles=[patch_sam, patch_cc, patch_les], loc="lower center",
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        f"Tratamento sequencial pós-SAM — caso {bare_id}\n"
        "Passo 3 = o que o modelo deveria ver no treino",
        fontsize=13, fontweight="bold",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"sam_refinement_{bare_id}.png"
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Salvo → {out}")
    plt.show()


if __name__ == "__main__":
    main()
