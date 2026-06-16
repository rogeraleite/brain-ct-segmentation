"""
Gera skull masks para todos os volumes de treino usando SAM 2.

Estratégia:
  - Para cada fatia 2D: ponto positivo no centróide do tecido cerebral
    (pixels > -4 HU no volume brain-windowed) → SAM segmenta o interior
  - Fatias sem tecido detectável ficam como máscara vazia (False)
  - As máscaras são salvas em data/skull_masks/<caso>.nii.gz

Uso:
    PYTHONPATH=. python scripts/generate_skull_masks_sam.py
    PYTHONPATH=. python scripts/generate_skull_masks_sam.py --max-cases 5  # teste rápido
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import build_index, load_nifti
from src.preprocessing.transforms import BRAIN_HU_MIN, BRAIN_HU_MAX

SAM2_CHECKPOINT = "models/sam2/sam2.1_hiera_tiny.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_t.yaml"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def slice_to_rgb(slc_hu: np.ndarray) -> np.ndarray:
    """Converte fatia HU → RGB uint8 para o SAM (brain window [−5, 75])."""
    clipped = np.clip(slc_hu, BRAIN_HU_MIN, BRAIN_HU_MAX)
    norm = (clipped - BRAIN_HU_MIN) / (BRAIN_HU_MAX - BRAIN_HU_MIN)
    gray = (norm * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)  # (H, W, 3)


def tissue_centroid(slc_hu: np.ndarray) -> tuple[int, int] | None:
    """Centróide dos pixels de tecido (> −4 HU). Retorna (x, y) ou None."""
    tissue = slc_hu > (BRAIN_HU_MIN + 1.0)
    if not tissue.any():
        return None
    ys, xs = np.where(tissue)
    return int(xs.mean()), int(ys.mean())  # (x, y) = (col, row) para SAM


def prompt_points(slc_hu: np.ndarray) -> list[tuple[int, int]]:
    """
    Lista de pontos positivos para o SAM, em ordem de preferência.

    1. Centro geométrico da imagem — o cérebro está sempre centralizado em TC de crânio.
    2. Centróide do tecido — fallback para casos onde o centro é ar (fatias de borda).
    """
    H, W = slc_hu.shape
    points = [(W // 2, H // 2)]  # centro fixo da imagem (col, row)
    centroid = tissue_centroid(slc_hu)
    if centroid is not None and centroid != points[0]:
        points.append(centroid)
    return points


def build_sam2_predictor(device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    return SAM2ImagePredictor(model)


def _pick_best_mask(masks: np.ndarray, scores: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Escolhe a máscara mais adequada entre as 3 candidatas do SAM.

    Critérios em ordem de prioridade:
      1. Não toca a borda da imagem (o interior do crânio nunca alcança a borda)
      2. Tamanho entre 5% e 80% da imagem (evita fragmentos minúsculos ou máscaras gigantes)
      3. Maior score entre as candidatas válidas

    Se nenhuma candidata passar nos critérios, usa a de maior score como fallback.
    """
    total_px = H * W
    ranked = np.argsort(scores)[::-1]

    for idx in ranked:
        m = masks[idx]
        touches = m[0,:].any() or m[-1,:].any() or m[:,0].any() or m[:,-1].any()
        frac = m.sum() / total_px
        if not touches and 0.05 <= frac <= 0.80:
            return m.astype(bool)

    # Fallback: maior score sem toque na borda (ignora restrição de tamanho)
    for idx in ranked:
        m = masks[idx]
        touches = m[0,:].any() or m[-1,:].any() or m[:,0].any() or m[:,-1].any()
        if not touches:
            return m.astype(bool)

    # Último recurso: maior score
    return masks[ranked[0]].astype(bool)


def generate_skull_mask_sam(volume: np.ndarray, predictor) -> np.ndarray:
    """
    Gera máscara intracraniana 3D (D, H, W) usando SAM 2 fatia por fatia.

    Para cada fatia:
      1. Converte para RGB brain-windowed
      2. Centróide dos pixels de tecido = ponto positivo (interior do cérebro)
      3. SAM prediz 3 máscaras candidatas para esse ponto
      4. _pick_best_mask seleciona a candidata que não toca a borda e tem tamanho razoável

    Retorna: bool array (D, H, W), True = interior do cérebro
    """
    D, H, W = volume.shape
    mask_3d = np.zeros((D, H, W), dtype=bool)

    for i in range(D):
        slc = volume[i]
        points = prompt_points(slc)
        if not points:
            continue

        rgb = slice_to_rgb(slc)
        predictor.set_image(rgb)

        # Tenta cada ponto em ordem; aceita o primeiro que gera uma máscara válida
        chosen = None
        for pt in points:
            masks, scores, _ = predictor.predict(
                point_coords=np.array([pt]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            candidate = _pick_best_mask(masks, scores, H, W)
            frac = candidate.sum() / (H * W)
            if 0.05 <= frac <= 0.80:
                chosen = candidate
                break

        if chosen is None:
            # Fallback: máscara de maior score do último ponto tentado
            chosen = masks[int(np.argmax(scores))].astype(bool)

        mask_3d[i] = chosen

    return mask_3d


def save_nifti_mask(mask: np.ndarray, affine: np.ndarray, out_path: Path) -> None:
    """Salva máscara (D, H, W) bool como NIfTI. Transpõe para convenção nibabel (H, W, D)."""
    arr_nib = np.transpose(mask.astype(np.uint8), (1, 2, 0))
    nib.save(nib.Nifti1Image(arr_nib, affine), str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--out-dir",   default="data/skull_masks")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    predictor = build_sam2_predictor(device)

    records = build_index(args.data_root)
    if args.max_cases:
        records = records[:args.max_cases]

    n_ok = 0
    for rec in tqdm(records, desc="Generating skull masks (SAM2)"):
        img_path = Path(rec["image"])
        out_path = out_dir / img_path.name

        if out_path.exists():
            continue  # já gerado, pula

        volume, _ = load_nifti(str(img_path))

        # Precisa do affine original para salvar no mesmo espaço
        nib_img  = nib.load(str(img_path))
        affine   = nib_img.affine

        with torch.inference_mode():
            skull_mask = generate_skull_mask_sam(volume, predictor)

        save_nifti_mask(skull_mask, affine, out_path)
        n_ok += 1

    print(f"\nConcluído: {n_ok} máscaras salvas em {out_dir}/")


if __name__ == "__main__":
    main()
