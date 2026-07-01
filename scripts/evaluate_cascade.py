"""
v15 detect-then-segment cascade evaluation on one fold.

Stage 1 (detector) produces a per-case P(hemorrhage) = max slice probability.
Stage 2 (v14 sliding-window segmenter) produces a probability map that is
thresholded + small-component-filtered into a mask. The cascade GATES stage 2:
if the case probability is below the case threshold, the prediction is forced
empty (the whole point — stop inventing false positives on lesion-free cases).

Reports, honestly split so the empty-case Dice=1.0 bonus is not mistaken for a
segmentation gain:
  - Dice over LESION-BEARING cases only  (did segmentation actually improve?)
  - Dice over ALL cases                  (includes the gating bonus)
  - case-level detection P/R/F1          (the gate's own quality)
Also prints the un-gated baseline so the delta is visible.

Example:
    PYTHONPATH=. python scripts/evaluate_cascade.py --fold 0 \
        --detector models/detector_v15_fold0.pth \
        --seg-prefix models/best_model_slidingWindow_v14 --seg-suffix _f1 \
        --seg-threshold 0.3 --seg-min-cc-ml 2.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from tqdm import tqdm

from api.inference import compute_volume_ml, get_device
from api.inference_sw import load_model_sw
from src.data.loader import build_index, load_nifti, stratified_kfold_split
from src.data.slice_dataset import SliceHemorrhageDataset
from src.models.detector import build_detector
from src.inference.sliding_window import sliding_window_predict, remove_small_components
from evaluate import segmentation_metrics


def dice_of(pred, target):
    return float(segmentation_metrics(pred, target)["dice"])


def main() -> None:
    p = argparse.ArgumentParser(description="v15 cascade eval on one fold")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--detector", default="models/detector_v15_fold0.pth")
    p.add_argument("--out-hw", type=int, default=224)
    p.add_argument("--seg-prefix", default="models/best_model_slidingWindow_v14")
    p.add_argument("--seg-suffix", default="_f1")
    p.add_argument("--seg-threshold", type=float, default=0.3)
    p.add_argument("--seg-min-cc-ml", type=float, default=2.0)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--case-thresholds", type=float, nargs="+",
                   default=[0.3, 0.4, 0.5, 0.6, 0.7])
    args = p.parse_args()

    device = get_device()
    records = build_index(args.data_root)
    _, val_records = stratified_kfold_split(records, n_folds=args.n_folds, seed=args.seed)[args.fold]

    # ── Stage 1: per-case detector probability (max over slices) ──────────────
    det = build_detector(pretrained=False).to(device)
    det.load_state_dict(torch.load(args.detector, map_location=device)["model_state_dict"])
    det.eval()
    ds = SliceHemorrhageDataset(val_records, out_hw=args.out_hw, augment=False)
    case_prob: dict[str, float] = {}
    with torch.no_grad():
        for i in range(len(ds)):
            x, _ = ds[i]
            pr = torch.sigmoid(det(x.unsqueeze(0).to(device))).item()
            cid = ds.case_ids[i]
            case_prob[cid] = max(case_prob.get(cid, 0.0), pr)

    # ── Stage 2: segmenter prob map + threshold/min-cc → mask (per case) ───────
    seg, _ = load_model_sw(f"{args.seg_prefix}_fold{args.fold}{args.seg_suffix}.pth")
    per_case = []  # (cid, has_lesion, seg_dice, empty_dice, voxel_pred_nonzero)
    for r in tqdm(val_records, desc="segmenting"):
        volume, spacing = load_nifti(r["image"])
        target, _ = load_nifti(r["mask"]); target = (target > 0.5).astype(np.uint8)
        voxel_mm3 = float(np.prod(spacing))
        spacing_hw_mm = float(min(spacing[1], spacing[2]))
        prob = sliding_window_predict(
            volume=volume, model=seg, device=device, stride=args.stride,
            spacing_hw_mm=spacing_hw_mm, skull_strip=True, skull_excl_mm=0.0, return_prob=True,
        )
        pred = (prob >= args.seg_threshold).astype(np.uint8)
        if args.seg_min_cc_ml > 0.0:
            pred = remove_small_components(pred, args.seg_min_cc_ml, voxel_mm3)
        cid = Path(r["image"]).name
        per_case.append({
            "cid": cid,
            "has_lesion": int(target.sum() > 0),
            "prob": case_prob.get(cid, 0.0),
            "seg_dice": dice_of(pred, target),          # ungated segmentation
            "empty_dice": 1.0 if target.sum() == 0 else 0.0,  # if gated to empty
        })

    lesion = [c for c in per_case if c["has_lesion"]]
    n_all, n_les = len(per_case), len(lesion)

    print(f"\nFold {args.fold}: {n_all} val cases ({n_les} with lesion, {n_all - n_les} lesion-free)")
    print(f"Seg operating point: thr={args.seg_threshold} min_cc={args.seg_min_cc_ml}  ({args.seg_prefix}{args.seg_suffix})\n")

    # Ungated baseline (always segment)
    base_all = np.mean([c["seg_dice"] for c in per_case])
    base_les = np.mean([c["seg_dice"] for c in lesion])
    print(f"UNGATED baseline   | Dice all={base_all:.4f}  lesion-only={base_les:.4f}  "
          f"median-all={np.median([c['seg_dice'] for c in per_case]):.4f}")

    print("\nGated cascade (case_thr sweep):")
    print(f"{'thr':>4} | {'det P/R/F1':>16} | {'Dice all':>8} | {'Dice lesion':>11} | {'median all':>10}")
    for ct in args.case_thresholds:
        # detection metrics
        tp = sum(c["prob"] >= ct and c["has_lesion"] for c in per_case)
        fp = sum(c["prob"] >= ct and not c["has_lesion"] for c in per_case)
        fn = sum(c["prob"] < ct and c["has_lesion"] for c in per_case)
        dp = tp / (tp + fp) if tp + fp else 0.0
        dr = tp / (tp + fn) if tp + fn else 0.0
        df = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        # gated dice: negative-called cases → empty prediction
        dices_all, dices_les = [], []
        for c in per_case:
            gated_empty = c["prob"] < ct
            d = c["empty_dice"] if gated_empty else c["seg_dice"]
            dices_all.append(d)
            if c["has_lesion"]:
                dices_les.append(d)
        print(f"{ct:>4} | {dp:.2f}/{dr:.2f}/{df:.2f}    | "
              f"{np.mean(dices_all):>8.4f} | {np.mean(dices_les):>11.4f} | {np.median(dices_all):>10.4f}")


if __name__ == "__main__":
    main()
