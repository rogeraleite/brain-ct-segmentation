"""
Out-of-fold evaluation of the v15 Stage-1 detector across all folds.

Each case is scored by the fold detector that did NOT train on it, then the
82 case-level predictions are pooled for a leak-free detection P/R/F1. Swept
over case thresholds so the high-recall operating point (for the cascade gate)
is visible.

Example:
    PYTHONPATH=. python scripts/evaluate_detector_oof.py \
        --prefix models/detector_v15 --n-folds 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.data.loader import build_index, stratified_kfold_split
from src.data.slice_dataset import SliceHemorrhageDataset
from src.models.detector import build_detector


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(description="OOF detection eval for v15 detector")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--prefix", default="models/detector_v15")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-hw", type=int, default=224)
    p.add_argument("--case-thresholds", type=float, nargs="+",
                   default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    args = p.parse_args()

    device = get_device()
    records = build_index(args.data_root)
    splits = stratified_kfold_split(records, n_folds=args.n_folds, seed=args.seed)

    case_prob: dict[str, float] = {}
    case_label: dict[str, int] = {}

    for k, (_, val_records) in enumerate(splits):
        ckpt = f"{args.prefix}_fold{k}.pth"
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Missing detector checkpoint: {ckpt}")
        det = build_detector(pretrained=False).to(device)
        det.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
        det.eval()
        ds = SliceHemorrhageDataset(val_records, out_hw=args.out_hw, augment=False)
        with torch.no_grad():
            for i in range(len(ds)):
                x, y = ds[i]
                pr = torch.sigmoid(det(x.unsqueeze(0).to(device))).item()
                cid = ds.case_ids[i]
                case_prob[cid] = max(case_prob.get(cid, 0.0), pr)
                case_label[cid] = max(case_label.get(cid, 0), int(y.item()))

    cids = list(case_prob)
    probs = np.array([case_prob[c] for c in cids])
    labels = np.array([case_label[c] for c in cids])
    n_pos = int(labels.sum())
    print(f"OOF pooled: {len(cids)} cases ({n_pos} lesion, {len(cids) - n_pos} lesion-free)\n")
    print(f"{'case_thr':>8} | {'TP':>3} {'FP':>3} {'FN':>3} | {'precision':>9} {'recall':>7} {'F1':>6}")
    for ct in args.case_thresholds:
        pred = probs >= ct
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        print(f"{ct:>8} | {tp:>3} {fp:>3} {fn:>3} | {pr:>9.3f} {rc:>7.3f} {f1:>6.3f}")

    # Highest recall achieved and the threshold that reaches full recall (if any)
    order = np.argsort(-probs)
    full_recall_thr = None
    for ct in sorted(args.case_thresholds):
        if (probs[labels == 1] >= ct).all():
            full_recall_thr = ct
    print(f"\nLowest lesion-case probability: {probs[labels == 1].min():.3f} "
          f"(cases below any gate threshold are un-recoverable misses)")
    if full_recall_thr is not None:
        fp_at = int(((probs >= full_recall_thr) & (labels == 0)).sum())
        print(f"Full recall (R=1.0) holds up to case_thr={full_recall_thr} "
              f"with {fp_at} false-positive gates.")


if __name__ == "__main__":
    main()
