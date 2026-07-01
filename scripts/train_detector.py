"""
Train the v15 Stage-1 hemorrhage detector on one stratified fold.

Slice-level binary classifier (does this axial slice contain hemorrhage?).
Reuses the exact v14 stratified 5-fold split (by CASE, so slices from one
patient never straddle train/val — no leakage), so detector and segmenter
share folds and the downstream OOF cascade stays leak-free.

Reports both slice-level and case-level precision/recall/F1. A case is called
positive if its max slice probability >= the case threshold. The detector is
tuned for HIGH RECALL: a missed case in the cascade is an un-recoverable miss.

Example:
    PYTHONPATH=. python scripts/train_detector.py \
        --fold 0 --n-folds 5 --epochs 25 \
        --save-path models/detector_v15_fold0.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.loader import build_index, stratified_kfold_split
from src.data.slice_dataset import SliceHemorrhageDataset
from src.models.detector import build_detector


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return p, r, f


@torch.no_grad()
def evaluate(model, ds, device, batch_size, slice_thr=0.5, case_thr=0.5):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = np.zeros(len(ds), dtype=np.float32)
    i = 0
    for x, _ in loader:
        p = torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy()
        probs[i:i + len(p)] = p
        i += len(p)

    y = ds.labels
    # slice-level
    sp = (probs >= slice_thr).astype(np.float32)
    tp = float(((sp == 1) & (y == 1)).sum())
    fp = float(((sp == 1) & (y == 0)).sum())
    fn = float(((sp == 0) & (y == 1)).sum())
    slice_prf = prf(tp, fp, fn)

    # case-level: a case is positive if ANY slice prob >= case_thr (max-pool)
    cases: dict[str, list[float]] = {}
    labels: dict[str, int] = {}
    for cid, pr, yy in zip(ds.case_ids, probs, y):
        cases.setdefault(cid, []).append(pr)
        labels[cid] = max(labels.get(cid, 0), int(yy))
    ctp = cfp = cfn = 0.0
    for cid, prs in cases.items():
        pred = 1 if max(prs) >= case_thr else 0
        gt = labels[cid]
        ctp += pred == 1 and gt == 1
        cfp += pred == 1 and gt == 0
        cfn += pred == 0 and gt == 1
    case_prf = prf(ctp, cfp, cfn)
    return slice_prf, case_prf, probs


def main() -> None:
    p = argparse.ArgumentParser(description="Train v15 Stage-1 slice detector on one fold")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--save-path", default="models/detector_v15_fold0.pth")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out-hw", type=int, default=224)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")

    records = build_index(args.data_root)
    splits = stratified_kfold_split(records, n_folds=args.n_folds, seed=args.seed)
    train_records, val_records = splits[args.fold]
    print(f"Split: fold {args.fold} | {len(train_records)} train cases | {len(val_records)} val cases")

    train_ds = SliceHemorrhageDataset(train_records, out_hw=args.out_hw, augment=True)
    val_ds = SliceHemorrhageDataset(val_records, out_hw=args.out_hw, augment=False)
    print(f"Slices: {len(train_ds)} train ({int(train_ds.labels.sum())} pos) | "
          f"{len(val_ds)} val ({int(val_ds.labels.sum())} pos)")

    pos_weight = torch.tensor([train_ds.pos_weight()], device=device)
    print(f"pos_weight (n_neg/n_pos): {pos_weight.item():.2f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = build_detector(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    best_case_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, yb in train_loader:
            x, yb = x.to(device), yb.to(device)
            opt.zero_grad()
            logit = model(x).squeeze(1)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
            tot += loss.item()

        (sp, sr, sf), (cp, cr, cf), _ = evaluate(model, val_ds, device, args.batch_size)
        flag = ""
        if cf > best_case_f1:
            best_case_f1 = cf
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "case_f1": cf, "args": vars(args)}, args.save_path)
            flag = " ← best_case_f1"
        print(f"Epoch {epoch:02d} | loss={tot/len(train_loader):.4f} | "
              f"slice P/R/F1={sp:.3f}/{sr:.3f}/{sf:.3f} | "
              f"case P/R/F1={cp:.3f}/{cr:.3f}/{cf:.3f}{flag}")

    print(f"\nDone. best case-F1={best_case_f1:.4f} → {args.save_path}")


if __name__ == "__main__":
    main()
