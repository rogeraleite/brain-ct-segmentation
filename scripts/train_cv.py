"""
Cross-validation driver for the sliding-window model (v14).

Trains one model per stratified fold by invoking train_sw.py as a subprocess,
saving models/<prefix>_fold{k}.pth (+ _f1). Each fold's checkpoints can then
be evaluated out-of-fold by scripts/evaluate_cv.py for a leak-free 82-case score.

v14 adds the left-right (lateral) flip augmentation and selects the secondary
checkpoint on micro-aggregated val F1 (precision + recall), not per-patch Dice.
Evaluate the F1 checkpoints with `evaluate_cv.py --suffix _f1`.

Example:
    PYTHONPATH=. python scripts/train_cv.py \
        --loss focal_tversky --tversky-alpha 0.3 --tversky-beta 0.7 --tversky-gamma 1.33 \
        --n-folds 5 --skull-strip --no-elastic --epochs 150 \
        --save-prefix models/best_model_slidingWindow_v14
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import build_index, stratified_kfold_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train k folds of the sliding-window model")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--save-prefix", default="models/best_model_slidingWindow_v14")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", choices=["bce_dice", "focal_tversky"], default="focal_tversky")
    p.add_argument("--tversky-alpha", type=float, default=0.3)
    p.add_argument("--tversky-beta", type=float, default=0.7)
    p.add_argument("--tversky-gamma", type=float, default=1.33)
    p.add_argument("--pos-weight", type=float, default=50.0)
    p.add_argument("--f1-min-epoch", type=int, default=50)
    p.add_argument("--skull-strip", action="store_true")
    p.add_argument("--skull-mask-dir", default=None)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-elastic", action="store_true")
    p.add_argument("--no-lateral-flip", action="store_true")
    p.add_argument("--only-fold", type=int, default=None,
                   help="Train just this fold index instead of all folds.")
    return p.parse_args()


def verify_disjoint(records, n_folds, seed) -> None:
    """Assert val sets are disjoint and cover every case exactly once."""
    splits = stratified_kfold_split(records, n_folds=n_folds, seed=seed)
    seen: dict[str, int] = {}
    for k, (_, val_records) in enumerate(splits):
        for r in val_records:
            cid = Path(r["image"]).name
            if cid in seen:
                raise AssertionError(f"case {cid} appears in folds {seen[cid]} and {k}")
            seen[cid] = k
    assert len(seen) == len(records), (
        f"OOF coverage gap: {len(seen)} val cases vs {len(records)} total"
    )
    sizes = [len(v) for _, v in splits]
    print(f"Fold disjointness OK — val sizes {sizes}, total {len(seen)} cases covered once.")


def main() -> None:
    args = parse_args()
    records = build_index(args.data_root)
    verify_disjoint(records, args.n_folds, args.seed)

    folds = [args.only_fold] if args.only_fold is not None else range(args.n_folds)
    suffix = Path(args.save_prefix).suffix or ".pth"
    stem = args.save_prefix[: -len(suffix)] if args.save_prefix.endswith(suffix) else args.save_prefix

    for k in folds:
        save_path = f"{stem}_fold{k}{suffix}"
        print(f"\n{'='*70}\nFOLD {k} → {save_path}\n{'='*70}")
        cmd = [
            sys.executable, "scripts/train_sw.py",
            "--data-root", args.data_root,
            "--save-path", save_path,
            "--fold", str(k),
            "--n-folds", str(args.n_folds),
            "--seed", str(args.seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--loss", args.loss,
            "--tversky-alpha", str(args.tversky_alpha),
            "--tversky-beta", str(args.tversky_beta),
            "--tversky-gamma", str(args.tversky_gamma),
            "--pos-weight", str(args.pos_weight),
            "--f1-min-epoch", str(args.f1_min_epoch),
        ]
        if args.skull_strip:
            cmd.append("--skull-strip")
        if args.skull_mask_dir:
            cmd += ["--skull-mask-dir", args.skull_mask_dir]
        if args.no_augment:
            cmd.append("--no-augment")
        if args.no_elastic:
            cmd.append("--no-elastic")
        if args.no_lateral_flip:
            cmd.append("--no-lateral-flip")

        subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent.parent))

    print("\nAll folds complete.")


if __name__ == "__main__":
    main()
