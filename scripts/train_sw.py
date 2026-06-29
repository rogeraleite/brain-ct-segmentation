"""
Training entry point for the sliding-window model.

Trains Small3DUNet on native-resolution patches (D_MAX×128×128)
instead of globally-resized volumes.  Saves best checkpoint to
models/best_model_slidingWindow.pth.

Usage:
    python scripts/train_sw.py \\
        --data-root data/raw \\
        --save-path models/best_model_slidingWindow.pth \\
        --epochs 50 --batch-size 2 --lr 1e-3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.loader import build_index, train_val_split, stratified_kfold_split
from src.data.patch_dataset import PatchBrainCTDataset, D_MAX, PATCH_HW
from src.models.unet import Small3DUNet
from src.training.trainer import train
from src.visualization.plots import show_training_curves


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train sliding-window Small3DUNet on native-resolution brain CT patches"
    )
    parser.add_argument("--data-root",  default="data/raw")
    parser.add_argument("--save-path",  default="models/best_model_slidingWindow_v8.pth")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val-split",  type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--no-augment",  action="store_true")
    parser.add_argument("--no-elastic",  action="store_true",
                        help="Disable elastic deformation augmentation")
    parser.add_argument("--no-lateral-flip", action="store_true",
                        help="Disable the v14 left-right (lateral) flip augmentation")
    parser.add_argument("--skull-strip", action="store_true",
                        help="Zero out extracranial voxels before normalization (old HU-threshold method)")
    parser.add_argument("--skull-mask-dir", default=None,
                        help="Directory with precomputed SAM skull masks (e.g. data/skull_masks). "
                             "When provided, overrides --skull-strip with accurate per-case masks.")
    parser.add_argument("--pos-weight",    type=float, default=50.0)
    parser.add_argument("--f1-min-epoch", type=int, default=0,
                        help="Only save the best-F1 checkpoint after this epoch (avoids early flukes).")
    parser.add_argument("--resume",      default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--loss", choices=["bce_dice", "focal_tversky"], default="bce_dice",
                        help="Loss function. bce_dice = legacy Dice+weighted BCE (uses --pos-weight); "
                             "focal_tversky = Focal Tversky + balanced BCE (uses --tversky-* flags).")
    parser.add_argument("--tversky-alpha", type=float, default=0.3, help="Focal Tversky FN weight")
    parser.add_argument("--tversky-beta",  type=float, default=0.7, help="Focal Tversky FP weight")
    parser.add_argument("--tversky-gamma", type=float, default=1.33, help="Focal Tversky focusing exponent")
    parser.add_argument("--fold", type=int, default=None,
                        help="If set, use stratified k-fold split and train this fold "
                             "(0-based) instead of the random --val-split.")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of folds for --fold stratified split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device      : {device}")
    print(f"Patch shape : ({D_MAX}, {PATCH_HW}, {PATCH_HW})  [native resolution, no resize]")

    records = build_index(args.data_root)
    if args.fold is not None:
        splits = stratified_kfold_split(records, n_folds=args.n_folds, seed=args.seed)
        if not 0 <= args.fold < len(splits):
            raise ValueError(f"--fold {args.fold} out of range for {args.n_folds} folds")
        train_records, val_records = splits[args.fold]
        print(f"Split       : stratified {args.n_folds}-fold | fold {args.fold}")
    else:
        train_records, val_records = train_val_split(
            records, val_fraction=args.val_split, seed=args.seed
        )
    print(f"Dataset     : {len(records)} total | {len(train_records)} train | {len(val_records)} val")

    train_ds = PatchBrainCTDataset(
        train_records, augment=not args.no_augment,
        skull_strip=args.skull_strip, no_elastic=args.no_elastic,
        no_lateral_flip=args.no_lateral_flip,
        skull_mask_dir=args.skull_mask_dir,
    )
    val_ds = PatchBrainCTDataset(
        val_records, augment=False,
        skull_strip=args.skull_strip,
        skull_mask_dir=args.skull_mask_dir,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Small3DUNet accepts any input where D, H, W are divisible by 8.
    # (40, 128, 128): 40/8=5, 128/8=16 — both valid.
    model = Small3DUNet().to(device)
    print(f"Model params: {model.count_parameters():,}")

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    if args.loss == "focal_tversky":
        print(f"loss        : focal_tversky  (alpha={args.tversky_alpha}, beta={args.tversky_beta}, "
              f"gamma={args.tversky_gamma}) + 0.5·BCE  [pos_weight ignored]")
    else:
        print(f"loss        : bce_dice  pos_weight={args.pos_weight} "
              f"(lesion voxels weighted {args.pos_weight:.0f}× in BCE)")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        device=device,
        pos_weight=args.pos_weight,
        resume_path=args.resume,
        f1_min_epoch=args.f1_min_epoch,
        loss_name=args.loss,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_gamma=args.tversky_gamma,
    )

    curve_path = str(Path(args.save_path).parent.parent / "tmp" / (Path(args.save_path).stem + "_curves.png"))
    try:
        show_training_curves(history["train_loss"], history["val_loss"], history["val_f1"], save_path=curve_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
