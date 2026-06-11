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

from src.data.loader import build_index, train_val_split
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
    parser.add_argument("--skull-strip", action="store_true",
                        help="Zero out extracranial voxels before normalization")
    parser.add_argument("--pos-weight",  type=float, default=50.0)
    parser.add_argument("--resume",      default=None,
                        help="Path to checkpoint to resume training from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device      : {device}")
    print(f"Patch shape : ({D_MAX}, {PATCH_HW}, {PATCH_HW})  [native resolution, no resize]")

    records = build_index(args.data_root)
    train_records, val_records = train_val_split(
        records, val_fraction=args.val_split, seed=args.seed
    )
    print(f"Dataset     : {len(records)} total | {len(train_records)} train | {len(val_records)} val")

    train_ds = PatchBrainCTDataset(train_records, augment=not args.no_augment, skull_strip=args.skull_strip, no_elastic=args.no_elastic)
    val_ds   = PatchBrainCTDataset(val_records,   augment=False,               skull_strip=args.skull_strip)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Small3DUNet accepts any input where D, H, W are divisible by 8.
    # (40, 128, 128): 40/8=5, 128/8=16 — both valid.
    model = Small3DUNet().to(device)
    print(f"Model params: {model.count_parameters():,}")

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"pos_weight  : {args.pos_weight}  (lesion voxels weighted {args.pos_weight:.0f}× in BCE)")
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
    )

    try:
        show_training_curves(history["train_loss"], history["val_loss"], history["val_dice"])
    except Exception:
        pass


if __name__ == "__main__":
    main()
