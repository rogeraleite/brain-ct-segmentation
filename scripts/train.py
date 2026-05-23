"""
Training entry point.

Usage:
    python scripts/train.py \
        --data-root data/raw \
        --save-path models/best_model.pth \
        --epochs 50 \
        --batch-size 2 \
        --lr 1e-3 \
        --val-split 0.2
"""

import argparse
import sys
from pathlib import Path

# Make src/ importable when running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.loader import build_index, train_val_split
from src.data.dataset import BrainCTDataset
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
    parser = argparse.ArgumentParser(description="Train Small3DUNet on brain CT data")
    parser.add_argument("--data-root",  default="data/raw",            help="Root dir with images/ and masks/ subdirs")
    parser.add_argument("--save-path",  default="models/best_model.pth", help="Path to save best model checkpoint")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val-split",  type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    records = build_index(args.data_root)
    train_records, val_records = train_val_split(
        records, val_fraction=args.val_split, seed=args.seed
    )
    print(f"Dataset: {len(records)} total | {len(train_records)} train | {len(val_records)} val")

    train_ds = BrainCTDataset(train_records, augment=not args.no_augment)
    val_ds   = BrainCTDataset(val_records,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Small3DUNet().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        device=device,
    )

    # ── Plot curves (non-blocking, skip in headless envs) ─────────────────────
    try:
        show_training_curves(history["train_loss"], history["val_loss"], history["val_dice"])
    except Exception:
        pass  # headless / no display


if __name__ == "__main__":
    main()
