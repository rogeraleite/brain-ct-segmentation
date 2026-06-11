from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Loss functions ─────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """
    Soft Dice loss.
    Works on continuous predictions (no thresholding needed during training).
    """
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """
    Dice + BCE combined loss.
    pos_weight > 1.0 upweights lesion voxels in BCE to counter class imbalance.
    Equivalent to BCEWithLogitsLoss(pos_weight=...) but works on sigmoid outputs.
    """
    if pos_weight != 1.0:
        # per-element weight: pos_weight for lesion voxels, 1.0 for background
        w = pos_weight * target + (1.0 - target)
        bce = nn.functional.binary_cross_entropy(pred, target, weight=w)
    else:
        bce = nn.functional.binary_cross_entropy(pred, target)
    return dice_loss(pred, target) + bce


# ── Metrics ────────────────────────────────────────────────────────────────────

def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Hard Dice score (F1 for segmentation).
    pred is thresholded at `threshold` before computing.
    Returns value in [0, 1]. Higher is better.
    """
    pred_bin = (pred >= threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    if union == 0:
        return 1.0  # both pred and target are empty — correct prediction
    return float((2.0 * intersection + 1e-5) / (union + 1e-5))


# ── Training step ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = combined_loss(pred, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ── Validation step ────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (mean_val_loss, mean_val_dice)."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="  val  ", leave=False):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += combined_loss(pred, y).item()
            total_dice += dice_score(pred, y)

    n = len(loader)
    return total_loss / n, total_dice / n


# ── Full training loop ─────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    save_path: str,
    device: torch.device,
    pos_weight: float = 1.0,
    resume_path: str | None = None,
) -> dict:
    """
    Train model, save two checkpoints independently:
      save_path          — best val_loss  (e.g. v8.pth)
      save_path + _dice  — best val_dice  (e.g. v8_dice.pth)

    Scheduler always steps on val_loss (more stable than noisy patch-based dice).

    pos_weight: upweight lesion voxels in BCE (1.0 = standard, 50.0 = lesion 50× heavier).

    resume_path: if provided, loads model + optimizer + scheduler + history from
                 this checkpoint and continues training from the next epoch.

    Returns history dict with lists: train_loss, val_loss, val_dice.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Always step scheduler on loss (more stable than noisy patch-based dice)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Dice checkpoint path: v8.pth → v8_dice.pth
    _p = Path(save_path)
    dice_save_path = str(_p.parent / (_p.stem + "_dice" + _p.suffix))

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_loss = float("inf")
    best_dice = 0.0
    start_epoch = 1

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        history = ckpt.get("history", history)
        best_loss = ckpt.get("val_loss", float("inf"))
        best_dice = ckpt.get("val_dice", 0.0)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {resume_path}  (epoch {ckpt['epoch']}, val_loss={best_loss:.4f}, val_dice={best_dice:.4f})")

    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight)
        val_loss, val_dice = evaluate(model, val_loader, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        flag = ""

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "history": history,
                    "val_dice": val_dice,
                    "val_loss": val_loss,
                },
                save_path,
            )
            flag += " ← best_loss"

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "history": history,
                    "val_dice": val_dice,
                    "val_loss": val_loss,
                },
                dice_save_path,
            )
            flag += " ← best_dice"

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f}{flag}"
        )

    print(f"\nTraining complete.  best_loss={best_loss:.4f} ({save_path})  |  best_dice={best_dice:.4f} ({dice_save_path})")
    return history
