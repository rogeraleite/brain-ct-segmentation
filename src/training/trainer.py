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


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Dice + BCE combined loss.
    BCE alone can diverge on class-imbalanced masks (brain lesions are sparse).
    Dice alone can be unstable early in training when predictions are near 0.5.
    Combined gives stable gradients throughout.
    """
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
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = combined_loss(pred, y)
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
    save_by: str = "dice",
) -> dict:
    """
    Train model, save best checkpoint.

    save_by:
      "dice" — save when val_dice improves (default, good for full-volume eval)
      "loss" — save when val_loss improves (better for patch-based eval where
               empty patches inflate dice to 1.0 artificially)

    Returns history dict with lists: train_loss, val_loss, val_dice.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Scheduler mode depends on criterion: maximise dice or minimise loss
    sched_mode = "max" if save_by == "dice" else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sched_mode, factor=0.5, patience=5
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = 0.0
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice = evaluate(model, val_loader, device)

        scheduler.step(val_dice if save_by == "dice" else val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        improved = (save_by == "dice" and val_dice > best_dice) or \
                   (save_by == "loss" and val_loss < best_loss)

        flag = ""
        if improved:
            best_dice = val_dice
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": val_dice,
                    "val_loss": val_loss,
                },
                save_path,
            )
            flag = " ← best"

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f}{flag}"
        )

    best_label = f"val_loss={best_loss:.4f}" if save_by == "loss" else f"val_dice={best_dice:.4f}"
    print(f"\nTraining complete. Best {best_label}  (save_by='{save_by}')")
    return history
