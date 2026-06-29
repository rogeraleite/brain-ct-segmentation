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


def focal_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 1.33,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Focal Tversky loss — a generalisation of Dice loss with tunable FP/FN penalties.

    alpha weights false negatives, beta weights false positives (alpha + beta = 1
    recovers the Dice family; alpha = beta = 0.5 is exactly soft Dice). For an
    over-predicting (FP-dominated) model, set beta > alpha to penalise false
    positives. gamma > 1 focuses gradient on hard (low-overlap) patches.
    """
    p = pred.view(-1)
    t = target.view(-1)
    tp = (p * t).sum()
    fn = (t * (1.0 - p)).sum()
    fp = (p * (1.0 - t)).sum()
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return (1.0 - tversky) ** gamma


def make_loss(
    loss_name: str = "bce_dice",
    pos_weight: float = 1.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    tversky_gamma: float = 1.33,
):
    """
    Build a (pred, target) -> scalar loss callable.

    "bce_dice"      — legacy Dice + weighted BCE (v8–v12 path, uses pos_weight).
    "focal_tversky" — Focal Tversky + 0.5 * balanced BCE stabiliser. The
                      alpha/beta ratio owns the class-balance lever, so pos_weight
                      is intentionally ignored (kept at 1.0 inside the BCE term).
    """
    if loss_name == "bce_dice":
        return lambda pred, target: combined_loss(pred, target, pos_weight=pos_weight)
    if loss_name == "focal_tversky":
        def _loss(pred, target):
            ftl = focal_tversky_loss(
                pred, target,
                alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma,
            )
            bce = nn.functional.binary_cross_entropy(pred, target)
            return ftl + 0.5 * bce
        return _loss
    raise ValueError(f"Unknown loss_name: {loss_name!r}")


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


def seg_counts(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> tuple[float, float, float]:
    """Hard-thresholded (TP, FP, FN) voxel counts for one batch."""
    pred_bin = (pred >= threshold).float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1.0 - target)).sum()
    fn = ((1.0 - pred_bin) * target).sum()
    return tp.item(), fp.item(), fn.item()


# ── Training step ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ── Validation step ────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn,
    threshold: float = 0.5,
) -> tuple[float, float, float, float]:
    """Returns (mean_val_loss, f1, precision, recall).

    F1/precision/recall are MICRO-aggregated: TP/FP/FN are summed over every
    val voxel and the ratios computed once at the end. This deliberately
    replaces the old per-patch mean Dice, which returned 1.0 for every
    empty-vs-empty patch and so inflated the score with phantom perfect hits.
    Under micro-aggregation an all-zeros collapse scores F1 = 0 (TP = 0),
    so the checkpoint selector can no longer be gamed by predicting nothing.

    F1 == Dice for binary segmentation; we report precision and recall
    separately so FP-domination (high recall, low precision) is visible
    per epoch instead of only at OOF time. val_loss still uses loss_fn so it
    stays comparable to train_loss and drives the scheduler."""
    model.eval()
    total_loss = 0.0
    tp = fp = fn = 0.0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="  val  ", leave=False):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += loss_fn(pred, y).item()
            b_tp, b_fp, b_fn = seg_counts(pred, y, threshold)
            tp += b_tp
            fp += b_fp
            fn += b_fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    return total_loss / len(loader), f1, precision, recall


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
    f1_min_epoch: int = 0,
    loss_name: str = "bce_dice",
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    tversky_gamma: float = 1.33,
) -> dict:
    """
    Train model, save two checkpoints independently:
      save_path        — best val_loss  (e.g. v14.pth)
      save_path + _f1  — best val_f1    (e.g. v14_f1.pth)

    The secondary checkpoint is selected on the micro-aggregated val F1 from
    evaluate() (= Dice, but immune to the empty-patch inflation that the old
    per-patch mean Dice suffered). Evaluate it OOF with
    `evaluate_cv.py --suffix _f1`.

    Scheduler always steps on val_loss (more stable than the noisy patch F1).

    pos_weight: upweight lesion voxels in BCE (1.0 = standard, 50.0 = lesion 50× heavier).

    resume_path: if provided, loads model + optimizer + scheduler + history from
                 this checkpoint and continues training from the next epoch.

    Returns history dict with lists: train_loss, val_loss, val_dice.
    """
    loss_fn = make_loss(
        loss_name=loss_name,
        pos_weight=pos_weight,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        tversky_gamma=tversky_gamma,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Always step scheduler on loss (more stable than noisy patch-based dice)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # F1 checkpoint path: v14.pth → v14_f1.pth
    _p = Path(save_path)
    f1_save_path = str(_p.parent / (_p.stem + "_f1" + _p.suffix))

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "val_f1": [], "val_precision": [], "val_recall": [],
    }
    best_loss = float("inf")
    best_f1 = 0.0
    start_epoch = 1

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        history = ckpt.get("history", history)
        best_loss = ckpt.get("val_loss", float("inf"))
        best_f1 = ckpt.get("val_f1", 0.0)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {resume_path}  (epoch {ckpt['epoch']}, val_loss={best_loss:.4f}, val_f1={best_f1:.4f})")

    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss, val_f1, val_precision, val_recall = evaluate(model, val_loader, device, loss_fn)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)

        flag = ""

        def _ckpt() -> dict:
            return {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "val_f1": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_loss": val_loss,
            }

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(_ckpt(), save_path)
            flag += " ← best_loss"

        if val_f1 > best_f1 and epoch >= f1_min_epoch:
            best_f1 = val_f1
            torch.save(_ckpt(), f1_save_path)
            flag += " ← best_f1"

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"P={val_precision:.3f} R={val_recall:.3f}{flag}"
        )

    print(f"\nTraining complete.  best_loss={best_loss:.4f} ({save_path})  |  best_f1={best_f1:.4f} ({f1_save_path})")
    return history
