import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional


def show_slices(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    title: str = "",
    cmap: str = "gray",
    alpha: float = 0.35,
) -> None:
    """
    Display axial, coronal, and sagittal center slices of a 3-D volume.
    Optionally overlay a binary mask (red, semi-transparent).

    Args:
        volume: (D, H, W) float array
        mask:   (D, H, W) binary uint8 array, or None
        title:  figure title
        cmap:   matplotlib colormap for the volume
        alpha:  mask overlay opacity
    """
    d, h, w = volume.shape
    slices = {
        "Axial (z={})".format(d // 2):    (volume[d // 2], mask[d // 2] if mask is not None else None),
        "Coronal (y={})".format(h // 2):  (volume[:, h // 2, :], mask[:, h // 2, :] if mask is not None else None),
        "Sagittal (x={})".format(w // 2): (volume[:, :, w // 2], mask[:, :, w // 2] if mask is not None else None),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if title:
        fig.suptitle(title, fontsize=14)

    for ax, (label, (img_slice, mask_slice)) in zip(axes, slices.items()):
        ax.imshow(img_slice, cmap=cmap, origin="lower")
        ax.set_title(label)
        ax.axis("off")

        if mask_slice is not None and mask_slice.any():
            overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
            overlay[mask_slice > 0] = [1.0, 0.0, 0.0, alpha]  # red
            ax.imshow(overlay, origin="lower")

    if mask is not None:
        patch = mpatches.Patch(color="red", alpha=alpha, label="Lesion mask")
        fig.legend(handles=[patch], loc="lower right")

    plt.tight_layout()
    plt.show()


def show_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_dices: list[float],
) -> None:
    """Plot loss and Dice curves from training history."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train loss")
    ax1.plot(epochs, val_losses, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_dices, color="green", label="Val Dice")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice Score")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    best_epoch = int(np.argmax(val_dices)) + 1
    best_dice = max(val_dices)
    ax2.axvline(best_epoch, color="red", linestyle="--", alpha=0.5,
                label=f"Best: {best_dice:.3f} @ epoch {best_epoch}")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def show_prediction(
    volume: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
) -> None:
    """
    Show a single axial slice with predicted mask overlay.
    Optionally also shows ground-truth mask for comparison.

    Green = prediction, Red = ground truth.
    """
    d = volume.shape[0]
    if slice_idx is None:
        # Pick the slice with the most predicted lesion voxels
        lesion_counts = pred_mask.sum(axis=(1, 2))
        slice_idx = int(np.argmax(lesion_counts)) if lesion_counts.max() > 0 else d // 2

    n_cols = 3 if true_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(volume[slice_idx], cmap="gray", origin="lower")
    axes[0].set_title(f"CT scan — axial slice {slice_idx}")
    axes[0].axis("off")

    axes[1].imshow(volume[slice_idx], cmap="gray", origin="lower")
    pred_overlay = np.zeros((*pred_mask[slice_idx].shape, 4), dtype=np.float32)
    pred_overlay[pred_mask[slice_idx] > 0] = [0.0, 1.0, 0.0, 0.4]
    axes[1].imshow(pred_overlay, origin="lower")
    axes[1].set_title("Prediction (green)")
    axes[1].axis("off")

    if true_mask is not None:
        axes[2].imshow(volume[slice_idx], cmap="gray", origin="lower")
        gt_overlay = np.zeros((*true_mask[slice_idx].shape, 4), dtype=np.float32)
        gt_overlay[true_mask[slice_idx] > 0] = [1.0, 0.0, 0.0, 0.4]
        axes[2].imshow(gt_overlay, origin="lower")
        axes[2].set_title("Ground truth (red)")
        axes[2].axis("off")

    plt.tight_layout()
    plt.show()
