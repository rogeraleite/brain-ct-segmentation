"""
Evaluate segmentation checkpoints against local NIfTI masks.

Examples:
    PYTHONPATH=. python scripts/evaluate.py --model sw_v8
    PYTHONPATH=. python scripts/evaluate.py --model sw_v8 --max-cases 10
    PYTHONPATH=. python scripts/evaluate.py --model mresize --threshold 0.5
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.inference import compute_volume_ml, get_device, load_model
from api.inference_sw import load_model_sw
from src.data.loader import build_index, load_nifti
from src.inference.sliding_window import sliding_window_predict
from src.preprocessing.transforms import TARGET_SHAPE, preprocess, resize_mask


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "mresize": {
        "kind": "mresize",
        "checkpoint": "models/best_model_small3DUNet.pth",
        "threshold": 0.5,
        "description": "Global resize baseline, output 64x128x128.",
    },
    "sw": {
        "kind": "sw",
        "checkpoint": "models/best_model_slidingWindow.pth",
        "threshold": 0.5,
        "skull_strip": False,
        "skull_excl_mm": 3.0,
        "description": "Sliding-window native-resolution model.",
    },
    "sw_v5": {
        "kind": "sw",
        "checkpoint": "models/best_model_slidingWindow_v5.pth",
        "threshold": 0.5,
        "skull_strip": False,
        "skull_excl_mm": 3.0,
        "description": "Sliding-window v5.",
    },
    "sw_v6": {
        "kind": "sw",
        "checkpoint": "models/best_model_slidingWindow_v6.pth",
        "threshold": 0.3,
        "skull_strip": False,
        "skull_excl_mm": 3.0,
        "description": "Sliding-window v6.",
    },
    "sw_v7": {
        "kind": "sw",
        "checkpoint": "models/best_model_slidingWindow_v7.pth",
        "threshold": 0.3,
        "skull_strip": False,
        "skull_excl_mm": 3.0,
        "description": "Sliding-window v7.",
    },
    "sw_v8": {
        "kind": "sw",
        "checkpoint": "models/best_model_slidingWindow_v8_dice.pth",
        "threshold": 0.3,
        "skull_strip": True,
        "skull_excl_mm": 3.0,
        "description": "Sliding-window v8 dice checkpoint with skull stripping.",
    },
    "sw_v9": {
        "kind": "sw",
        "checkpoint": "models/best_model_slidingWindow_v9_dice.pth",
        "threshold": 0.3,
        "skull_strip": True,
        "skull_excl_mm": 0.0,
        "description": "Sliding-window v9: SAM skull masks at training, largest-CC at inference.",
    },
    "sw_v10": {
        "kind": "sw",
        "checkpoint": "models/best_model_slidingWindow_v10_dice.pth",
        "threshold": 0.3,
        "skull_strip": True,
        "skull_excl_mm": 0.0,
        "description": "Sliding-window v10: SAM skull masks + pos_weight=5 (down from 50).",
    },
    "sw_v11": {
        "kind": "sw",
        "checkpoint": "models/best_model_slidingWindow_v11.pth",
        "threshold": 0.3,
        "skull_strip": True,
        "skull_excl_mm": 0.0,
        "description": "Sliding-window v11: CC skull strip identical at train and inference, pos_weight=5.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate brain CT segmentation models")
    parser.add_argument("--data-root", default="data/raw", help="Root with images/ and masks/")
    parser.add_argument("--model", default="sw_v11", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--threshold", type=float, default=None, help="Override default threshold")
    parser.add_argument("--stride", type=int, default=64, help="Sliding-window stride")
    parser.add_argument("--max-cases", type=int, default=None, help="Evaluate only the first N cases")
    parser.add_argument("--out-dir", default="reports", help="Directory for CSV and Markdown reports")
    return parser.parse_args()


def confusion_counts(pred: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    return {
        "tp_voxels": int(np.logical_and(pred_b, target_b).sum()),
        "fp_voxels": int(np.logical_and(pred_b, ~target_b).sum()),
        "fn_voxels": int(np.logical_and(~pred_b, target_b).sum()),
        "tn_voxels": int(np.logical_and(~pred_b, ~target_b).sum()),
    }


def safe_div(num: float, den: float, empty_value: float = 1.0) -> float:
    if den == 0:
        return empty_value
    return num / den


def segmentation_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    counts = confusion_counts(pred, target)
    tp = counts["tp_voxels"]
    fp = counts["fp_voxels"]
    fn = counts["fn_voxels"]
    tn = counts["tn_voxels"]

    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn

    metrics: dict[str, float | int] = {
        **counts,
        "dice": safe_div(2 * tp, dice_den),
        "iou": safe_div(tp, iou_den),
        "precision": safe_div(tp, tp + fp, empty_value=1.0 if target.sum() == 0 else 0.0),
        "recall": safe_div(tp, tp + fn),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
    }
    return metrics


def load_checkpoint_metadata(path: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "epoch": checkpoint.get("epoch"),
        "val_dice": checkpoint.get("val_dice"),
        "val_loss": checkpoint.get("val_loss"),
    }


def predict_case(
    volume: np.ndarray,
    spacing: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any],
    threshold: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    if config["kind"] == "mresize":
        volume_pp, _ = preprocess(volume, np.zeros_like(volume, dtype=np.uint8))
        x = torch.from_numpy(volume_pp).unsqueeze(0).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            prob = model(x)[0, 0].cpu().numpy()
        pred = (prob >= threshold).astype(np.uint8)
        scale = np.array(TARGET_SHAPE, dtype=np.float32) / np.array(volume.shape, dtype=np.float32)
        eval_spacing = spacing / scale
        return pred, eval_spacing

    spacing_hw_mm = float(min(spacing[1], spacing[2]))
    pred = sliding_window_predict(
        volume=volume,
        model=model,
        device=device,
        stride=stride,
        threshold=threshold,
        spacing_hw_mm=spacing_hw_mm,
        skull_strip=bool(config.get("skull_strip", False)),
        skull_excl_mm=float(config.get("skull_excl_mm", 0.0)),
    )
    return pred, spacing


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = MODEL_CONFIGS[args.model].copy()
    checkpoint = args.checkpoint or config["checkpoint"]
    threshold = args.threshold if args.threshold is not None else float(config["threshold"])

    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = get_device()
    if config["kind"] == "mresize":
        model, device = load_model(checkpoint)
    else:
        model, device = load_model_sw(checkpoint)

    records = build_index(args.data_root)
    if args.max_cases is not None:
        records = records[: args.max_cases]

    rows: list[dict[str, Any]] = []
    for record in tqdm(records, desc=f"evaluating {args.model}"):
        volume, spacing = load_nifti(record["image"])
        target, _ = load_nifti(record["mask"])
        target = (target > 0.5).astype(np.uint8)

        pred, eval_spacing = predict_case(
            volume=volume,
            spacing=spacing,
            model=model,
            device=device,
            config=config,
            threshold=threshold,
            stride=args.stride,
        )

        if target.shape != pred.shape:
            target_eval = resize_mask(target, pred.shape)
        else:
            target_eval = target

        metrics = segmentation_metrics(pred, target_eval)
        true_volume_ml = compute_volume_ml(target_eval, eval_spacing)
        pred_volume_ml = compute_volume_ml(pred, eval_spacing)
        abs_volume_error_ml = abs(pred_volume_ml - true_volume_ml)

        rows.append(
            {
                "case_id": Path(record["image"]).name,
                "image_path": record["image"],
                "mask_path": record["mask"],
                "model": args.model,
                "checkpoint": checkpoint,
                "threshold": threshold,
                "stride": args.stride if config["kind"] == "sw" else "",
                "prediction_shape": "x".join(str(v) for v in pred.shape),
                "target_shape": "x".join(str(v) for v in target_eval.shape),
                **metrics,
                "true_volume_ml": round(true_volume_ml, 4),
                "pred_volume_ml": round(pred_volume_ml, 4),
                "abs_volume_error_ml": round(abs_volume_error_ml, 4),
            }
        )

    metadata = {
        "model": args.model,
        "checkpoint": checkpoint,
        "threshold": threshold,
        "stride": args.stride if config["kind"] == "sw" else None,
        "device": str(device),
        "description": config["description"],
        "checkpoint_metadata": load_checkpoint_metadata(checkpoint),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_cases": len(rows),
    }
    return rows, metadata


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def metric_summary(rows: list[dict[str, Any]], key: str) -> tuple[float, float, float, float]:
    values = [float(row[key]) for row in rows]
    return mean(values), median(values), min(values), max(values)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No cases._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            cells.append(fmt(value, 4) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(rows: list[dict[str, Any]], metadata: dict[str, Any], md_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["dice", "iou", "precision", "recall", "specificity", "abs_volume_error_ml"]
    summary_rows = []
    for metric in metrics:
        avg, med, low, high = metric_summary(rows, metric)
        summary_rows.append(
            {
                "metric": metric,
                "mean": avg,
                "median": med,
                "min": low,
                "max": high,
            }
        )

    worst_dice = sorted(rows, key=lambda r: float(r["dice"]))[:5]
    worst_recall = sorted(rows, key=lambda r: float(r["recall"]))[:5]
    worst_volume = sorted(rows, key=lambda r: float(r["abs_volume_error_ml"]), reverse=True)[:5]
    ckpt_meta = metadata["checkpoint_metadata"]

    content = f"""# Evaluation Report - {metadata["model"]}

Generated: `{metadata["generated_at"]}`

## Setup

| Field | Value |
|---|---|
| Model | `{metadata["model"]}` |
| Description | {metadata["description"]} |
| Checkpoint | `{metadata["checkpoint"]}` |
| Checkpoint epoch | {fmt(ckpt_meta.get("epoch"))} |
| Checkpoint val Dice | {fmt(ckpt_meta.get("val_dice"), 4)} |
| Checkpoint val loss | {fmt(ckpt_meta.get("val_loss"), 4)} |
| Threshold | {fmt(metadata["threshold"], 2)} |
| Stride | {fmt(metadata["stride"])} |
| Device | `{metadata["device"]}` |
| Cases | {metadata["num_cases"]} |

## Aggregate Metrics

{markdown_table(summary_rows, ["metric", "mean", "median", "min", "max"])}

## Worst Dice Cases

{markdown_table(worst_dice, ["case_id", "dice", "recall", "precision", "true_volume_ml", "pred_volume_ml", "abs_volume_error_ml"])}

## Lowest Recall Cases

{markdown_table(worst_recall, ["case_id", "dice", "recall", "precision", "fn_voxels", "true_volume_ml", "pred_volume_ml"])}

## Largest Volume Errors

{markdown_table(worst_volume, ["case_id", "dice", "recall", "precision", "true_volume_ml", "pred_volume_ml", "abs_volume_error_ml"])}

## Notes

- Dice and IoU measure mask overlap, but they can hide clinically important false negatives on small lesions.
- Recall/sensitivity is especially important for screening-style stroke workflows because missed lesions are costly.
- Specificity is voxel-level and often very high because background dominates CT volumes; interpret it alongside false-positive voxels and visual review.
- This is an internal validation report on local data, not evidence of clinical readiness.
"""
    md_path.write_text(content)


def main() -> None:
    args = parse_args()
    rows, metadata = evaluate(args)

    out_dir = Path(args.out_dir)
    csv_path = out_dir / f"evaluation_{args.model}.csv"
    md_path = out_dir / f"evaluation_{args.model}.md"

    write_csv(rows, csv_path)
    write_markdown(rows, metadata, md_path)

    dice_mean = metric_summary(rows, "dice")[0]
    recall_mean = metric_summary(rows, "recall")[0]
    volume_error_mean = metric_summary(rows, "abs_volume_error_ml")[0]
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(
        "Summary: "
        f"mean Dice={dice_mean:.3f}, "
        f"mean recall={recall_mean:.3f}, "
        f"mean abs volume error={volume_error_mean:.3f} mL"
    )


if __name__ == "__main__":
    main()
