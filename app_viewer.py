"""
M6 + M8 — Interactive Slice Viewer with Distance Measurement and 3D Viewer

Run:
    streamlit run app_viewer.py

Requires the FastAPI running (for lesion segmentation):
    PYTHONPATH=. uvicorn api.main:app --reload --port 8000
"""

import base64
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates
from scipy.ndimage import binary_closing, binary_fill_holes, zoom
from skimage.measure import marching_cubes, find_contours
from src.inference.sliding_window import _skull_exclusion_mask, _extracranial_mask, _SKULL_HU_THRESH

from src.data.loader import load_nifti
from src.preprocessing.transforms import BRAIN_HU_MIN, BRAIN_HU_MAX, apply_brain_window

# ── Constants ─────────────────────────────────────────────────────────────────

POINT_RADIUS = 4
POINT_COLOR_1 = (255, 80, 80)
POINT_COLOR_2 = (80, 200, 80)
LINE_COLOR = (255, 220, 50)
MASK_COLOR = (220, 50, 50)
MASK_ALPHA = 0.40
EXCL_COLOR = (220, 180, 50)
EXCL_CONTOUR_COLOR = (255, 220, 50)
EXCL_ALPHA = 0.40
INNERSKULL_CONTOUR_COLOR = (50, 200, 220)
GT_COLOR = (0, 200, 80)
GT_ALPHA = 0.40
CLICK_STEP = 8  # px between click-detection grid points

POLY_POINT_COLOR = (80, 180, 255)
POLY_LINE_COLOR = (80, 180, 255)
POLY_FILL_COLOR = (80, 180, 255)
POLY_FILL_ALPHA = 0.18
CLOSE_THRESHOLD_PX = 15  # native pixels — click this close to P1 to auto-close

MODEL_OPTIONS = {
    "v15 Detect-then-Segment — detection F1 0.83 · gates FP on healthy scans · ~20s": "/segment/sw_v15",
    "Sliding Window v8 — Dice 0.38 · ep132 · 3D skull excl · ~20s":             "/segment/sw_v8",
    "Sliding Window v7 — Dice ~0.30 · ep95 · ~20s":                             "/segment/sw_v7",
    "Sliding Window v6 — Dice ~0.28 · no-elastic · bone excl post-inf · ~20s":  "/segment/sw_v6",
    "Sliding Window v4 — Dice 0.34 · native (650×650) · ~20s":                  "/segment/sw",
    "Sliding Window v5 — Dice 0.17 · + augment + bone suppression · ~20s":      "/segment/sw_v5",
    "Resize v1        — Dice 0.27 · 64×128×128        · ~3s":                   "/segment/mresize",
}

MODEL_DEFAULT_THRESHOLD = {
    "/segment/sw_v15":  0.30,
    "/segment/sw_v8":   0.30,
    "/segment/sw_v7":   0.30,
    "/segment/sw_v6":   0.30,
    "/segment/sw":      0.50,
    "/segment/sw_v5":   0.50,
    "/segment/mresize": 0.50,
}

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
MASKS_DIR = os.path.join(_APP_DIR, "data", "raw", "masks")
REPORTS_DIR = os.path.join(_APP_DIR, "reports")

METRIC_HELP = {
    "dice": "Overlap between prediction and ground truth. 1.0 is perfect; 0.0 means no overlap.",
    "iou": "Intersection over Union. Similar to Dice, but stricter because the union is the denominator.",
    "precision": "Of all voxels predicted as lesion, the fraction that really were lesion. Low precision means many false positives.",
    "recall": "Of all true lesion voxels, the fraction the model found. Also called sensitivity; low recall means missed lesion voxels.",
    "sensitivity": "Same as recall: the fraction of true lesion voxels detected by the model.",
    "specificity": "Of all true background voxels, the fraction correctly left as background.",
    "true_volume_ml": "Ground-truth lesion volume in millilitres, computed from mask voxels and voxel spacing.",
    "pred_volume_ml": "Predicted lesion volume in millilitres, computed from predicted mask voxels and voxel spacing.",
    "abs_volume_error_ml": "Absolute difference between predicted lesion volume and ground-truth volume, in millilitres.",
    "fp_voxels": "False-positive voxels: background voxels incorrectly predicted as lesion.",
    "fn_voxels": "False-negative voxels: lesion voxels missed by the model.",
}

METRIC_LABELS = {
    "dice": "Dice ⓘ",
    "iou": "IoU ⓘ",
    "precision": "Precision ⓘ",
    "recall": "Recall ⓘ",
    "sensitivity": "Sensitivity ⓘ",
    "specificity": "Specificity ⓘ",
    "true_volume_ml": "True Volume ⓘ",
    "pred_volume_ml": "Pred Volume ⓘ",
    "abs_volume_error_ml": "Volume Error ⓘ",
    "fp_voxels": "FP Voxels ⓘ",
    "fn_voxels": "FN Voxels ⓘ",
}

# HU thresholds for direct CT-based 3D reconstruction
SKULL_EXCL_MM = 3.0      # default skull exclusion margin in mm — measured from inner skull surface
DISPLAY_HW    = 450      # max display size (px) for the 2D slice viewer
SKULL_HU_THRESH = 400    # cortical bone (compact: 700–2000+, spongy: 400–700)
BRAIN_HU_LO     = 20     # brain parenchyma lower bound (white matter ~20–30)
BRAIN_HU_HI     = 80     # brain parenchyma upper bound (grey matter ~35–45, acute blood 50–90)
HEAD_OUTLINE_HU  = -200  # non-air threshold: captures scalp, skull, brain

STRUCTURES = {
    "brain":      {"label": "Brain",      "color": "#E8A4A4", "opacity": 0.40},
    "skull":      {"label": "Skull",      "color": "#F5DEB3", "opacity": 0.20},
    "cerebellum": {"label": "Cerebellum", "color": "#90CAF9", "opacity": 0.70},
    "brainstem":  {"label": "Brainstem",  "color": "#A5D6A7", "opacity": 0.80},
}

STRUCTURE_FILENAMES = {
    "brain":      ["brain.nii.gz"],
    "skull":      ["skull.nii.gz", "skull_3d.nii.gz"],
    "cerebellum": ["cerebellum.nii.gz"],
    "brainstem":  ["brainstem.nii.gz"],
}


# ── Logging ────────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO") -> None:
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["logs"].append(f"[{ts}] [{level}] {msg}")


# ── Cached data loaders ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading volume...")
def load_volume(file_bytes: bytes, filename: str) -> tuple[np.ndarray, np.ndarray]:
    suffix = ".nii.gz" if filename.endswith(".gz") else ".nii"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, file_bytes)
        os.close(tmp_fd)
        volume, spacing = load_nifti(tmp_path)
    finally:
        os.unlink(tmp_path)
    return volume, spacing


@st.cache_data(show_spinner="Running segmentation...")
def call_segment_api(
    file_bytes: bytes, filename: str, api_url: str, endpoint: str,
    threshold: float = 0.5, case_threshold: float = 0.5,
) -> dict:
    response = requests.post(
        f"{api_url}{endpoint}",
        files={"file": (filename, file_bytes, "application/octet-stream")},
        params={"threshold": threshold, "case_threshold": case_threshold},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


# ── 2D rendering helpers ───────────────────────────────────────────────────────

def slice_to_uint8(volume: np.ndarray, slice_idx: int) -> np.ndarray:
    slc = apply_brain_window(volume[slice_idx].copy())
    slc = (slc - BRAIN_HU_MIN) / (BRAIN_HU_MAX - BRAIN_HU_MIN)
    return (np.clip(slc, 0, 1) * 255).astype(np.uint8)


def build_frame(
    volume: np.ndarray,
    slice_idx: int,
    mask_full: np.ndarray | None,
    show_mask: bool,
    points: list[dict],
    spacing: np.ndarray | None = None,
    excl_mask: np.ndarray | None = None,
    show_excl: bool = False,
    poly_points: list[dict] | None = None,
    poly_closed: bool = False,
    gt_mask: np.ndarray | None = None,
    show_gt: bool = False,
) -> Image.Image:
    arr = slice_to_uint8(volume, slice_idx)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    rgb = np.array(img, dtype=np.float32)

    # All numpy blending in one pass (order: model pred → GT → exclusion zone)
    if show_mask and mask_full is not None:
        m = mask_full[slice_idx].astype(bool)
        if m.any():
            overlay = np.zeros_like(rgb)
            overlay[m] = MASK_COLOR
            rgb = rgb * (1 - MASK_ALPHA) + overlay * MASK_ALPHA

    if show_gt and gt_mask is not None:
        gm = gt_mask[slice_idx].astype(bool)
        if gm.any():
            gt_overlay = np.zeros_like(rgb)
            gt_overlay[gm] = GT_COLOR
            rgb = rgb * (1 - GT_ALPHA) + gt_overlay * GT_ALPHA

    if show_excl and excl_mask is not None:
        excl_slice = excl_mask[slice_idx].astype(bool)
        if excl_slice.any():
            excl_overlay = np.zeros_like(rgb)
            excl_overlay[excl_slice] = EXCL_COLOR
            rgb = rgb * (1 - EXCL_ALPHA) + excl_overlay * EXCL_ALPHA

    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

    # Polygon fill (RGBA composite)
    if poly_points and poly_closed and len(poly_points) >= 3:
        pts_fill = [(p["x"], p["y"]) for p in poly_points]
        fill_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        fill_draw = ImageDraw.Draw(fill_layer)
        fill_draw.polygon(pts_fill, fill=(*POLY_FILL_COLOR, int(255 * POLY_FILL_ALPHA)))
        img = Image.alpha_composite(img.convert("RGBA"), fill_layer).convert("RGB")

    draw = ImageDraw.Draw(img)

    # Exclusion zone inner-boundary contour line
    if show_excl and excl_mask is not None:
        excl_slice = excl_mask[slice_idx].astype(bool)
        if excl_slice.any():
            for contour in find_contours(excl_slice.astype(float), 0.5):
                pts = [(int(c[1]), int(c[0])) for c in contour]
                for i in range(len(pts) - 1):
                    draw.line([pts[i], pts[i + 1]], fill=EXCL_CONTOUR_COLOR, width=2)

    for i, pt in enumerate(points):
        x, y = pt["x"], pt["y"]
        color = POINT_COLOR_1 if i == 0 else POINT_COLOR_2
        r = POINT_RADIUS
        draw.rectangle([x - r, y - r, x + r, y + r], fill=color, outline="white", width=2)
        draw.text((x + r + 3, y - r), f"P{i+1}", fill=color)

    if len(points) == 2:
        x1, y1 = points[0]["x"], points[0]["y"]
        x2, y2 = points[1]["x"], points[1]["y"]
        draw.line([x1, y1, x2, y2], fill=LINE_COLOR, width=2)
        if spacing is not None:
            dist = compute_distance_mm(points[0], points[1], spacing)
            font = ImageFont.load_default(size=13)
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            draw.text((mx + 4, my - 16), f"{dist:.1f} mm", fill=LINE_COLOR, font=font)

    # Polygon edges and vertex markers
    if poly_points:
        pts_tuples = [(p["x"], p["y"]) for p in poly_points]
        for i in range(len(pts_tuples) - 1):
            draw.line([pts_tuples[i], pts_tuples[i + 1]], fill=POLY_LINE_COLOR, width=2)
        if poly_closed and len(poly_points) >= 3:
            draw.line([pts_tuples[-1], pts_tuples[0]], fill=POLY_LINE_COLOR, width=2)
        r = 4
        for i, pt in enumerate(poly_points):
            x, y = pt["x"], pt["y"]
            color = (255, 165, 0) if i == 0 else POLY_POINT_COLOR  # orange = P1 (close target)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white", width=1)

    return img


def build_plotly_2d(
    volume: np.ndarray,
    slice_idx: int,
    mask_full: np.ndarray | None,
    show_mask: bool,
    points: list[dict],
    spacing: np.ndarray | None = None,
    excl_mask: np.ndarray | None = None,
    show_excl: bool = False,
    poly_points: list[dict] | None = None,
    poly_closed: bool = False,
    zoom: int = 1,
    dragmode: str = "pan",
    gt_mask: np.ndarray | None = None,
    show_gt: bool = False,
    innerskull_mask: np.ndarray | None = None,
    show_innerskull: bool = False,
) -> go.Figure:
    arr = slice_to_uint8(volume, slice_idx)
    rgb = np.array(Image.fromarray(arr, mode="L").convert("RGB"), dtype=np.float32)

    if show_mask and mask_full is not None:
        m = mask_full[slice_idx].astype(bool)
        if m.any():
            overlay = np.zeros_like(rgb)
            overlay[m] = MASK_COLOR
            rgb = rgb * (1 - MASK_ALPHA) + overlay * MASK_ALPHA

    if show_gt and gt_mask is not None:
        gm = gt_mask[slice_idx].astype(bool)
        if gm.any():
            gt_overlay = np.zeros_like(rgb)
            gt_overlay[gm] = GT_COLOR
            rgb = rgb * (1 - GT_ALPHA) + gt_overlay * GT_ALPHA

    if show_excl and excl_mask is not None:
        excl_slice = excl_mask[slice_idx].astype(bool)
        if excl_slice.any():
            excl_overlay = np.zeros_like(rgb)
            excl_overlay[excl_slice] = EXCL_COLOR
            rgb = rgb * (1 - EXCL_ALPHA) + excl_overlay * EXCL_ALPHA

    h, w = arr.shape
    fig = go.Figure()
    fig.add_trace(go.Image(z=rgb.astype(np.uint8), hoverinfo="skip"))

    if len(points) == 2:
        x1, y1 = points[0]["x"], points[0]["y"]
        x2, y2 = points[1]["x"], points[1]["y"]
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2], mode="lines",
            line=dict(color=f"rgb{LINE_COLOR}", width=2),
            showlegend=False, hoverinfo="skip",
        ))
        if spacing is not None:
            dist = compute_distance_mm(points[0], points[1], spacing)
            fig.add_annotation(
                x=(x1 + x2) / 2, y=(y1 + y2) / 2,
                text=f"{dist:.1f} mm", showarrow=False,
                font=dict(color=f"rgb{LINE_COLOR}", size=13),
                xshift=4, yshift=8,
            )
    for i, pt in enumerate(points):
        color = f"rgb{POINT_COLOR_1}" if i == 0 else f"rgb{POINT_COLOR_2}"
        fig.add_trace(go.Scatter(
            x=[pt["x"]], y=[pt["y"]], mode="markers+text",
            marker=dict(color=color, size=10, line=dict(color="white", width=2)),
            text=[f"P{i+1}"], textposition="top right",
            textfont=dict(color=color),
            showlegend=False, hoverinfo="skip",
        ))

    if poly_points:
        xs_p = [p["x"] for p in poly_points]
        ys_p = [p["y"] for p in poly_points]
        if poly_closed and len(poly_points) >= 3:
            xs_p = xs_p + [xs_p[0]]
            ys_p = ys_p + [ys_p[0]]
        fig.add_trace(go.Scatter(
            x=xs_p, y=ys_p, mode="lines",
            fill="toself" if poly_closed else "none",
            fillcolor=f"rgba({POLY_FILL_COLOR[0]},{POLY_FILL_COLOR[1]},{POLY_FILL_COLOR[2]},{POLY_FILL_ALPHA})",
            line=dict(color=f"rgb{POLY_LINE_COLOR}", width=2),
            showlegend=False, hoverinfo="skip",
        ))
        for i, pt in enumerate(poly_points):
            marker_color = "rgb(255,165,0)" if i == 0 else f"rgb{POLY_POINT_COLOR}"
            fig.add_trace(go.Scatter(
                x=[pt["x"]], y=[pt["y"]], mode="markers",
                marker=dict(color=marker_color, size=8, line=dict(color="white", width=1)),
                showlegend=False, hoverinfo="skip",
            ))

    # Exclusion zone contour line at inner boundary
    if show_excl and excl_mask is not None:
        excl_slice = excl_mask[slice_idx].astype(bool)
        if excl_slice.any():
            for contour in find_contours(excl_slice.astype(float), 0.5):
                fig.add_trace(go.Scatter(
                    x=contour[:, 1].tolist(), y=contour[:, 0].tolist(),
                    mode="lines",
                    line=dict(color=f"rgb{EXCL_CONTOUR_COLOR}", width=2),
                    showlegend=False, hoverinfo="skip",
                ))

    # Inner skull (intracranial boundary) contour
    if show_innerskull and innerskull_mask is not None:
        is_slice = innerskull_mask[slice_idx].astype(bool)
        if is_slice.any():
            for contour in find_contours(is_slice.astype(float), 0.5):
                fig.add_trace(go.Scatter(
                    x=contour[:, 1].tolist(), y=contour[:, 0].tolist(),
                    mode="lines",
                    line=dict(color=f"rgb{INNERSKULL_CONTOUR_COLOR}", width=2, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                    name="Inner skull",
                ))

    # Near-invisible grid — opacity>0 keeps SVG pointer-events active for click detection
    xs_g = list(range(0, w, CLICK_STEP))
    ys_g = list(range(0, h, CLICK_STEP))
    XX, YY = np.meshgrid(xs_g, ys_g)
    fig.add_trace(go.Scatter(
        x=XX.flatten().tolist(), y=YY.flatten().tolist(),
        mode="markers",
        marker=dict(size=16, opacity=0.01, color="rgba(128,128,128,0.01)"),
        showlegend=False, name="_click", hoverinfo="none",
        selected=dict(marker=dict(opacity=0.01)),
        unselected=dict(marker=dict(opacity=0.01)),
    ))

    if zoom > 1:
        cx, cy = w / 2, h / 2
        hw = w / (2 * zoom)
        hh = h / (2 * zoom)
        xrange = [cx - hw, cx + hw]
        yrange = [cy + hh, cy - hh]
    else:
        xrange = [-0.5, w - 0.5]
        yrange = [h - 0.5, -0.5]

    fig.update_layout(
        uirevision=f"{zoom}_{dragmode}",
        dragmode=dragmode,
        clickmode="event+select",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="black",
        height=DISPLAY_HW,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=xrange),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=yrange, scaleanchor="x"),
    )
    return fig


def compute_distance_mm(p1: dict, p2: dict, spacing: np.ndarray) -> float:
    dx = (p2["x"] - p1["x"]) * spacing[2]
    dy = (p2["y"] - p1["y"]) * spacing[1]
    return math.sqrt(dx**2 + dy**2)


def compute_area_mm2(mask_full: np.ndarray, slice_idx: int, spacing: np.ndarray) -> float:
    return float(mask_full[slice_idx].sum()) * spacing[1] * spacing[2]


def compute_polygon_area_mm2(poly_pts: list[dict], spacing: np.ndarray) -> float:
    """Shoelace formula on polygon vertices, result in mm²."""
    n = len(poly_pts)
    if n < 3:
        return 0.0
    xs = [p["x"] * float(spacing[2]) for p in poly_pts]
    ys = [p["y"] * float(spacing[1]) for p in poly_pts]
    area = sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))
    return abs(area) / 2.0


def evaluation_column_config(columns: list[str]) -> dict:
    """Streamlit column labels + hover help for evaluation metrics."""
    config = {}
    for col in columns:
        if col not in METRIC_HELP:
            continue
        if col.endswith("_voxels"):
            config[col] = st.column_config.NumberColumn(
                METRIC_LABELS[col],
                help=METRIC_HELP[col],
                format="%d",
            )
        elif col.endswith("_ml"):
            config[col] = st.column_config.NumberColumn(
                METRIC_LABELS[col],
                help=METRIC_HELP[col],
                format="%.2f",
            )
        else:
            config[col] = st.column_config.NumberColumn(
                METRIC_LABELS[col],
                help=METRIC_HELP[col],
                format="%.3f",
            )
    return config


def render_evaluation_report() -> None:
    """Show the latest offline evaluation report generated by scripts/evaluate.py."""
    st.subheader("Evaluation Report")

    report_files = sorted(
        [f for f in os.listdir(REPORTS_DIR)] if os.path.isdir(REPORTS_DIR) else [],
        reverse=True,
    )
    csv_files = [f for f in report_files if f.startswith("evaluation_") and f.endswith(".csv")]

    if not csv_files:
        st.info(
            "No evaluation report found yet. Generate one with:\n\n"
            "```bash\n"
            "PYTHONPATH=. python scripts/evaluate.py --model sw_v8\n"
            "```"
        )
        return

    selected_csv = st.selectbox("Report", csv_files)
    csv_path = os.path.join(REPORTS_DIR, selected_csv)
    md_path = csv_path[:-4] + ".md"

    df = pd.read_csv(csv_path)
    if df.empty:
        st.warning("The selected report is empty.")
        return

    mean_dice = float(df["dice"].mean())
    mean_recall = float(df["recall"].mean())
    mean_precision = float(df["precision"].mean())
    mean_volume_error = float(df["abs_volume_error_ml"].mean())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean Dice ⓘ", f"{mean_dice:.3f}", help=METRIC_HELP["dice"])
    m2.metric("Mean Recall ⓘ", f"{mean_recall:.3f}", help=METRIC_HELP["recall"])
    m3.metric("Mean Precision ⓘ", f"{mean_precision:.3f}", help=METRIC_HELP["precision"])
    m4.metric(
        "Mean Volume Error ⓘ",
        f"{mean_volume_error:.2f} mL",
        help=METRIC_HELP["abs_volume_error_ml"],
    )

    with st.expander("Metric quick guide"):
        guide_rows = [
            {"Metric": METRIC_LABELS[key], "Meaning": METRIC_HELP[key]}
            for key in [
                "dice", "iou", "precision", "recall", "specificity",
                "abs_volume_error_ml", "fp_voxels", "fn_voxels",
            ]
        ]
        st.dataframe(guide_rows, use_container_width=True, hide_index=True)

    if os.path.exists(md_path):
        with st.expander("Summary", expanded=True):
            with open(md_path, "r") as f:
                st.markdown(f.read())

    st.divider()
    st.caption(f"{len(df)} cases · `{selected_csv}`")

    cols = [
        "case_id", "dice", "iou", "precision", "recall", "specificity",
        "true_volume_ml", "pred_volume_ml", "abs_volume_error_ml",
        "fp_voxels", "fn_voxels",
    ]
    visible_cols = [c for c in cols if c in df.columns]
    st.dataframe(
        df[visible_cols].sort_values("dice", ascending=True),
        use_container_width=True,
        hide_index=True,
        column_config=evaluation_column_config(visible_cols),
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Worst Dice**")
        st.dataframe(
            df.nsmallest(5, "dice")[["case_id", "dice", "recall", "precision"]],
            use_container_width=True,
            hide_index=True,
            column_config=evaluation_column_config(["dice", "recall", "precision"]),
        )
    with c2:
        st.markdown("**Lowest Recall**")
        st.dataframe(
            df.nsmallest(5, "recall")[["case_id", "dice", "recall", "fn_voxels"]],
            use_container_width=True,
            hide_index=True,
            column_config=evaluation_column_config(["dice", "recall", "fn_voxels"]),
        )
    with c3:
        st.markdown("**Largest Volume Error**")
        st.dataframe(
            df.nlargest(5, "abs_volume_error_ml")[
                ["case_id", "true_volume_ml", "pred_volume_ml", "abs_volume_error_ml"]
            ],
            use_container_width=True,
            hide_index=True,
            column_config=evaluation_column_config(
                ["true_volume_ml", "pred_volume_ml", "abs_volume_error_ml"]
            ),
        )


def decode_mask(result: dict, volume_shape: tuple) -> np.ndarray:
    mask_bytes = base64.b64decode(result["mask_base64"])
    mask_pred = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(result["mask_shape"])
    factors = [o / m for o, m in zip(volume_shape, mask_pred.shape)]
    mask_full = zoom(mask_pred.astype(float), factors, order=0)
    return (mask_full > 0.5).astype(np.uint8)


# ── 3D helpers ─────────────────────────────────────────────────────────────────

def ts_installed() -> tuple[bool, str]:
    """Returns (is_installed, version_or_error)."""
    try:
        from importlib.metadata import version
        return True, version("TotalSegmentator")
    except Exception:
        return False, "not installed"


def ts_device() -> str:
    """Pick best available device: mps (Apple Silicon) > cpu."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def ts_weights_exist() -> bool:
    weights_dir = os.path.expanduser("~/.totalsegmentator/nnunet/results/")
    return os.path.isdir(weights_dir) and len(os.listdir(weights_dir)) > 0


def run_totalsegmentator(file_bytes: bytes, filename: str) -> dict[str, np.ndarray]:
    """Run TotalSegmentator via subprocess (captures stdout/stderr for logs)."""
    suffix = ".nii.gz" if filename.endswith(".gz") else ".nii"
    output_dir = tempfile.mkdtemp()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, file_bytes)
        os.close(tmp_fd)

        log(f"Input:  {tmp_path}")
        log(f"Output: {output_dir}")
        log(f"Python: {sys.executable}")

        device = ts_device()
        log(f"Device: {device}")

        # Use subprocess so stdout/stderr are fully captured
        code = (
            "from totalsegmentator.python_api import totalsegmentator; "
            f"totalsegmentator({json.dumps(tmp_path)}, {json.dumps(output_dir)}, "
            f"fast=True, device={json.dumps(device)})"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=600,
        )

        if proc.stdout.strip():
            for line in proc.stdout.strip().splitlines():
                log(f"TS stdout: {line}")
        if proc.stderr.strip():
            for line in proc.stderr.strip().splitlines():
                log(f"TS stderr: {line}", level="WARN")

        if proc.returncode != 0:
            raise RuntimeError(
                f"TotalSegmentator exited with code {proc.returncode}. "
                f"See the Logs tab for details."
            )

        masks = {}
        for name, candidates in STRUCTURE_FILENAMES.items():
            for fname in candidates:
                fpath = os.path.join(output_dir, fname)
                if os.path.exists(fpath):
                    arr, _ = load_nifti(fpath)
                    masks[name] = (arr > 0.5).astype(np.uint8)
                    log(f"Structure loaded: {name} ({int(arr.sum())} voxels)")
                    break
            else:
                log(f"Structure not found: {name}", level="WARN")

        return masks

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        shutil.rmtree(output_dir, ignore_errors=True)


def compute_mesh(
    mask: np.ndarray, spacing: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    if mask.sum() == 0:
        return None
    try:
        verts, faces, _, _ = marching_cubes(
            mask, level=0.5, spacing=tuple(spacing), step_size=2
        )
        return verts, faces
    except Exception as e:
        log(f"marching_cubes failed: {e}", level="ERROR")
        return None


def compute_full_head_mesh(
    masks: dict[str, np.ndarray], spacing: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Union of all TS masks → single closed-surface mesh covering the full head."""
    if not masks:
        return None
    combined = None
    for mask in masks.values():
        combined = mask if combined is None else np.maximum(combined, mask)
    if combined is None or combined.sum() == 0:
        return None
    combined = binary_closing(combined, iterations=3).astype(np.uint8)
    try:
        verts, faces, _, _ = marching_cubes(
            combined, level=0.5, spacing=tuple(spacing), step_size=3
        )
        return verts, faces
    except Exception as e:
        log(f"Full-head marching_cubes failed: {e}", level="ERROR")
        return None


def compute_hu_meshes(volume: np.ndarray, spacing: np.ndarray) -> dict:
    """Build skull + brain meshes directly from CT HU thresholds.

    Skull: cortical bone (HU > 400).
    Brain: parenchyma inside the head contour (HU 20–80).
    Downsample H×W by 2 for performance before meshing.
    """
    vol_ds = volume[:, ::2, ::2]
    sp_ds = np.array([spacing[0], spacing[1] * 2.0, spacing[2] * 2.0])

    # Skull mask: bone HU + 2D closing per slice to patch small holes
    skull_raw = (vol_ds > SKULL_HU_THRESH).astype(np.uint8)
    skull_closed = np.zeros_like(skull_raw)
    for i in range(skull_raw.shape[0]):
        skull_closed[i] = binary_closing(skull_raw[i], iterations=1).astype(np.uint8)

    # Head outline: fill each axial slice to get the enclosed head region
    head_outer = vol_ds > HEAD_OUTLINE_HU
    head_filled = np.zeros_like(head_outer, dtype=bool)
    for i in range(head_outer.shape[0]):
        head_filled[i] = binary_fill_holes(head_outer[i])

    # Brain: enclosed non-bone tissue within brain HU range
    brain_raw = (
        head_filled &
        ~skull_closed.astype(bool) &
        (vol_ds >= BRAIN_HU_LO) &
        (vol_ds <= BRAIN_HU_HI)
    ).astype(np.uint8)
    brain_closed = np.zeros_like(brain_raw)
    for i in range(brain_raw.shape[0]):
        brain_closed[i] = binary_closing(brain_raw[i], iterations=2).astype(np.uint8)

    log(f"HU meshes: skull={int(skull_closed.sum())} voxels, brain={int(brain_closed.sum())} voxels")
    result = {}
    s = compute_mesh(skull_closed, sp_ds)
    if s:
        result["skull"] = s
    b = compute_mesh(brain_closed, sp_ds)
    if b:
        result["brain"] = b
    return result


def clip_mesh_at_z(
    verts: np.ndarray, faces: np.ndarray, z_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only faces where ALL 3 vertices have depth (verts[:,0]) >= z_mm.

    Creates the 'bowl' effect: shows the skull below the selected slice plane.
    """
    keep = (
        (verts[faces[:, 0], 0] >= z_mm) &
        (verts[faces[:, 1], 0] >= z_mm) &
        (verts[faces[:, 2], 0] >= z_mm)
    )
    return verts, faces[keep]


def build_3d_figure(
    hu_meshes: dict,
    slice_idx: int,
    spacing: np.ndarray,
    vol_shape: tuple,
    volume: np.ndarray | None = None,
    full_head_mesh: tuple | None = None,
    gt_mask: np.ndarray | None = None,
    show_gt: bool = False,
    lesion_mask: np.ndarray | None = None,
    show_lesion: bool = True,
) -> go.Figure:
    """Render the prediction (dark red), ground truth (green), and current CT slice plane."""
    fig = go.Figure()

    W_mm = float(vol_shape[2]) * spacing[2]
    H_mm = float(vol_shape[1]) * spacing[1]
    D_mm = float(vol_shape[0]) * spacing[0]
    z_mm = float(slice_idx) * spacing[0]

    # Current CT slice — actual scan image as a plane at the selected depth.
    if volume is not None:
        slc = volume[slice_idx].astype(np.float32)
        slc_win = np.clip(slc, BRAIN_HU_MIN, BRAIN_HU_MAX)
        slc_norm = (slc_win - BRAIN_HU_MIN) / (BRAIN_HU_MAX - BRAIN_HU_MIN)
        h, w = slc_norm.shape
        step_h = max(1, h // 150)
        step_w = max(1, w // 150)
        slc_ds = slc_norm[::step_h, ::step_w]
        Hd, Wd = slc_ds.shape
        xs = np.linspace(0.0, W_mm, Wd)
        ys = np.linspace(H_mm, 0.0, Hd)
        Z = np.full((Hd, Wd), z_mm)
        fig.add_trace(go.Surface(
            x=xs.tolist(), y=ys.tolist(), z=Z.tolist(),
            surfacecolor=slc_ds.tolist(),
            colorscale="Gray", showscale=False,
            opacity=1.0, name=f"Slice {slice_idx}", showlegend=True,
        ))

    # Model lesion prediction — dark red surface. Drawn FIRST so the green
    # ground truth (added after) renders on top and is never occluded by it.
    if show_lesion and lesion_mask is not None and lesion_mask.any():
        lesion_result = compute_mesh(lesion_mask, spacing)
        if lesion_result is not None:
            lv, lf = lesion_result
            fig.add_trace(go.Mesh3d(
                x=lv[:, 2].tolist(), y=(H_mm - lv[:, 1]).tolist(), z=lv[:, 0].tolist(),
                i=lf[:, 0].tolist(), j=lf[:, 1].tolist(), k=lf[:, 2].tolist(),
                color="#DC3232", opacity=0.85,
                name="Prediction", showlegend=True,
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
            ))

    # Ground truth lesion mesh — green surface, drawn LAST (on top).
    if show_gt and gt_mask is not None:
        gt_result = compute_mesh(gt_mask, spacing)
        if gt_result is not None:
            gv, gf = gt_result
            fig.add_trace(go.Mesh3d(
                x=gv[:, 2].tolist(), y=(H_mm - gv[:, 1]).tolist(), z=gv[:, 0].tolist(),
                i=gf[:, 0].tolist(), j=gf[:, 1].tolist(), k=gf[:, 2].tolist(),
                color="#00C850", opacity=0.70,
                name="Ground truth", showlegend=True,
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
            ))

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="W (mm)", range=[0, W_mm]),
            yaxis=dict(title="H (mm)", range=[0, H_mm]),
            zaxis=dict(title="D (mm)", range=[0, D_mm]),
            bgcolor="rgb(10,10,15)",
        ),
        paper_bgcolor="rgb(10,10,15)",
        plot_bgcolor="rgb(10,10,15)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        legend=dict(x=0, y=1, font=dict(size=11)),
    )
    return fig


# ── App ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Brain CT Viewer", layout="wide")
st.title("Brain CT — Slice Viewer")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    api_port = st.number_input("API port", min_value=1, max_value=65535, value=8000, step=1)
    api_url = f"http://localhost:{api_port}"
    st.caption(f"API: {api_url}")
    uploaded = st.file_uploader("CT Volume (.nii / .nii.gz)", type=["nii", "gz"])
    st.caption("Positives: 49–53, 58, 66–94, 97")

    volume = None
    spacing = None
    seg_result = None
    mask_full = None
    gt_mask_full = None
    show_area = False
    show_mask = False
    show_excl = False
    show_innerskull = False
    show_gt = False
    slice_idx = 0
    endpoint = list(MODEL_OPTIONS.values())[0]

    if uploaded:
        file_bytes = uploaded.read()
        filename = uploaded.name

        if st.session_state.get("ts_file") != filename:
            st.session_state.pop("ts_meshes", None)
            st.session_state.pop("ts_masks", None)
            st.session_state.pop("ts_full_head", None)
            st.session_state.pop("hu_meshes", None)
            st.session_state.pop("excl_mask", None)
            st.session_state.pop("excl_mask_key", None)
            st.session_state.pop("innerskull_mask", None)
            st.session_state["ts_file"] = filename

        try:
            volume, spacing = load_volume(file_bytes, filename)
            if st.session_state.get("_slice_vol") != filename:
                st.session_state["slice_idx"] = volume.shape[0] // 2
                st.session_state["_slice_vol"] = filename
        except Exception as e:
            log(f"Error loading volume: {e}", level="ERROR")
            st.error(f"Error loading volume: {e}")
            st.stop()

        # Auto-compute HU meshes once per file (skull + brain from CT directly)
        if "hu_meshes" not in st.session_state:
            with st.spinner("Computing 3D meshes (HU: skull + brain)..."):
                log("=== HU meshes started ===")
                st.session_state["hu_meshes"] = compute_hu_meshes(volume, spacing)
                log(f"=== HU meshes completed: {list(st.session_state['hu_meshes'].keys())} ===")

        # ── Ground truth mask ──────────────────────────────────────────────
        _gt_path = os.path.join(MASKS_DIR, filename)
        if os.path.exists(_gt_path):
            with open(_gt_path, "rb") as _f:
                _gt_bytes = _f.read()
            _gt_vol, _ = load_volume(_gt_bytes, filename)
            gt_mask_full = (_gt_vol > 0.5).astype(np.uint8)
            st.caption(f"Ground truth auto-loaded: {filename}")
        else:
            _uploaded_gt = st.file_uploader(
                "Ground truth mask (.nii / .nii.gz)",
                type=["nii", "gz"],
                key="gt_uploader",
            )
            if _uploaded_gt:
                _gt_bytes = _uploaded_gt.read()
                _gt_vol, _ = load_volume(_gt_bytes, _uploaded_gt.name)
                gt_mask_full = (_gt_vol > 0.5).astype(np.uint8)

        # ── Lesion segmentation (API) ──────────────────────────────────────
        st.divider()
        st.subheader("Lesion Segmentation")
        model_label = st.radio("Model", list(MODEL_OPTIONS.keys()))
        endpoint = MODEL_OPTIONS[model_label]

        with st.expander("Model comparison"):
            st.markdown(
                "| Model | Dice | Resolution | Speed | Notes |\n"
                "|---|---|---|---|---|\n"
                "| **SW v8** | 0.377 | 650×650 native | ~20s | ep132 · 3D skull excl · dual ckpt · patience=10 |\n"
                "| SW v7 | ~0.30 | 650×650 native | ~20s | ep95 |\n"
                "| SW v6 | ~0.28 | 650×650 native | ~20s | no elastic aug, bone excl post-inference only |\n"
                "| SW v4 | 0.339 | 650×650 native | ~20s | lesion-centered 70/30, pos_weight=50 |\n"
                "| SW v5 | 0.172 | 650×650 native | ~20s | + augment (rot/elastic/jitter) + bone suppression |\n"
                "| Resize v1 | 0.272 | 64×128×128 | ~3s | baseline |\n\n"
                "_Dice on validation set. "
                "SW v8 = epoch 132 (dice ckpt), SW v7 = epoch 95, SW v6 = patch-based estimate (ep78), SW v4 = epoch 79, SW v5 = epoch 74, Resize v1 = epoch 46._"
            )

        _model_default_thresh = MODEL_DEFAULT_THRESHOLD.get(endpoint, 0.50)
        if st.session_state.get("_thresh_endpoint") != endpoint:
            st.session_state["_thresh_val"] = _model_default_thresh
            st.session_state["_thresh_endpoint"] = endpoint
        elif "_thresh_val" not in st.session_state:
            st.session_state["_thresh_val"] = _model_default_thresh
        seg_threshold = st.slider(
            "Prediction threshold", 0.05, 0.95, step=0.05, key="_thresh_val",
            help="Probability cutoff for binarising model output. Lower values show more (weaker) predictions.",
        )

        is_v15 = endpoint == "/segment/sw_v15"
        case_threshold = 0.5
        if is_v15:
            case_threshold = st.slider(
                "Detector gate threshold (case-level)", 0.05, 0.95, step=0.05, value=0.5,
                key="_case_thresh_val",
                help="Stage-1 gate: if the scan's hemorrhage probability is below this, "
                     "segmentation is suppressed and the scan is reported lesion-free. "
                     "Lower = catch more (higher recall), fewer suppressions.",
            )

        if st.button("Run Segmentation", type="primary"):
            st.session_state.pop("seg_result", None)
            st.session_state.pop("seg_endpoint", None)
            st.session_state.pop("seg_threshold", None)
            st.session_state.pop("seg_case_threshold", None)
            call_segment_api.clear()

        _threshold_changed = (
            st.session_state.get("seg_threshold") != seg_threshold
            or st.session_state.get("seg_case_threshold") != case_threshold
        )
        if "seg_result" not in st.session_state or st.session_state.get("seg_endpoint") != endpoint or _threshold_changed:
            log(f"Calling API: {api_url}{endpoint} threshold={seg_threshold} case_threshold={case_threshold}")
            try:
                result = call_segment_api(file_bytes, filename, api_url, endpoint, seg_threshold, case_threshold)
                st.session_state["seg_result"] = result
                st.session_state["seg_endpoint"] = endpoint
                st.session_state["seg_threshold"] = seg_threshold
                st.session_state["seg_case_threshold"] = case_threshold
                log(f"API OK — lesion: {result['lesion_volume_ml']:.3f} mL · {result['hemisphere']}")
            except requests.exceptions.ConnectionError:
                msg = f"API not found at {api_url}"
                log(msg, level="ERROR")
                st.error(msg)
            except requests.exceptions.HTTPError as e:
                msg = f"API error ({e.response.status_code}): {e.response.text}"
                log(msg, level="ERROR")
                st.error(msg)
            except Exception as e:
                log(f"Segmentation error: {e}", level="ERROR")
                st.error(f"Segmentation error: {e}")

        if "seg_result" in st.session_state:
            seg_result = st.session_state["seg_result"]
            mask_full = decode_mask(seg_result, volume.shape)

            # ── v15 detect-then-segment banner ─────────────────────────────
            if "hemorrhage_detected" in seg_result:
                p = seg_result["case_probability"]
                ct = seg_result["case_threshold"]
                if seg_result["hemorrhage_detected"]:
                    st.error(
                        f"🔴 **Hemorrhage detected** — case probability "
                        f"**{p:.0%}** ≥ gate {ct:.0%}. Running segmentation.",
                        icon="🩸",
                    )
                else:
                    st.success(
                        f"🟢 **No hemorrhage detected** — case probability "
                        f"**{p:.0%}** < gate {ct:.0%}. Segmentation suppressed "
                        f"(scan reported lesion-free — no false positives invented).",
                        icon="✅",
                    )

            st.caption(
                f"Model: {seg_result.get('model_version', 'v1.0')} · "
                f"endpoint: `{endpoint}`"
            )

        # ── View controls ──────────────────────────────────────────────────
        st.divider()
        show_mask = st.checkbox("Show segmentation overlay", value=True)
        show_excl = st.checkbox("Show bone exclusion zone", value=True)
        show_innerskull = st.checkbox("Show inner skull boundary", value=True)
        skull_excl_mm = st.slider(
            "Skull exclusion margin (mm)", 0.0, 15.0, SKULL_EXCL_MM,
            step=0.5, disabled=not show_excl,
            help="Inward margin from the inner skull surface. Updates the overlay and lesion volume in real time — no re-run needed.",
        )
        show_gt = st.checkbox("Show ground truth", value=True, disabled=gt_mask_full is None)
        show_area = st.checkbox("Show lesion area (slice)", value=True)
        slice_idx = st.session_state.get("slice_idx", volume.shape[0] // 2)

        # Recompute exclusion mask whenever file or margin changes
        _excl_key = f"excl_{filename}_{skull_excl_mm}"
        if st.session_state.get("excl_mask_key") != _excl_key:
            spacing_hw_mm = float(min(spacing[1], spacing[2]))
            st.session_state["excl_mask"] = _skull_exclusion_mask(
                volume, _SKULL_HU_THRESH, skull_excl_mm, spacing_hw_mm,
            )
            st.session_state["excl_mask_key"] = _excl_key

        # Compute intracranial mask once per file (inverse of extracranial)
        if "innerskull_mask" not in st.session_state:
            st.session_state["innerskull_mask"] = ~_extracranial_mask(volume, _SKULL_HU_THRESH)
            st.session_state["bone_mask"] = (volume > _SKULL_HU_THRESH).astype(bool)

        # ── v15 per-slice detector confidence bar ──────────────────────────
        _seg = st.session_state.get("seg_result", {})
        if _seg.get("slice_probabilities"):
            sp = _seg["slice_probabilities"]
            ct = _seg.get("case_threshold", 0.5)
            det_colors = [
                "rgba(220,50,50,0.9)" if v >= ct else "rgba(80,140,220,0.55)"
                for v in sp
            ]
            fig_det = go.Figure()
            fig_det.add_bar(x=list(range(len(sp))), y=sp, marker_color=det_colors)
            fig_det.add_hline(y=ct, line_color="rgba(255,255,255,0.5)", line_width=1, line_dash="dot")
            fig_det.add_vline(x=slice_idx, line_color="yellow", line_width=1.5)
            fig_det.update_layout(
                height=64,
                margin=dict(l=0, r=0, t=2, b=2),
                showlegend=False,
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig_det, use_container_width=True,
                config={"displayModeBar": False}, key="det_sparkline",
            )
            st.caption(
                "Stage-1 detector confidence per slice · Red = above gate · "
                "Dotted line = gate threshold · Yellow = current slice"
            )

        if mask_full is not None:
            per_slice = mask_full.sum(axis=(1, 2))
            colors = [
                "rgba(220,50,50,0.85)" if v > 0 else "rgba(90,90,90,0.4)"
                for v in per_slice
            ]
            fig_sp = go.Figure()
            fig_sp.add_bar(
                x=list(range(len(per_slice))),
                y=per_slice.tolist(),
                marker_color=colors,
            )
            fig_sp.add_vline(x=slice_idx, line_color="yellow", line_width=1.5)
            fig_sp.update_layout(
                height=52,
                margin=dict(l=0, r=0, t=2, b=2),
                showlegend=False,
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig_sp, use_container_width=True,
                config={"displayModeBar": False}, key="sparkline",
            )
            st.caption("Red = slices with lesion · Yellow line = current slice")



        # ── TotalSegmentator 3D ────────────────────────────────────────────
        st.divider()
        st.subheader("3D Visualization")

        ts_ok, ts_ver = ts_installed()
        if ts_ok:
            weights_ok = ts_weights_exist()
            device = ts_device()
            st.caption(
                f"TotalSegmentator {ts_ver} · fast (3mm) · device: {device}  \n"
                f"Dataset297 · 1559 subjects"
            )
            if not weights_ok:
                st.warning(
                    "Model weights not found. "
                    "First run will download ~1.3 GB — may take 10-20 min.",
                    icon="⚠️",
                )
        else:
            st.error("TotalSegmentator not installed.\n```\npip install totalsegmentator\n```")

        if ts_ok and st.button("Run TotalSegmentator", type="secondary"):
            with st.spinner("Running TotalSegmentator (fast)… see the Logs tab for progress"):
                try:
                    log("=== TotalSegmentator started ===")
                    masks = run_totalsegmentator(file_bytes, filename)
                    meshes = {}
                    ts_stats = {}
                    for name, mask in masks.items():
                        result_mesh = compute_mesh(mask, spacing)
                        if result_mesh is not None:
                            meshes[name] = result_mesh
                        vox = int(mask.sum())
                        vol_ml = vox * float(spacing[0]) * float(spacing[1]) * float(spacing[2]) / 1000.0
                        ts_stats[name] = {"voxels": vox, "volume_ml": vol_ml}
                    full_head = compute_full_head_mesh(masks, spacing)
                    st.session_state["ts_meshes"] = meshes
                    st.session_state["ts_stats"] = ts_stats
                    st.session_state["ts_full_head"] = full_head
                    found = [STRUCTURES[n]["label"] for n in meshes]
                    log(f"=== TotalSegmentator completed: {', '.join(found)} ===")
                    st.success(f"Structures: {', '.join(found)}")
                except Exception as e:
                    log(f"TotalSegmentator failed: {e}", level="ERROR")
                    st.error(str(e))

        if "ts_stats" in st.session_state and st.session_state["ts_stats"]:
            st.caption("Volumes by structure (TS):")
            for name, cfg in STRUCTURES.items():
                stats = st.session_state["ts_stats"].get(name, {})
                if stats:
                    st.caption(f"  {cfg['label']}: {stats['volume_ml']:.0f} mL")

# ── Main area (tabs) ───────────────────────────────────────────────────────────
tab_viewer, tab_report, tab_logs = st.tabs(["Viewer", "Evaluation Report", "Logs"])

with tab_report:
    render_evaluation_report()

with tab_logs:
    logs = st.session_state.get("logs", [])
    col_l, col_r = st.columns([4, 1])
    col_l.caption(f"{len(logs)} entries")
    if col_r.button("Clear logs"):
        st.session_state["logs"] = []
        st.rerun()

    if logs:
        st.code("\n".join(reversed(logs)), language=None)
    else:
        st.info("No logs yet. Run a segmentation or TotalSegmentator.")

if volume is None:
    with tab_viewer:
        st.info("Upload a NIfTI file (.nii or .nii.gz) in the sidebar.")
    st.stop()

if "points" not in st.session_state:
    st.session_state["points"] = []
if "poly_points" not in st.session_state:
    st.session_state["poly_points"] = []
if "poly_closed" not in st.session_state:
    st.session_state["poly_closed"] = False

points = st.session_state["points"]

# ── Viewer tab ─────────────────────────────────────────────────────────────────
with tab_viewer:
    col_2d, col_3d = st.columns([2, 3])

    with col_2d:
        slice_idx = st.slider(
            "Axial slice", 0, volume.shape[0] - 1,
            st.session_state.get("slice_idx", volume.shape[0] // 2),
            key="slice_idx",
        )

        _viewer_mode = st.session_state.get("_viewer_mode", "Navigate")
        # Clear last-click dedup when mode changes so first click always registers
        if _viewer_mode != st.session_state.get("_prev_viewer_mode"):
            st.session_state["_prev_viewer_mode"] = _viewer_mode
            st.session_state.pop("_last_sel", None)
            st.session_state.pop("viewer_2d_sic", None)  # discard stale click from previous mode
        _poly_pts = st.session_state.get("poly_points", [])
        _poly_closed = st.session_state.get("poly_closed", False)

        # Streamlit stores the component value in session_state[key] before the rerun
        # fires, so we can process the click here — before build_frame() — and show
        # the updated overlay in the same render pass. No st.rerun() needed.
        _sic_click = st.session_state.get("viewer_2d_sic")
        if _sic_click is not None and _viewer_mode != "Navigate":
            _sel = (_sic_click["x"], _sic_click["y"])
            if _sel != st.session_state.get("_last_sel"):
                st.session_state["_last_sel"] = _sel
                _new_pt = {"x": _sic_click["x"], "y": _sic_click["y"]}
                if _viewer_mode == "Distance":
                    _d_pts = st.session_state.get("points", [])
                    if len(_d_pts) < 2:
                        _d_pts.append(_new_pt)
                    else:
                        _d_pts[0] = _d_pts[1]
                        _d_pts[1] = _new_pt
                    st.session_state["points"] = _d_pts
                elif _viewer_mode == "Area":
                    if not _poly_closed:
                        if len(_poly_pts) >= 3:
                            _first = _poly_pts[0]
                            _dist_px = math.sqrt(
                                (_new_pt["x"] - _first["x"]) ** 2 +
                                (_new_pt["y"] - _first["y"]) ** 2
                            )
                            if _dist_px <= CLOSE_THRESHOLD_PX:
                                st.session_state["poly_closed"] = True
                                _poly_closed = True
                            else:
                                _poly_pts.append(_new_pt)
                                st.session_state["poly_points"] = _poly_pts
                        else:
                            _poly_pts.append(_new_pt)
                            st.session_state["poly_points"] = _poly_pts
        points = st.session_state["points"]
        _poly_pts = st.session_state.get("poly_points", [])
        _poly_closed = st.session_state.get("poly_closed", False)

        _zoom = st.session_state.get("_zoom_slider", 1)
        _excl_arr = st.session_state.get("excl_mask")
        _bone_arr = st.session_state.get("bone_mask")
        # Visual overlay: fill the full bone ring (between yellow and blue lines) + inner margin
        if _excl_arr is not None and _bone_arr is not None:
            _excl_display = _excl_arr | _bone_arr
        else:
            _excl_display = _excl_arr
        if show_excl and mask_full is not None and _excl_arr is not None:
            mask_filtered = mask_full.copy()
            mask_filtered[_excl_display] = 0
        else:
            mask_filtered = mask_full

        if _viewer_mode == "Navigate":
            # Plotly viewer for pan/zoom only — no click detection needed here
            fig_2d = build_plotly_2d(
                volume, slice_idx, mask_filtered, show_mask,
                [], spacing,
                excl_mask=_excl_display, show_excl=show_excl,
                poly_points=None, poly_closed=False,
                zoom=_zoom, dragmode="pan",
                gt_mask=gt_mask_full, show_gt=show_gt,
                innerskull_mask=st.session_state.get("innerskull_mask"),
                show_innerskull=show_innerskull,
            )
            if seg_result:
                fig_2d.add_annotation(
                    xref="paper", yref="paper",
                    x=0.99, y=0.01,
                    text=seg_result.get("model_version", "v1.0"),
                    showarrow=False,
                    font=dict(color="rgba(200,200,200,0.65)", size=10),
                    xanchor="right", yanchor="bottom",
                )
            st.plotly_chart(fig_2d, key="viewer_2d", use_container_width=True,
                            config={"displayModeBar": False})
            click_data = None
        else:
            # PIL viewer + streamlit_image_coordinates for reliable click detection
            frame = build_frame(
                volume, slice_idx, mask_filtered, show_mask,
                points if _viewer_mode == "Distance" else [],
                spacing,
                excl_mask=_excl_display, show_excl=show_excl,
                poly_points=_poly_pts if _viewer_mode == "Area" else None,
                poly_closed=_poly_closed,
                gt_mask=gt_mask_full, show_gt=show_gt,
            )
            if seg_result:
                _ov_draw = ImageDraw.Draw(frame)
                _ov_font = ImageFont.load_default(size=11)
                _ov_text = seg_result.get("model_version", "v1.0")
                _fw, _fh = frame.size
                _bb = _ov_draw.textbbox((0, 0), _ov_text, font=_ov_font)
                _ov_draw.text(
                    (_fw - (_bb[2] - _bb[0]) - 6, _fh - (_bb[3] - _bb[1]) - 6),
                    _ov_text, fill=(200, 200, 200), font=_ov_font,
                )
            click_data = streamlit_image_coordinates(frame, key="viewer_2d_sic")

        _ctrl_left, _ctrl_mid, _ctrl_right = st.columns([2, 4, 2])
        _ctrl_left.select_slider("Zoom", options=[1, 2, 4, 8], key="_zoom_slider",
                                 format_func=lambda x: f"{x}×",
                                 disabled=_viewer_mode != "Navigate")
        _ctrl_mid.radio("Mode", ["Navigate", "Distance", "Area"], horizontal=True,
                        key="_viewer_mode", label_visibility="collapsed")
        if _viewer_mode == "Distance":
            if _ctrl_right.button("Reset", key="_rst_dist"):
                st.session_state["points"] = []
                st.session_state.pop("_last_sel", None)
                st.rerun()
        elif _viewer_mode == "Area":
            _n_poly = len(st.session_state.get("poly_points", []))
            _poly_done = st.session_state.get("poly_closed", False)
            if not _poly_done and _n_poly >= 3:
                if _ctrl_right.button("Close", key="_close_poly"):
                    st.session_state["poly_closed"] = True
                    st.rerun()
            else:
                if _ctrl_right.button("Reset", key="_rst_poly"):
                    st.session_state["poly_points"] = []
                    st.session_state["poly_closed"] = False
                    st.session_state.pop("_last_sel", None)
                    st.rerun()

        # ── Colour legend ─────────────────────────────────────────────────────
        _legend_items = []
        if show_mask and mask_filtered is not None and mask_filtered.any():
            _legend_items.append(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:rgb{MASK_COLOR};border-radius:2px;margin-right:4px;vertical-align:middle"></span>'
                f'<span style="vertical-align:middle">Prediction</span>'
            )
        if show_gt and gt_mask_full is not None:
            _legend_items.append(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:rgb{GT_COLOR};border-radius:2px;margin-right:4px;vertical-align:middle"></span>'
                f'<span style="vertical-align:middle">Ground truth</span>'
            )
        if show_excl and _excl_arr is not None:
            _legend_items.append(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:rgb{EXCL_COLOR};border-radius:2px;margin-right:4px;vertical-align:middle"></span>'
                f'<span style="vertical-align:middle">Bone exclusion</span>'
            )
        if show_innerskull and st.session_state.get("innerskull_mask") is not None:
            _legend_items.append(
                f'<span style="display:inline-block;width:22px;height:3px;'
                f'background:rgb{INNERSKULL_CONTOUR_COLOR};margin-right:4px;vertical-align:middle"></span>'
                f'<span style="vertical-align:middle">Inner skull</span>'
            )
        if _legend_items:
            _legend_html = "&nbsp;&nbsp;".join(_legend_items)
            st.markdown(
                f'<p style="font-size:0.78rem;color:#aaa;margin:2px 0 6px 0">{_legend_html}</p>',
                unsafe_allow_html=True,
            )

        if _viewer_mode == "Distance":
            if len(points) == 0:
                st.caption("Click on the image to mark P1")
            elif len(points) == 1:
                st.caption("P1 marked. Click to mark P2.")
            else:
                dist = compute_distance_mm(points[0], points[1], spacing)
                st.markdown(f"**Distance: {dist:.1f} mm**")
        elif _viewer_mode == "Area":
            if len(_poly_pts) == 0:
                st.caption("Click to add polygon vertices")
            elif not _poly_closed:
                st.caption(
                    f"{len(_poly_pts)} point(s). "
                    "Click near P1 (orange) or press **Close** to finish."
                )
            else:
                poly_area = compute_polygon_area_mm2(_poly_pts, spacing)
                st.markdown(
                    f"**Area: {poly_area:.1f} mm²** "
                    f"({poly_area / 100:.2f} cm²)"
                )

        if show_area and mask_filtered is not None:
            area = compute_area_mm2(mask_filtered, slice_idx, spacing)
            st.markdown(f"**Lesion area (slice {slice_idx}):** {area:.1f} mm²")

        if seg_result:
            m1, m2 = st.columns(2)
            if mask_filtered is not None:
                vol_ml = float(mask_filtered.sum()) * float(np.prod(spacing)) / 1000.0
                vol_label = "Lesion vol (bone excl.)" if show_excl else "Lesion volume"
            else:
                vol_ml = seg_result["lesion_volume_ml"]
                vol_label = "Lesion volume"
            m1.metric(vol_label, f"{vol_ml:.3f} mL")
            m2.metric("Hemisphere", seg_result["hemisphere"])

        st.caption(
            f"Spacing: D={spacing[0]:.2f} mm · H={spacing[1]:.2f} mm · W={spacing[2]:.2f} mm  |  "
            f"Shape: {volume.shape[0]}×{volume.shape[1]}×{volume.shape[2]}"
        )

    with col_3d:
        fig = build_3d_figure(
            hu_meshes=st.session_state.get("hu_meshes", {}),
            slice_idx=slice_idx,
            spacing=spacing,
            vol_shape=volume.shape,
            volume=volume,
            full_head_mesh=st.session_state.get("ts_full_head"),
            gt_mask=gt_mask_full,
            show_gt=show_gt,
            lesion_mask=mask_filtered,
            show_lesion=show_mask,
        )
        st.plotly_chart(fig, use_container_width=True, key="viewer_3d")
