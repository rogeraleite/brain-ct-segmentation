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
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_closing, binary_fill_holes, zoom
from skimage.measure import marching_cubes, find_contours
from src.inference.sliding_window import _skull_exclusion_mask, _SKULL_HU_THRESH
from streamlit_image_coordinates import streamlit_image_coordinates

from src.data.loader import load_nifti
from src.preprocessing.transforms import BRAIN_HU_MIN, BRAIN_HU_MAX, apply_brain_window

# ── Constants ─────────────────────────────────────────────────────────────────

POINT_RADIUS = 4
POINT_COLOR_1 = (255, 80, 80)
POINT_COLOR_2 = (80, 200, 80)
LINE_COLOR = (255, 220, 50)
MASK_COLOR = (220, 50, 50)
MASK_ALPHA = 0.40

MODEL_OPTIONS = {
    "Sliding Window v4 — Dice 0.34 · nativo (650×650) · ~20s":             "/segment/sw",
    "Sliding Window v5 — Dice 0.17 · + augment + supressão de osso · ~20s": "/segment/sw_v5",
    "Resize v1        — Dice 0.27 · 64×128×128        · ~3s":               "/segment/mresize",
}

# HU thresholds for direct CT-based 3D reconstruction
SKULL_EXCL_MM = 20.0     # must match skull_excl_mm in sliding_window_predict
DISPLAY_HW    = 450      # max display size (px) for the 2D slice viewer
SKULL_HU_THRESH = 400    # cortical bone (compact: 700–2000+, spongy: 400–700)
BRAIN_HU_LO     = 20     # brain parenchyma lower bound (white matter ~20–30)
BRAIN_HU_HI     = 80     # brain parenchyma upper bound (grey matter ~35–45, acute blood 50–90)
HEAD_OUTLINE_HU  = -200  # non-air threshold: captures scalp, skull, brain

STRUCTURES = {
    "brain":      {"label": "Cérebro",         "color": "#E8A4A4", "opacity": 0.40},
    "skull":      {"label": "Crânio",          "color": "#F5DEB3", "opacity": 0.20},
    "cerebellum": {"label": "Cerebelo",        "color": "#90CAF9", "opacity": 0.70},
    "brainstem":  {"label": "Tronco cerebral", "color": "#A5D6A7", "opacity": 0.80},
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

@st.cache_data(show_spinner="Carregando volume...")
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


@st.cache_data(show_spinner="Rodando segmentação...")
def call_segment_api(
    file_bytes: bytes, filename: str, api_url: str, endpoint: str
) -> dict:
    response = requests.post(
        f"{api_url}{endpoint}",
        files={"file": (filename, file_bytes, "application/octet-stream")},
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
) -> Image.Image:
    arr = slice_to_uint8(volume, slice_idx)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    rgb = np.array(img, dtype=np.float32)

    if show_mask and mask_full is not None:
        m = mask_full[slice_idx].astype(bool)
        if m.any():
            overlay = np.zeros_like(rgb)
            overlay[m] = MASK_COLOR
            rgb = rgb * (1 - MASK_ALPHA) + overlay * MASK_ALPHA

    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)

    if show_excl and excl_mask is not None:
        excl_slice = excl_mask[slice_idx].astype(np.float32)
        for contour in find_contours(excl_slice, 0.5):
            pts = [(int(round(c[1])), int(round(c[0]))) for c in contour]
            if len(pts) > 1:
                draw.line(pts + [pts[0]], fill=(0, 220, 0), width=2)

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

    return img


def compute_distance_mm(p1: dict, p2: dict, spacing: np.ndarray) -> float:
    dx = (p2["x"] - p1["x"]) * spacing[2]
    dy = (p2["y"] - p1["y"]) * spacing[1]
    return math.sqrt(dx**2 + dy**2)


def compute_area_mm2(mask_full: np.ndarray, slice_idx: int, spacing: np.ndarray) -> float:
    return float(mask_full[slice_idx].sum()) * spacing[1] * spacing[2]


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
        return False, "não instalado"


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
                f"TotalSegmentator saiu com código {proc.returncode}. "
                f"Veja a aba Logs para detalhes."
            )

        masks = {}
        for name, candidates in STRUCTURE_FILENAMES.items():
            for fname in candidates:
                fpath = os.path.join(output_dir, fname)
                if os.path.exists(fpath):
                    arr, _ = load_nifti(fpath)
                    masks[name] = (arr > 0.5).astype(np.uint8)
                    log(f"Estrutura carregada: {name} ({int(arr.sum())} voxels)")
                    break
            else:
                log(f"Estrutura não encontrada: {name}", level="WARN")

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
        log(f"marching_cubes falhou: {e}", level="ERROR")
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
        log(f"Full-head marching_cubes falhou: {e}", level="ERROR")
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
) -> go.Figure:
    """Render skull + brain (HU-based, clipped below slice) + CT texture at slice plane."""
    fig = go.Figure()

    W_mm = float(vol_shape[2]) * spacing[2]
    H_mm = float(vol_shape[1]) * spacing[1]
    D_mm = float(vol_shape[0]) * spacing[0]
    z_mm = float(slice_idx) * spacing[0]

    # 1. TS full-head shell (very transparent — just overall volume context)
    if full_head_mesh is not None:
        v, f = full_head_mesh
        fig.add_trace(go.Mesh3d(
            x=v[:, 2].tolist(), y=v[:, 1].tolist(), z=v[:, 0].tolist(),
            i=f[:, 0].tolist(), j=f[:, 1].tolist(), k=f[:, 2].tolist(),
            color="#C8BAA0", opacity=0.07,
            name="Volume (TS)", showlegend=True,
            lighting=dict(ambient=0.8, diffuse=0.4, specular=0.05),
        ))

    # 2. HU skull — full mesh, semi-transparent shell
    if "skull" in hu_meshes:
        sv, sf = hu_meshes["skull"]
        fig.add_trace(go.Mesh3d(
            x=sv[:, 2].tolist(), y=sv[:, 1].tolist(), z=sv[:, 0].tolist(),
            i=sf[:, 0].tolist(), j=sf[:, 1].tolist(), k=sf[:, 2].tolist(),
            color="#D4C5A0", opacity=0.18,
            name="Crânio (HU>400)", showlegend=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
        ))

    # 3. HU brain — full mesh, semi-transparent
    if "brain" in hu_meshes:
        bv, bf = hu_meshes["brain"]
        fig.add_trace(go.Mesh3d(
            x=bv[:, 2].tolist(), y=bv[:, 1].tolist(), z=bv[:, 0].tolist(),
            i=bf[:, 0].tolist(), j=bf[:, 1].tolist(), k=bf[:, 2].tolist(),
            color="#E8A4A4", opacity=0.40,
            name="Cérebro (HU 20–80)", showlegend=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
        ))

    # 4. CT slice — actual scan image as the cutting plane surface
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
        ys = np.linspace(0.0, H_mm, Hd)
        Z = np.full((Hd, Wd), z_mm)
        fig.add_trace(go.Surface(
            x=xs.tolist(), y=ys.tolist(), z=Z.tolist(),
            surfacecolor=slc_ds.tolist(),
            colorscale="Gray", showscale=False,
            opacity=1.0, name=f"Fatia {slice_idx}", showlegend=True,
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
    st.header("Configurações")
    api_url = st.text_input("URL da API", value="http://localhost:8011")
    uploaded = st.file_uploader("CT Volume (.nii / .nii.gz)", type=["nii", "gz"])

    volume = None
    spacing = None
    seg_result = None
    mask_full = None
    show_area = False
    show_mask = False
    show_excl = False
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
            st.session_state["ts_file"] = filename

        try:
            volume, spacing = load_volume(file_bytes, filename)
        except Exception as e:
            log(f"Erro ao carregar volume: {e}", level="ERROR")
            st.error(f"Erro ao carregar volume: {e}")
            st.stop()

        # Auto-compute HU meshes once per file (skull + brain from CT directly)
        if "hu_meshes" not in st.session_state:
            with st.spinner("Computando malhas 3D (HU: crânio + cérebro)..."):
                log("=== HU meshes iniciado ===")
                st.session_state["hu_meshes"] = compute_hu_meshes(volume, spacing)
                log(f"=== HU meshes concluído: {list(st.session_state['hu_meshes'].keys())} ===")

        # Compute skull exclusion mask once per file+version for visualization
        _EXCL_CACHE_KEY = f"excl_mask_v{SKULL_EXCL_MM}"
        if st.session_state.get("excl_mask_ver") != _EXCL_CACHE_KEY:
            spacing_hw_mm = float(min(spacing[1], spacing[2]))
            st.session_state["excl_mask"] = _skull_exclusion_mask(
                volume, _SKULL_HU_THRESH, SKULL_EXCL_MM, spacing_hw_mm
            )
            st.session_state["excl_mask_ver"] = _EXCL_CACHE_KEY

        # ── Lesion segmentation (API) ──────────────────────────────────────
        st.divider()
        st.subheader("Segmentação de Lesão")
        model_label = st.radio("Modelo", list(MODEL_OPTIONS.keys()))
        endpoint = MODEL_OPTIONS[model_label]

        with st.expander("Comparativo de modelos"):
            st.markdown(
                "| Modelo | Dice | Resolução | Velocidade | Novidades |\n"
                "|---|---|---|---|---|\n"
                "| **SW v4** | 0.339 | 650×650 nativo | ~20s | lesion-centered 70/30, pos_weight=50 |\n"
                "| SW v5 | 0.172 | 650×650 nativo | ~20s | + augment (rot/elastic/jitter) + supressão de osso |\n"
                "| Resize v1 | 0.272 | 64×128×128 | ~3s | baseline |\n\n"
                "_Dice no validation set (40 volumes). "
                "SW v4 = epoch 79, SW v5 = epoch 74, Resize v1 = epoch 46._"
            )

        if st.button("Rodar Segmentação", type="primary"):
            st.session_state.pop("seg_result", None)
            st.session_state.pop("seg_endpoint", None)

        if "seg_result" not in st.session_state or st.session_state.get("seg_endpoint") != endpoint:
            log(f"Chamando API: {api_url}{endpoint}")
            try:
                result = call_segment_api(file_bytes, filename, api_url, endpoint)
                st.session_state["seg_result"] = result
                st.session_state["seg_endpoint"] = endpoint
                log(f"API OK — lesão: {result['lesion_volume_ml']:.3f} mL · {result['hemisphere']}")
            except requests.exceptions.ConnectionError:
                msg = f"API não encontrada em {api_url}"
                log(msg, level="ERROR")
                st.error(msg)
            except requests.exceptions.HTTPError as e:
                msg = f"Erro da API ({e.response.status_code}): {e.response.text}"
                log(msg, level="ERROR")
                st.error(msg)
            except Exception as e:
                log(f"Erro na segmentação: {e}", level="ERROR")
                st.error(f"Erro na segmentação: {e}")

        if "seg_result" in st.session_state:
            seg_result = st.session_state["seg_result"]
            mask_full = decode_mask(seg_result, volume.shape)
            st.caption(
                f"Modelo: {seg_result.get('model_version', 'v1.0')} · "
                f"endpoint: `{endpoint}`"
            )

        # ── View controls ──────────────────────────────────────────────────
        st.divider()
        show_mask = st.checkbox("Mostrar overlay de segmentação", value=True)
        show_excl = st.checkbox("Mostrar zona de exclusão de osso", value=False)
        show_area = st.checkbox("Mostrar área da lesão (fatia)", value=False)
        slice_idx = st.slider("Fatia axial", 0, volume.shape[0] - 1, volume.shape[0] // 2)

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
            st.caption("Vermelho = fatias com lesão · Linha amarela = fatia atual")

        st.divider()
        if st.button("Resetar medição"):
            st.session_state["points"] = []

        # ── TotalSegmentator 3D ────────────────────────────────────────────
        st.divider()
        st.subheader("Visualização 3D")

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
                    "Pesos do modelo não encontrados. "
                    "A primeira execução fará download de ~1.3 GB — pode demorar 10-20 min.",
                    icon="⚠️",
                )
        else:
            st.error("TotalSegmentator não instalado.\n```\npip install totalsegmentator\n```")

        if ts_ok and st.button("Rodar TotalSegmentator", type="secondary"):
            with st.spinner("Rodando TotalSegmentator (fast)… veja a aba Logs para progresso"):
                try:
                    log("=== TotalSegmentator iniciado ===")
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
                    log(f"=== TotalSegmentator concluído: {', '.join(found)} ===")
                    st.success(f"Estruturas: {', '.join(found)}")
                except Exception as e:
                    log(f"TotalSegmentator falhou: {e}", level="ERROR")
                    st.error(str(e))

        if "ts_stats" in st.session_state and st.session_state["ts_stats"]:
            st.caption("Volumes por estrutura (TS):")
            for name, cfg in STRUCTURES.items():
                stats = st.session_state["ts_stats"].get(name, {})
                if stats:
                    st.caption(f"  {cfg['label']}: {stats['volume_ml']:.0f} mL")

# ── Main area (tabs) ───────────────────────────────────────────────────────────
if volume is None:
    st.info("Faça upload de um arquivo NIfTI (.nii ou .nii.gz) na barra lateral.")
    st.stop()

if "points" not in st.session_state:
    st.session_state["points"] = []

points = st.session_state["points"]

tab_viewer, tab_logs = st.tabs(["Viewer", "Logs"])

# ── Viewer tab ─────────────────────────────────────────────────────────────────
with tab_viewer:
    col_2d, col_3d = st.columns([2, 3])

    with col_2d:
        # Version badge above image
        if seg_result:
            model_ver = seg_result.get("model_version", "v1.0")
            st.caption(f"Identificação: {model_ver} · `{endpoint}`")

        frame = build_frame(
            volume, slice_idx, mask_full, show_mask, points, spacing,
            excl_mask=st.session_state.get("excl_mask"),
            show_excl=show_excl,
        )
        # Resize to DISPLAY_HW so the image fits the column without overflowing.
        # Coords from streamlit_image_coordinates are in the resized image space —
        # scale back to native before storing so build_frame always draws correctly.
        nat_w, nat_h = frame.width, frame.height
        disp_w = min(DISPLAY_HW, nat_w)
        disp_h = min(DISPLAY_HW, nat_h)
        frame_disp = frame.resize((disp_w, disp_h), Image.LANCZOS)

        clicked = streamlit_image_coordinates(frame_disp, key="viewer")

        last = st.session_state.get("last_click")
        if clicked is not None and clicked != last:
            st.session_state["last_click"] = clicked
            new_pt = {
                "x": int(round(clicked["x"] * nat_w / disp_w)),
                "y": int(round(clicked["y"] * nat_h / disp_h)),
            }
            if len(points) < 2:
                points.append(new_pt)
            else:
                points[0] = points[1]
                points[1] = new_pt
            st.session_state["points"] = points
            st.rerun()

        if len(points) == 0:
            st.caption("Clique na imagem para marcar P1")
        elif len(points) == 1:
            st.caption("P1 marcado. Clique para marcar P2.")
        else:
            dist = compute_distance_mm(points[0], points[1], spacing)
            st.markdown(f"**Distância: {dist:.1f} mm**")

        if show_area and mask_full is not None:
            area = compute_area_mm2(mask_full, slice_idx, spacing)
            st.markdown(f"**Área da lesão (fatia {slice_idx}):** {area:.1f} mm²")

        if seg_result:
            m1, m2 = st.columns(2)
            m1.metric("Volume da lesão", f"{seg_result['lesion_volume_ml']:.3f} mL")
            m2.metric("Hemisfério", seg_result["hemisphere"])

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
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Logs tab ───────────────────────────────────────────────────────────────────
with tab_logs:
    logs = st.session_state.get("logs", [])
    col_l, col_r = st.columns([4, 1])
    col_l.caption(f"{len(logs)} entradas")
    if col_r.button("Limpar logs"):
        st.session_state["logs"] = []
        st.rerun()

    if logs:
        st.code("\n".join(reversed(logs)), language=None)
    else:
        st.info("Nenhum log ainda. Execute uma segmentação ou rode o TotalSegmentator.")
