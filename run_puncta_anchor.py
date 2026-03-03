"""
Puncta Anchor Validation Pipeline
==================================
Uses Hyb4 as a calibration/anchor round to validate puncta across all rounds.

Biological rationale
--------------------
smFISH mRNA molecules are spatially fixed — only the hybridization probe
color changes between rounds.  A genuine punctum detected at position (y, x)
in Hyb4 *must* appear at the same (y, x) in Hyb3 and Hyb2 (after registration
correction).  Signals that appear only in one round are noise.

Algorithm
---------
1. Per nucleus — detect ALL puncta positions in Hyb4:
       LoG blob detection on max-projection(Ch1, Ch2, Ch3) within nucleus mask.
2. Per candidate position — cross-reference in Hyb3 and Hyb2:
       HybN position = (y_h4 − dy,  x_h4 − dx)        [registration formula]
       Measure max intensity in a search_radius window for each channel.
       call_color = argmax(Ch1, Ch2, Ch3) if max ≥ min_signal else "None".
       confirmed_h3 / confirmed_h2 = signal present in that round.
3. Report ALL candidates per nucleus (not just the best), with QC figures for
   manual inspection.

Outputs  (python_results/puncta_anchor/)
-----------------------------------------
  anchor_candidates.csv  — one row per candidate per nucleus
  anchor_summary.csv     — one row per nucleus (n_candidates, best barcode)
  nucleus_crops/         — nucleus_XXXX.png QC figure for every nucleus

QC figure layout (3 rounds × 4 columns + candidate table)
----------------------------------------------------------
  Columns : Ch1(Purple) | Ch2(Blue) | Ch3(Yellow) | Merged-RGB
  Overlays: white circles = Hyb4 detected positions
            green circles = confirmed in HybN (signal ≥ min_signal)
            red   circles = not confirmed in HybN
  Table   : per-candidate signals, color calls, confirmation status, barcode
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from aicsimageio import AICSImage
from loguru import logger
from scipy.ndimage import shift as ndimage_shift
from skimage.feature import blob_log
from skimage.segmentation import find_boundaries

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Load config ──────────────────────────────────────────────────────────────
_CFG_PATH = PROJECT_ROOT / "config/puncta_anchor.yaml"
with open(_CFG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

LOG_MIN_SIGMA  = float(_CFG["detection"]["min_sigma"])
LOG_MAX_SIGMA  = float(_CFG["detection"]["max_sigma"])
LOG_THRESHOLD  = float(_CFG["detection"]["log_threshold"])
SEARCH_RADIUS  = int(_CFG["validation"]["search_radius"])
MIN_SIGNAL     = float(_CFG["validation"]["min_signal"])
FIG_DPI        = int(_CFG["output"].get("figure_dpi", 100))
SAVE_FIGURES   = bool(_CFG["output"].get("save_figures", True))

# ── Constants (mirrors other pipeline scripts) ────────────────────────────────
CHANNELS = ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]

# Fluorophore colour assignments
CH_COLOR_NAME = {
    "Ch1_AF647": "Purple",
    "Ch2_AF590": "Blue",
    "Ch3_AF488": "Yellow",
}
CH_LABELS = {
    "Ch1_AF647": "Ch1\n(Purple)",
    "Ch2_AF590": "Ch2\n(Blue)",
    "Ch3_AF488": "Ch3\n(Yellow)",
}
# RGB display colour vectors (0–1 scale)
CH_COLORS_RGB = {
    "Ch1_AF647": np.array([0.608, 0.349, 0.714]),  # Purple  #9B59B6
    "Ch2_AF590": np.array([0.204, 0.596, 0.859]),  # Blue    #3498DB
    "Ch3_AF488": np.array([0.957, 0.816, 0.247]),  # Yellow  #F4D03F
}
COLOR_DISPLAY = {
    "Purple": "#9B59B6",
    "Blue":   "#3498DB",
    "Yellow": "#F4D03F",
    "None":   "#95A5A6",
}

ROUNDS       = ["Hyb4", "Hyb3", "Hyb2"]
ICC_CHANNELS = {"Ch1_AF647": 3, "Ch2_AF590": 1, "Ch3_AF488": 2}
PAD_PX         = 25   # padding around nucleus radius for figure crop
MAX_TABLE_ROWS = 10   # max candidates shown in the bottom table

# ── File paths ────────────────────────────────────────────────────────────────
HYBNPATH = {
    "hyb2": PROJECT_ROOT / "IMAGES/20260123_psy_redo_3 hyb2_B7-20x-3x3-Z-FOVB_ICC_Processed001.tif",
    "hyb3": PROJECT_ROOT / "IMAGES/20260123_psy_redo_3 hyb3_B7-20x-3x3-Z-FOVB_ICC_Processed001.tif",
}
HYB4_NPY = {
    "Ch1_AF647": PROJECT_ROOT / "python_results/module1/hyb4_crop_Ch1_AF647.npy",
    "Ch2_AF590": PROJECT_ROOT / "python_results/module1/hyb4_crop_Ch2_AF590.npy",
    "Ch3_AF488": PROJECT_ROOT / "python_results/module1/hyb4_crop_Ch3_AF488.npy",
}
CROP_COORDS = PROJECT_ROOT / "python_results/module1/crop_coords.json"
REG_HYB2    = PROJECT_ROOT / "python_results/module3/registration_hyb2_to_hyb4.json"
REG_HYB3    = PROJECT_ROOT / "python_results/module3/registration_hyb3_to_hyb4.json"
LABELS_PATH = PROJECT_ROOT / "python_results/module4/nucleus_labels.npy"
PROPS_PATH  = PROJECT_ROOT / "python_results/module4/nucleus_properties.csv"

OUT_DIR   = PROJECT_ROOT / "python_results/puncta_anchor"
CROPS_DIR = OUT_DIR / "nucleus_crops"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
logger.add(str(OUT_DIR / "puncta_anchor.log"), rotation="5 MB", level="DEBUG")


# ── Image I/O ─────────────────────────────────────────────────────────────────
def load_all_images() -> dict:
    """Load all 3 rounds × 3 channels.  Returns {round: {ch: 2D np.ndarray}}."""
    with open(CROP_COORDS) as f:
        cc = json.load(f)
    y0, x0, h, w = cc["y0"], cc["x0"], cc["crop_h"], cc["crop_w"]

    images = {}
    logger.info("Loading Hyb4 channels (npy)...")
    images["Hyb4"] = {ch: np.load(str(HYB4_NPY[ch])) for ch in CHANNELS}

    for rnd_key, tif_path in [("Hyb3", HYBNPATH["hyb3"]), ("Hyb2", HYBNPATH["hyb2"])]:
        logger.info(f"Loading {rnd_key} ICC_Processed TIF...")
        img  = AICSImage(str(tif_path))
        data = img.get_image_data("TCZYX")[0, 0]   # shape: (5, H, W)
        images[rnd_key] = {
            ch: data[z_idx, y0:y0+h, x0:x0+w]
            for ch, z_idx in ICC_CHANNELS.items()
        }
    return images


def load_registrations() -> dict:
    """Returns {round: (dy, dx)}.  Hyb4 = (0, 0) by definition."""
    regs = {}
    for rnd, path in [("Hyb2", REG_HYB2), ("Hyb3", REG_HYB3)]:
        with open(path) as f:
            d = json.load(f)
        regs[rnd] = (float(d["dy"]), float(d["dx"]))
    regs["Hyb4"] = (0.0, 0.0)
    return regs


# ── Coordinate helpers ────────────────────────────────────────────────────────
def shift_labels(labels: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """
    Shift nucleus label mask from Hyb4 frame into HybN image coordinates.
    ndimage_shift(labels, shift=(−dy, −dx)) moves a pixel originally at
    (y, x) to (y − dy, x − dx), i.e. Hyb4 → HybN frame.
    """
    if abs(dy) < 0.01 and abs(dx) < 0.01:
        return labels.astype(np.int32)
    return ndimage_shift(
        labels.astype(np.float32), shift=(-dy, -dx),
        order=0, mode="constant", cval=0
    ).astype(np.int32)


def hybn_position(y_h4: float, x_h4: float,
                  dy: float, dx: float) -> tuple:
    """
    Transform a Hyb4 pixel coordinate to the corresponding HybN image coordinate.
    Formula: y_hn = y_h4 − dy,  x_hn = x_h4 − dx
    """
    return y_h4 - dy, x_h4 - dx


# ── Step 1: Hyb4 detection ────────────────────────────────────────────────────
def detect_positions_hyb4(images_hyb4: dict,
                           nucleus_mask: np.ndarray) -> list:
    """
    Detect puncta in Hyb4 via LoG on the channel max-projection.

    Scientific logic
    ----------------
    mRNA position is colour-agnostic (same spot emits in whichever channel
    the current probe targets).  Taking max(Ch1, Ch2, Ch3) ensures we detect
    the spot regardless of which colour it is in Hyb4.

    Parameters
    ----------
    images_hyb4 : {ch: 2D array}  — full-image Hyb4 channels
    nucleus_mask : bool 2D array  — True pixels belong to this nucleus

    Returns
    -------
    List of (y_global, x_global) integer tuples in full-image coordinates.
    """
    ys, xs = np.where(nucleus_mask)
    if len(ys) == 0:
        return []

    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1

    # Max-projection across channels within the nucleus bounding box
    max_proj = np.zeros((r1 - r0, c1 - c0), dtype=np.float64)
    for ch in CHANNELS:
        max_proj = np.maximum(max_proj,
                              images_hyb4[ch][r0:r1, c0:c1].astype(np.float64))

    # Apply nucleus mask (zero outside) so LoG doesn't detect background blobs
    mask_crop = nucleus_mask[r0:r1, c0:c1]
    max_proj_masked = np.where(mask_crop, max_proj, 0.0)

    max_val = max_proj_masked.max()
    if max_val < 1.0:
        return []   # empty / very dark nucleus

    norm = max_proj_masked / max_val

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        blobs = blob_log(
            norm,
            min_sigma=LOG_MIN_SIGMA,
            max_sigma=LOG_MAX_SIGMA,
            num_sigma=10,
            threshold=LOG_THRESHOLD,
        )

    # Convert crop-local coordinates → global image coordinates, filter to mask
    positions = []
    for blob in blobs:
        y_c, x_c = int(round(blob[0])), int(round(blob[1]))
        if (0 <= y_c < mask_crop.shape[0] and
                0 <= x_c < mask_crop.shape[1] and
                mask_crop[y_c, x_c]):
            positions.append((y_c + r0, x_c + c0))

    return positions


# ── Step 2: Cross-round signal measurement ────────────────────────────────────
def measure_round_signals(images: dict, regs: dict,
                           y_h4: int, x_h4: int) -> dict:
    """
    For each round and channel, measure max intensity in a SEARCH_RADIUS
    window around the position anchored from Hyb4.

    Coordinate transform: HybN position = (y_h4 − dy, x_h4 − dx).
    A small search window (±SEARCH_RADIUS px) accounts for residual
    sub-pixel registration error after the Phase-Correlation alignment.

    Returns: {round: {ch: float}}
    """
    H, W = next(iter(images["Hyb4"].values())).shape
    signals = {}
    for rnd in ROUNDS:
        dy, dx = regs[rnd]
        y_rnd, x_rnd = hybn_position(y_h4, x_h4, dy, dx)

        y0 = max(0, int(y_rnd) - SEARCH_RADIUS)
        y1 = min(H, int(y_rnd) + SEARCH_RADIUS + 1)
        x0 = max(0, int(x_rnd) - SEARCH_RADIUS)
        x1 = min(W, int(x_rnd) + SEARCH_RADIUS + 1)

        signals[rnd] = {}
        for ch in CHANNELS:
            window = images[rnd][ch][y0:y1, x0:x1]
            signals[rnd][ch] = float(window.max()) if window.size > 0 else 0.0

    return signals


def call_color(signals_rnd: dict) -> tuple:
    """
    Given {ch: signal} for one round, return (color_name, max_signal).
    If no channel exceeds MIN_SIGNAL → ("None", max_signal).
    """
    max_ch  = max(signals_rnd, key=signals_rnd.get)
    max_sig = signals_rnd[max_ch]
    if max_sig < MIN_SIGNAL:
        return "None", max_sig
    return CH_COLOR_NAME[max_ch], max_sig


# ── Step 4: QC figure ─────────────────────────────────────────────────────────
def _norm_crop(raw: np.ndarray) -> np.ndarray:
    """1st–99th percentile normalization, returns float [0, 1]."""
    p1, p99 = np.percentile(raw, [1, 99])
    return np.clip((raw.astype(np.float64) - p1) / max(p99 - p1, 1.0), 0.0, 1.0)


def _crop_canvas(image_2d: np.ndarray,
                 cy_c: int, cx_c: int,
                 radius: int, H: int, W: int) -> tuple:
    """
    Create a fixed-size (2*radius) × (2*radius) float64 canvas centered on
    (cy_c, cx_c).  Regions outside the image are zero-padded.

    This guarantees all panels have identical pixel dimensions regardless of
    whether the nucleus centroid is near an image edge (solves the asymmetric-
    crop / "over-crop" issue seen in Hyb3 and Hyb2 panels).

    Returns
    -------
    canvas  : ndarray float64, shape (2*radius, 2*radius)
    r0_req  : int — row of canvas[0, 0] in global image coordinates (may be <0)
    c0_req  : int — col of canvas[0, 0] in global image coordinates (may be <0)

    Coordinate conversion:
        canvas[y_canvas, x_canvas]  ←→  global (y_canvas + r0_req, x_canvas + c0_req)
    """
    sz = 2 * radius
    canvas = np.zeros((sz, sz), dtype=np.float64)

    r0_req = cy_c - radius   # may be negative (nucleus near top edge)
    c0_req = cx_c - radius   # may be negative (nucleus near left edge)

    r0_act = max(0, r0_req)
    r1_act = min(H, cy_c + radius)
    c0_act = max(0, c0_req)
    c1_act = min(W, cx_c + radius)

    if r1_act > r0_act and c1_act > c0_act:
        canvas[r0_act - r0_req: r1_act - r0_req,
               c0_act - c0_req: c1_act - c0_req] = (
            image_2d[r0_act:r1_act, c0_act:c1_act].astype(np.float64)
        )

    return canvas, r0_req, c0_req


def _draw_puncta_circles(ax, candidates: list, rnd: str,
                          r0_req: int, c0_req: int,
                          dy: float, dx: float):
    """
    Overlay puncta position circles on a canvas-coordinate axis panel.
    Canvas coordinate = global coordinate − (r0_req, c0_req).

    - Hyb4 row:  white circle at detected position
    - HybN rows: green = confirmed, red = not confirmed
    Each circle is labelled with its 1-based candidate index.
    """
    confirmed_key = f"confirmed_h{rnd[-1]}"   # confirmed_h3 or confirmed_h2

    for idx, cand in enumerate(candidates):
        y_h4, x_h4 = cand["y_h4"], cand["x_h4"]

        if rnd == "Hyb4":
            y_plot = y_h4 - r0_req
            x_plot = x_h4 - c0_req
            edge_color = "white"
        else:
            y_rnd, x_rnd = hybn_position(y_h4, x_h4, dy, dx)
            y_plot = y_rnd - r0_req
            x_plot = x_rnd - c0_req
            edge_color = "#00EE44" if cand.get(confirmed_key, False) else "#FF3333"

        circle = plt.Circle(
            (x_plot, y_plot), radius=6,
            fill=False, edgecolor=edge_color, linewidth=1.5, zorder=5
        )
        ax.add_patch(circle)
        ax.text(
            x_plot + 8, y_plot - 8, str(idx + 1),
            color=edge_color, fontsize=6, fontweight="bold",
            ha="left", va="top", zorder=6
        )


def make_nucleus_figure(
    nid: int,
    images: dict,
    labels_shifted_all: dict,   # {rnd: labels_shifted 2D array}
    regs: dict,
    df_props: pd.DataFrame,
    candidates: list,
) -> plt.Figure:
    """
    Build the 3×4 QC figure for one nucleus.

    Rows    : Hyb4, Hyb3, Hyb2
    Columns : Ch1 (purple) | Ch2 (blue) | Ch3 (yellow) | Merged RGB

    Layout fixes vs v1
    ------------------
    1. Fixed-size canvas (_crop_canvas): all panels are always 2*radius × 2*radius
       pixels regardless of whether the nucleus centroid is near an image edge.
       Regions outside the image are zero-padded so nucleus stays centered
       in Hyb3 and Hyb2 panels even after the registration shift.

    2. Table height is capped: only the first MAX_TABLE_ROWS candidates are shown.
       Figure height is fixed at 11 inches so the table never covers image panels.
    """
    prop_row = df_props[df_props["nucleus_id"] == nid].iloc[0]
    cy = int(prop_row["centroid_y"])
    cx = int(prop_row["centroid_x"])

    if "area_px" in prop_row.index:
        radius = int(np.sqrt(float(prop_row["area_px"]) / np.pi) + PAD_PX)
    elif "area" in prop_row.index:
        radius = int(np.sqrt(float(prop_row["area"]) / np.pi) + PAD_PX)
    else:
        radius = PAD_PX * 3

    H, W = next(iter(images["Hyb4"].values())).shape
    n_cands = len(candidates)

    # Fixed figure size — table never grows large enough to hide image panels
    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(4, 4,
                            height_ratios=[1, 1, 1, 1.1],
                            hspace=0.35, wspace=0.12)

    col_headers = [CH_LABELS[ch] for ch in CHANNELS] + ["Merged\n(RGB)"]

    for r_idx, rnd in enumerate(ROUNDS):
        dy, dx = regs[rnd]

        # Centroid in this round's image frame:  Hyb4(y,x) → HybN(y−dy, x−dx)
        cy_rnd = int(round(cy - dy))
        cx_rnd = int(round(cx - dx))

        # Fixed-size canvas for labels (nucleus mask / boundary)
        lbl_canvas, r0_req, c0_req = _crop_canvas(
            labels_shifted_all[rnd], cy_rnd, cx_rnd, radius, H, W
        )
        mask_crop = (lbl_canvas.astype(np.int32) == nid)
        boundary  = find_boundaries(mask_crop, mode="inner")

        # Fixed-size canvas for each channel
        norm_crops = {}
        for ch in CHANNELS:
            ch_canvas, _, _ = _crop_canvas(
                images[rnd][ch], cy_rnd, cx_rnd, radius, H, W
            )
            norm_crops[ch] = _norm_crop(ch_canvas)

        # ── 3 coloured channel panels ──────────────────────────────────────
        for c_idx, ch in enumerate(CHANNELS):
            ax    = fig.add_subplot(gs[r_idx, c_idx])
            color = CH_COLORS_RGB[ch]
            norm  = norm_crops[ch]

            colored = np.clip(
                np.stack([norm * color[0], norm * color[1], norm * color[2]],
                         axis=-1),
                0.0, 1.0
            )
            ax.imshow(colored, interpolation="nearest")

            bd = np.zeros((*boundary.shape, 4), dtype=np.float32)
            bd[boundary] = [0.0, 1.0, 1.0, 0.9]
            ax.imshow(bd, interpolation="nearest")

            _draw_puncta_circles(ax, candidates, rnd, r0_req, c0_req, dy, dx)

            if r_idx == 0:
                ax.set_title(col_headers[c_idx], fontsize=8, pad=3)
            if c_idx == 0:
                ax.set_ylabel(rnd, fontsize=8, labelpad=3)
            ax.set_xticks([]); ax.set_yticks([])

        # ── Merged RGB panel ───────────────────────────────────────────────
        ax_merge = fig.add_subplot(gs[r_idx, 3])
        merged = np.clip(np.stack([
            norm_crops["Ch1_AF647"],   # R — Purple
            norm_crops["Ch3_AF488"],   # G — Yellow
            norm_crops["Ch2_AF590"],   # B — Blue
        ], axis=-1), 0.0, 1.0)
        ax_merge.imshow(merged, interpolation="nearest")

        bd_w = np.zeros((*boundary.shape, 4), dtype=np.float32)
        bd_w[boundary] = [1.0, 1.0, 1.0, 0.9]
        ax_merge.imshow(bd_w, interpolation="nearest")

        _draw_puncta_circles(ax_merge, candidates, rnd, r0_req, c0_req, dy, dx)

        if r_idx == 0:
            ax_merge.set_title(col_headers[3], fontsize=8, pad=3)
        ax_merge.set_xticks([]); ax_merge.set_yticks([])

    # ── Candidate table (capped at MAX_TABLE_ROWS) ─────────────────────────
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis("off")

    if n_cands == 0:
        ax_table.text(
            0.5, 0.5,
            "No puncta detected in Hyb4  (LoG found no blobs above threshold)",
            ha="center", va="center", fontsize=10, color="#CC0000",
            transform=ax_table.transAxes
        )
    else:
        cands_shown = candidates[:MAX_TABLE_ROWS]
        n_hidden    = n_cands - len(cands_shown)

        col_lbls = [
            "#", "Pos (y,x)", "H4 color", "H4 max",
            "H3 color", "H3 max", "H2 color", "H2 max",
            "H3 ✓", "H2 ✓", "Barcode",
        ]
        table_data  = []
        cell_colors = []

        for idx, cand in enumerate(cands_shown):
            row = [
                str(idx + 1),
                f"{cand['y_h4']}, {cand['x_h4']}",
                cand["color_h4"], f"{cand['max_h4']:.0f}",
                cand["color_h3"], f"{cand['max_h3']:.0f}",
                cand["color_h2"], f"{cand['max_h2']:.0f}",
                "✓" if cand["confirmed_h3"] else "✗",
                "✓" if cand["confirmed_h2"] else "✗",
                cand["barcode"],
            ]
            c = [
                "#DDDDDD", "#EEEEEE",
                COLOR_DISPLAY.get(cand["color_h4"], "#FFFFFF"), "#EEEEEE",
                COLOR_DISPLAY.get(cand["color_h3"], "#FFFFFF"), "#EEEEEE",
                COLOR_DISPLAY.get(cand["color_h2"], "#FFFFFF"), "#EEEEEE",
                "#90EE90" if cand["confirmed_h3"] else "#FFB6C1",
                "#90EE90" if cand["confirmed_h2"] else "#FFB6C1",
                "#EEEEEE",
            ]
            table_data.append(row)
            cell_colors.append(c)

        t = ax_table.table(
            cellText=table_data,
            colLabels=col_lbls,
            cellLoc="center",
            loc="upper center",
            cellColours=cell_colors,
        )
        t.auto_set_font_size(False)
        t.set_fontsize(7)
        t.scale(1, 1.4)

        if n_hidden > 0:
            ax_table.text(
                0.5, 0.01,
                f"… {n_hidden} more candidates not shown  "
                f"(lower log_threshold in config/puncta_anchor.yaml to reduce detections)",
                ha="center", va="bottom", fontsize=6, color="#888888",
                transform=ax_table.transAxes
            )

    fig.suptitle(
        f"Nucleus {nid}  |  centroid=({cx}, {cy})  |  "
        f"{n_cands} Hyb4 candidate(s)  "
        f"[LoG thr={LOG_THRESHOLD}  search_r={SEARCH_RADIUS}px  min_sig={MIN_SIGNAL:.0f}]",
        fontsize=9, fontweight="bold", y=0.99,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 65)
    logger.info("Puncta Anchor Validation Pipeline: START")
    logger.info(f"  Detection : LoG  min_σ={LOG_MIN_SIGMA}  max_σ={LOG_MAX_SIGMA}"
                f"  threshold={LOG_THRESHOLD}")
    logger.info(f"  Validation: search_radius={SEARCH_RADIUS}px"
                f"  min_signal={MIN_SIGNAL:.0f} ADU")
    logger.info("=" * 65)

    images   = load_all_images()
    regs     = load_registrations()
    labels   = np.load(str(LABELS_PATH))
    df_props = pd.read_csv(str(PROPS_PATH))

    # Pre-compute shifted labels once per round (expensive ndimage_shift
    # called only 2 times total, not 1230 × 2 times inside the nucleus loop)
    logger.info("Pre-computing shifted label masks for Hyb3 and Hyb2...")
    labels_shifted_all = {
        rnd: shift_labels(labels, *regs[rnd])
        for rnd in ROUNDS
    }

    nucleus_ids = df_props["nucleus_id"].tolist()
    logger.info(f"Processing {len(nucleus_ids)} nuclei...")

    all_candidates: list[dict] = []
    all_summaries:  list[dict] = []

    for i, nid in enumerate(nucleus_ids):
        nid = int(nid)
        nucleus_mask = (labels == nid)

        # ── Step 1: Detect positions in Hyb4 ─────────────────────────────
        positions = detect_positions_hyb4(images["Hyb4"], nucleus_mask)

        # ── Step 2 & 3: Measure signals in all rounds per candidate ───────
        candidates: list[dict] = []
        for cand_idx, (y_h4, x_h4) in enumerate(positions):
            signals      = measure_round_signals(images, regs, y_h4, x_h4)
            color_h4, max_h4 = call_color(signals["Hyb4"])
            color_h3, max_h3 = call_color(signals["Hyb3"])
            color_h2, max_h2 = call_color(signals["Hyb2"])

            confirmed_h3 = max_h3 >= MIN_SIGNAL
            confirmed_h2 = max_h2 >= MIN_SIGNAL

            if confirmed_h3 and confirmed_h2:
                # Barcode = experimental order: Hyb4 (first imaged) → Hyb3 → Hyb2
                barcode = f"{color_h4}-{color_h3}-{color_h2}"
            else:
                barcode = "unconfirmed"

            cand = dict(
                nucleus_id   = nid,
                candidate_id = cand_idx + 1,
                y_h4         = y_h4,
                x_h4         = x_h4,
                # Raw channel signals (useful for threshold-tuning)
                ch1_h4 = signals["Hyb4"]["Ch1_AF647"],
                ch2_h4 = signals["Hyb4"]["Ch2_AF590"],
                ch3_h4 = signals["Hyb4"]["Ch3_AF488"],
                ch1_h3 = signals["Hyb3"]["Ch1_AF647"],
                ch2_h3 = signals["Hyb3"]["Ch2_AF590"],
                ch3_h3 = signals["Hyb3"]["Ch3_AF488"],
                ch1_h2 = signals["Hyb2"]["Ch1_AF647"],
                ch2_h2 = signals["Hyb2"]["Ch2_AF590"],
                ch3_h2 = signals["Hyb2"]["Ch3_AF488"],
                # Color calls and confirmation
                color_h4     = color_h4,  max_h4 = max_h4,
                color_h3     = color_h3,  max_h3 = max_h3,
                color_h2     = color_h2,  max_h2 = max_h2,
                confirmed_h3 = confirmed_h3,
                confirmed_h2 = confirmed_h2,
                barcode      = barcode,
            )
            candidates.append(cand)
            all_candidates.append(cand)

        # ── Summary row: best confirmed candidate ─────────────────────────
        confirmed = [c for c in candidates if c["confirmed_h3"] and c["confirmed_h2"]]
        if confirmed:
            best = max(confirmed, key=lambda c: c["max_h4"] + c["max_h3"] + c["max_h2"])
            best_barcode = best["barcode"]
            decoded_ok   = best_barcode not in ("unconfirmed",)
        else:
            best_barcode = "None"
            decoded_ok   = False

        all_summaries.append(dict(
            nucleus_id    = nid,
            n_candidates  = len(candidates),
            n_confirmed   = len(confirmed),
            best_barcode  = best_barcode,
            decoded_ok    = decoded_ok,
        ))

        # ── Step 4: QC figure ─────────────────────────────────────────────
        if SAVE_FIGURES:
            fig = make_nucleus_figure(
                nid               = nid,
                images            = images,
                labels_shifted_all = labels_shifted_all,
                regs              = regs,
                df_props          = df_props,
                candidates        = candidates,
            )
            fig.savefig(
                str(CROPS_DIR / f"nucleus_{nid:04d}.png"),
                dpi=FIG_DPI, bbox_inches="tight"
            )
            plt.close(fig)

        if (i + 1) % 100 == 0 or (i + 1) == len(nucleus_ids):
            logger.info(f"  {i+1}/{len(nucleus_ids)} nuclei processed...")

    # ── Save output tables ────────────────────────────────────────────────────
    df_candidates = pd.DataFrame(all_candidates)
    df_summary    = pd.DataFrame(all_summaries)
    df_candidates.to_csv(str(OUT_DIR / "anchor_candidates.csv"), index=False)
    df_summary.to_csv(str(OUT_DIR / "anchor_summary.csv"),    index=False)

    # ── Final report ──────────────────────────────────────────────────────────
    n_total    = len(df_summary)
    n_decoded  = df_summary["decoded_ok"].sum()
    decoded_rt = n_decoded / n_total * 100

    logger.info("=" * 65)
    logger.info("COMPLETE")
    logger.info(f"  Total nuclei        : {n_total}")
    logger.info(f"  Decoded (confirmed) : {n_decoded}  ({decoded_rt:.1f}%)")
    logger.info(f"  0 candidates        : {(df_summary['n_candidates'] == 0).sum()}")
    logger.info(f"  1 candidate         : {(df_summary['n_candidates'] == 1).sum()}")
    logger.info(f"  2 candidates        : {(df_summary['n_candidates'] == 2).sum()}")
    logger.info(f"  ≥3 candidates       : {(df_summary['n_candidates'] >= 3).sum()}")
    logger.info(f"  Avg candidates/nucleus: {df_summary['n_candidates'].mean():.2f}")
    if SAVE_FIGURES:
        logger.info(f"  QC figures          : {CROPS_DIR}")
    logger.info("=" * 65)
    logger.info("Next steps:")
    logger.info("  1. Review nucleus_crops/ — white=Hyb4 detected, green=confirmed, red=not")
    logger.info("  2. If too many/few candidates: adjust log_threshold in config/puncta_anchor.yaml")
    logger.info("  3. If cross-round misses real signals: increase search_radius or lower min_signal")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
