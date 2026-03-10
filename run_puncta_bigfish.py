"""
Puncta Big-FISH Parallel Pipeline (v6-bigfish)
===============================================
Parallel pipeline to run_puncta_anchor.py (v5).  Two changes only:

  1. Detection: skimage blob_log → Big-FISH detect_spots()
     Big-FISH uses a single LoG sigma (not a scale range) and auto-selects
     the threshold via L-curve elbow detection per nucleus/channel.

  2. Color calling: call_color_normalized (signal/p25_bg) →
                    call_color_spectral (best_channel / second_channel purity)

     Root cause of v5 wrong_winner failures (73/250 reviewed):
     call_color_normalized divides by nucleus_p25_background.
     AF488 (Yellow) p25 ≈ 336 ADU (very low) → ratio = 60496/336 = 180
     AF647 (Purple) p25 ≈ 1072 ADU           → ratio = 65520/1072 = 61
     Result: Yellow "wins" even when the Yellow brightness is only PSF
     bleedthrough from a saturated Purple spot 5 px away.

     Why local annulus SNR failed:
     At the bleedthrough position (370,250), BOTH AF647=65520 and AF488=60496
     are bright. The AF488 annulus is still clean (low bg), so local SNR for
     AF488 remains high (69.73 > AF647 SNR 3.69). The local annulus approach
     does not distinguish true signal from bleedthrough.

     Spectral purity fix:
     At (365,247) [true Purple spot]: ch1/ch3 = 65520/512 = 128 → clear Purple
     At (370,250) [bleedthrough]:     ch1/ch3 = 65520/60496 = 1.08 → reject
     Rule: best_channel must be >= MIN_PURITY × second_channel.
     This is invariant to overall brightness and directly identifies positions
     where two channels are simultaneously elevated (bleedthrough indicator).

Everything else — image loading, registration, cross-round validation,
winner selection, QC figure layout, CSV structure — is identical to v5.
Outputs go to python_results/puncta_bigfish/ for side-by-side comparison.

Biological rationale (same as v5)
----------------------------------
smFISH mRNA molecules are spatially fixed.  A genuine punctum detected at
(y, x) in Hyb4 MUST appear at the same (y, x) in Hyb3 and Hyb2 (after
registration).  Signals present only in one round are noise.

Outputs  (python_results/puncta_bigfish/)
-----------------------------------------
  anchor_candidates.csv  — one row per candidate (same columns as v5)
  anchor_summary.csv     — one row per nucleus
  nucleus_crops/         — QC figures (same 3×4 layout as v5)
"""

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import bigfish.detection as bf_detection
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from aicsimageio import AICSImage
from loguru import logger
from scipy.ndimage import shift as ndimage_shift
from skimage.segmentation import find_boundaries

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Load config ──────────────────────────────────────────────────────────────
_CFG_PATH = PROJECT_ROOT / "config/puncta_bigfish.yaml"
with open(_CFG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

SEARCH_RADIUS = int(_CFG["validation"]["search_radius"])
FIG_DPI       = int(_CFG["output"].get("figure_dpi", 100))
SAVE_FIGURES  = bool(_CFG["output"].get("save_figures", True))

# Big-FISH detection parameters
LOG_KERNEL_SZ    = float(_CFG["detection"]["log_kernel_size"])
MIN_DISTANCE     = int(_CFG["detection"]["minimum_distance"])
MIN_BLOB_SNR     = float(_CFG["detection"].get("min_blob_snr", 2.0))
MAX_BLOBS_PER_CH = int(_CFG["detection"].get("max_blobs_per_channel", 3))

# Spectral purity color calling parameters
_CC = _CFG["color_calling"]
MIN_PURITY     = float(_CC.get("min_purity", 2.0))
MIN_ABS_SIGNAL = int(_CC.get("min_absolute_signal", 300))

# ── Constants ─────────────────────────────────────────────────────────────────
CHANNELS = ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]

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
CH_COLORS_RGB = {
    "Ch1_AF647": np.array([0.608, 0.349, 0.714]),
    "Ch2_AF590": np.array([0.204, 0.596, 0.859]),
    "Ch3_AF488": np.array([0.957, 0.816, 0.247]),
}
COLOR_DISPLAY = {
    "Purple": "#9B59B6",
    "Blue":   "#3498DB",
    "Yellow": "#F4D03F",
    "None":   "#95A5A6",
}

ROUNDS       = ["Hyb4", "Hyb3", "Hyb2"]
ICC_CHANNELS = {"Ch1_AF647": 3, "Ch2_AF590": 1, "Ch3_AF488": 2}
PAD_PX         = 25
MAX_TABLE_ROWS = 10

# ── File paths ─────────────────────────────────────────────────────────────────
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

OUT_DIR   = PROJECT_ROOT / "python_results/puncta_bigfish"
CROPS_DIR = OUT_DIR / "nucleus_crops"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
logger.add(str(OUT_DIR / "puncta_bigfish.log"), rotation="5 MB", level="DEBUG")


# ── Image I/O ──────────────────────────────────────────────────────────────────
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
        data = img.get_image_data("TCZYX")[0, 0]
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


# ── Coordinate helpers ─────────────────────────────────────────────────────────
def shift_labels(labels: np.ndarray, dy: float, dx: float) -> np.ndarray:
    if abs(dy) < 0.01 and abs(dx) < 0.01:
        return labels.astype(np.int32)
    return ndimage_shift(
        labels.astype(np.float32), shift=(-dy, -dx),
        order=0, mode="constant", cval=0
    ).astype(np.int32)


def hybn_position(y_h4: float, x_h4: float, dy: float, dx: float) -> tuple:
    """Transform Hyb4 coordinate to HybN: y_hn = y_h4 − dy, x_hn = x_h4 − dx."""
    return y_h4 - dy, x_h4 - dx


# ── SNR helper (unchanged from v5, used for detection filtering) ───────────────
def _compute_snr(image: np.ndarray, mask: np.ndarray,
                 yc: int, xc: int, r_blob: float) -> float:
    """
    Peak-to-background SNR for a detected blob.
    SNR = max(inside_disk) / mean(annulus)
    Used to filter LoG candidates during detection (same as v5).
    """
    H, W = image.shape
    ys_grid, xs_grid = np.ogrid[:H, :W]
    dist = np.sqrt((ys_grid - yc) ** 2 + (xs_grid - xc) ** 2)

    inside_mask  = (dist <= r_blob) & mask
    annulus_mask = (dist > r_blob) & (dist <= 2.0 * r_blob) & mask

    if not inside_mask.any():
        return 0.0

    peak_inside = float(image[inside_mask].max())

    if annulus_mask.any():
        outside_mean = float(image[annulus_mask].mean())
    else:
        fallback_mask = mask & ~inside_mask
        if not fallback_mask.any():
            return 0.0
        outside_mean = float(image[fallback_mask].mean())

    if outside_mean < 1.0:
        return float("inf") if peak_inside > 0 else 0.0

    return peak_inside / outside_mean


# ── Step 1: Hyb4 detection (Big-FISH per-channel, SNR filter) ─────────────────
def detect_per_channel_bigfish_hyb4(images_hyb4: dict,
                                     nucleus_mask: np.ndarray) -> list:
    """
    Run Big-FISH detect_spots() independently on each Hyb4 channel.
    Return all blobs whose local SNR exceeds MIN_BLOB_SNR.

    Changes from v5:
    - big-fish detect_spots() replaces skimage blob_log()
    - Single sigma (LOG_KERNEL_SZ) instead of sigma range
    - Auto L-curve threshold instead of fixed LOG_THRESHOLD
    - Normalization before detection is NOT needed (Big-FISH handles uint16 directly)

    Parameters
    ----------
    images_hyb4  : {ch: 2D uint16 array} — full-image Hyb4 channels
    nucleus_mask : bool 2D array — True = pixels belonging to this nucleus

    Returns
    -------
    List of N tuples: [(y_global, x_global, sigma, snr), ...]
    """
    ys, xs = np.where(nucleus_mask)
    if len(ys) == 0:
        return []

    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    mask_crop = nucleus_mask[r0:r1, c0:c1]

    all_blobs = []

    for ch in CHANNELS:
        ch_crop = images_hyb4[ch][r0:r1, c0:c1].copy()
        ch_crop[~mask_crop] = 0   # zero outside nucleus

        # Big-FISH: auto threshold via L-curve elbow, single sigma
        try:
            spots = bf_detection.detect_spots(
                images=ch_crop,
                threshold=None,
                log_kernel_size=LOG_KERNEL_SZ,
                minimum_distance=MIN_DISTANCE,
            )
        except Exception:
            # If auto-threshold fails (e.g. all-zero crop), spots = empty
            spots = np.zeros((0, 2), dtype=np.int64)

        # spots: (N, 2) int64 array, columns [y_crop, x_crop]
        for spot in spots:
            y_c, x_c = int(spot[0]), int(spot[1])
            if y_c < 0 or y_c >= mask_crop.shape[0]: continue
            if x_c < 0 or x_c >= mask_crop.shape[1]: continue
            if not mask_crop[y_c, x_c]: continue

            snr = _compute_snr(ch_crop.astype(np.float64), mask_crop,
                               y_c, x_c, np.sqrt(2.0) * LOG_KERNEL_SZ)
            if snr < MIN_BLOB_SNR:
                continue
            all_blobs.append(dict(
                y_global  = y_c + r0,
                x_global  = x_c + c0,
                sigma     = LOG_KERNEL_SZ,
                ch        = ch,
                snr       = snr,
                peak_raw  = float(images_hyb4[ch][y_c + r0, x_c + c0]),
            ))

    if not all_blobs:
        return []

    # Cap at MAX_BLOBS_PER_CH highest-SNR blobs per channel
    by_ch = defaultdict(list)
    for b in all_blobs:
        by_ch[b["ch"]].append(b)

    filtered = []
    for ch, blobs in by_ch.items():
        blobs.sort(key=lambda b: b["snr"], reverse=True)
        filtered.extend(blobs[:MAX_BLOBS_PER_CH])

    return [(b["y_global"], b["x_global"], b["sigma"], b["snr"]) for b in filtered]


# ── Step 2: Cross-round signal measurement (unchanged from v5) ─────────────────
def measure_round_signals(images: dict, regs: dict,
                           y_h4: int, x_h4: int) -> dict:
    """Max intensity in SEARCH_RADIUS window per round per channel."""
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


# ── Step 2b: Per-channel per-round SNR (unchanged from v5) ────────────────────
def measure_per_channel_snr(images: dict,
                             labels_shifted_all: dict,
                             regs: dict,
                             nid: int,
                             y_h4: int, x_h4: int,
                             sigma: float) -> dict:
    """9-value SNR matrix (3 channels × 3 rounds) at anchored position."""
    r_blob = np.sqrt(2.0) * sigma
    snr_all = {}

    for rnd in ROUNDS:
        dy, dx = regs[rnd]
        y_rnd = int(round(y_h4 - dy))
        x_rnd = int(round(x_h4 - dx))

        lbl_rnd  = labels_shifted_all[rnd]
        nuc_mask = (lbl_rnd == nid)
        ys, xs   = np.where(nuc_mask)

        if len(ys) == 0:
            snr_all[rnd] = {ch: 0.0 for ch in CHANNELS}
            continue

        r0, r1 = int(ys.min()), int(ys.max()) + 1
        c0, c1 = int(xs.min()), int(xs.max()) + 1
        mask_crop = nuc_mask[r0:r1, c0:c1]
        y_local   = y_rnd - r0
        x_local   = x_rnd - c0

        snr_all[rnd] = {}
        for ch in CHANNELS:
            ch_crop   = images[rnd][ch][r0:r1, c0:c1].astype(np.float64)
            ch_masked = np.where(mask_crop, ch_crop, 0.0)
            snr_all[rnd][ch] = _compute_snr(
                ch_masked, mask_crop, y_local, x_local, r_blob
            )

    return snr_all


# ── Step 2c: Per-nucleus background (unchanged, kept for CSV output) ───────────
def compute_nucleus_background(images: dict,
                                labels_shifted_all: dict,
                                nid: int) -> dict:
    """p25 background per channel per round (kept for CSV comparison with v5)."""
    bg = {}
    for rnd in ROUNDS:
        nuc_mask = (labels_shifted_all[rnd] == nid)
        bg[rnd] = {}
        for ch in CHANNELS:
            pixels = images[rnd][ch][nuc_mask]
            if len(pixels) >= 4:
                bg[rnd][ch] = float(np.percentile(pixels, 25))
            else:
                bg[rnd][ch] = 1.0
    return bg


# ── Step 3: Color calling — SPECTRAL PURITY (KEY CHANGE vs v5) ────────────────
def call_color_spectral(signals_rnd: dict) -> tuple:
    """
    Color call using SPECTRAL PURITY: best_channel / second_channel >= MIN_PURITY.

    Scientific logic
    ----------------
    v5 failed because signal/p25_bg inflates channels with low nucleus background
    (AF488 p25=336 ADU vs AF647 p25=1072 ADU → 3× advantage).

    Local annulus SNR failed because at the bleedthrough position (370,250):
      - AF647 peak = 65520 (bleed from real spot 5.83 px away)
      - AF488 peak = 60496 (bleedthrough)
      - Both annuli are clean → both local SNRs are high → Yellow still wins

    Spectral purity directly identifies bleedthrough:
      (365,247) true Purple:     ch1/ch3 = 65520/512   = 128 → clear Purple ✓
      (370,250) bleedthrough:    ch1/ch3 = 65520/60496 = 1.08 → both equal → None ✓

    Rule: color is called only if best_channel >= MIN_PURITY × second_channel
    This is brightness-invariant: it measures spectral selectivity, not absolute signal.

    Parameters
    ----------
    signals_rnd : {ch: float} — max pixel in search window per channel
                  (from measure_round_signals, already computed)

    Returns
    -------
    (color_name, best_signal_adu, purity_ratio)
      color_name   : "Purple" / "Blue" / "Yellow" / "None"
      best_signal  : raw ADU of the best channel's search-window max
      purity_ratio : best_signal / second_best_signal
    """
    sorted_sigs   = sorted(signals_rnd.values(), reverse=True)
    best_ch       = max(signals_rnd, key=signals_rnd.get)
    best_sig      = sorted_sigs[0]
    second_sig    = sorted_sigs[1] if len(sorted_sigs) > 1 else 0.0
    purity        = best_sig / max(second_sig, 1.0)

    if purity >= MIN_PURITY and best_sig >= MIN_ABS_SIGNAL:
        return CH_COLOR_NAME[best_ch], best_sig, purity
    return "None", best_sig, purity


# ── Step 4: QC figure (unchanged from v5 except suptitle) ─────────────────────
def _norm_crop(raw: np.ndarray) -> np.ndarray:
    """1st–99th percentile normalization, returns float [0, 1]."""
    p1, p99 = np.percentile(raw, [1, 99])
    return np.clip((raw.astype(np.float64) - p1) / max(p99 - p1, 1.0), 0.0, 1.0)


def _crop_canvas(image_2d: np.ndarray,
                 cy_c: int, cx_c: int,
                 radius: int, H: int, W: int) -> tuple:
    """Fixed-size (2*radius)×(2*radius) canvas, zero-padded at edges."""
    sz = 2 * radius
    canvas = np.zeros((sz, sz), dtype=np.float64)
    r0_req = cy_c - radius
    c0_req = cx_c - radius
    r0_act = max(0, r0_req); r1_act = min(H, cy_c + radius)
    c0_act = max(0, c0_req); c1_act = min(W, cx_c + radius)
    if r1_act > r0_act and c1_act > c0_act:
        canvas[r0_act - r0_req: r1_act - r0_req,
               c0_act - c0_req: c1_act - c0_req] = (
            image_2d[r0_act:r1_act, c0_act:c1_act].astype(np.float64)
        )
    return canvas, r0_req, c0_req


def _draw_puncta_circles(ax, candidates: list, rnd: str,
                          r0_req: int, c0_req: int,
                          dy: float, dx: float):
    """Circle overlays: gold=winner, white=Hyb4 detected, green=confirmed, red=not."""
    confirmed_key = f"confirmed_h{rnd[-1]}"
    for idx, cand in enumerate(candidates):
        y_h4, x_h4   = cand["y_h4"], cand["x_h4"]
        is_winner     = cand.get("is_winner", False)
        sigma         = cand.get("sigma_h4", 3.0)
        circle_r      = float(np.clip(np.sqrt(2.0) * sigma, 2, 15))

        if rnd == "Hyb4":
            y_plot = y_h4 - r0_req
            x_plot = x_h4 - c0_req
            edge_color = "#FFD700" if is_winner else "white"
        else:
            y_rnd, x_rnd = hybn_position(y_h4, x_h4, dy, dx)
            y_plot = y_rnd - r0_req
            x_plot = x_rnd - c0_req
            if is_winner:
                edge_color = "#FFD700"
            else:
                edge_color = "#00EE44" if cand.get(confirmed_key, False) else "#FF3333"

        lw = 2.5 if is_winner else 1.2
        ls = "-"  if is_winner else "--"
        circle = plt.Circle(
            (x_plot, y_plot), radius=circle_r,
            fill=False, edgecolor=edge_color, linewidth=lw, linestyle=ls, zorder=5
        )
        ax.add_patch(circle)
        ax.text(
            x_plot + circle_r + 2, y_plot - circle_r - 2, str(idx + 1),
            color=edge_color, fontsize=6, fontweight="bold",
            ha="left", va="top", zorder=6
        )


def make_nucleus_figure(nid, images, labels_shifted_all, regs, df_props, candidates):
    """3×4 QC figure: rows=rounds, cols=Ch1/Ch2/Ch3/Merged + candidate table."""
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

    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.1],
                            hspace=0.35, wspace=0.12)
    col_headers = [CH_LABELS[ch] for ch in CHANNELS] + ["Merged\n(RGB)"]

    for r_idx, rnd in enumerate(ROUNDS):
        dy, dx  = regs[rnd]
        cy_rnd  = int(round(cy - dy))
        cx_rnd  = int(round(cx - dx))

        lbl_canvas, r0_req, c0_req = _crop_canvas(
            labels_shifted_all[rnd], cy_rnd, cx_rnd, radius, H, W
        )
        mask_crop = (lbl_canvas.astype(np.int32) == nid)
        boundary  = find_boundaries(mask_crop, mode="inner")

        norm_crops = {}
        for ch in CHANNELS:
            ch_canvas, _, _ = _crop_canvas(images[rnd][ch], cy_rnd, cx_rnd, radius, H, W)
            norm_crops[ch] = _norm_crop(ch_canvas)

        for c_idx, ch in enumerate(CHANNELS):
            ax    = fig.add_subplot(gs[r_idx, c_idx])
            color = CH_COLORS_RGB[ch]
            norm  = norm_crops[ch]
            colored = np.clip(
                np.stack([norm * color[0], norm * color[1], norm * color[2]], axis=-1),
                0.0, 1.0
            )
            ax.imshow(colored, interpolation="nearest")
            bd = np.zeros((*boundary.shape, 4), dtype=np.float32)
            bd[boundary] = [0.0, 1.0, 1.0, 0.9]
            ax.imshow(bd, interpolation="nearest")
            _draw_puncta_circles(ax, candidates, rnd, r0_req, c0_req, dy, dx)
            if r_idx == 0: ax.set_title(col_headers[c_idx], fontsize=8, pad=3)
            if c_idx == 0: ax.set_ylabel(rnd, fontsize=8, labelpad=3)
            ax.set_xticks([]); ax.set_yticks([])

        ax_merge = fig.add_subplot(gs[r_idx, 3])
        merged = np.zeros((*norm_crops["Ch1_AF647"].shape, 3), dtype=np.float64)
        for _ch in CHANNELS:
            _nc    = norm_crops[_ch]
            _color = CH_COLORS_RGB[_ch]
            merged += np.stack([_nc * _color[0], _nc * _color[1], _nc * _color[2]], axis=-1)
        merged = np.clip(merged, 0.0, 1.0)
        ax_merge.imshow(merged, interpolation="nearest")
        bd_w = np.zeros((*boundary.shape, 4), dtype=np.float32)
        bd_w[boundary] = [1.0, 1.0, 1.0, 0.9]
        ax_merge.imshow(bd_w, interpolation="nearest")
        _draw_puncta_circles(ax_merge, candidates, rnd, r0_req, c0_req, dy, dx)
        if r_idx == 0: ax_merge.set_title(col_headers[3], fontsize=8, pad=3)
        ax_merge.set_xticks([]); ax_merge.set_yticks([])

    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis("off")

    if n_cands == 0:
        ax_table.text(0.5, 0.5,
            "No puncta detected in Hyb4  (Big-FISH found no blobs above threshold)",
            ha="center", va="center", fontsize=10, color="#CC0000",
            transform=ax_table.transAxes)
    else:
        cands_shown = candidates[:MAX_TABLE_ROWS]
        n_hidden    = n_cands - len(cands_shown)
        col_lbls = [
            "#", "Pos (y,x)", "H4 color", "H4 snr", "H4 max",
            "H3 color", "H3 snr", "H2 color", "H2 snr",
            "H3 ✓", "H2 ✓", "Barcode",
        ]
        table_data  = []
        cell_colors = []
        for idx, cand in enumerate(cands_shown):
            row = [
                str(idx + 1),
                f"{cand['y_h4']}, {cand['x_h4']}",
                cand["color_h4"],
                f"{cand.get('norm_max_h4', 0.0):.2f}",
                f"{cand['max_h4']:.0f}",
                cand["color_h3"],
                f"{cand.get('norm_max_h3', 0.0):.2f}",
                cand["color_h2"],
                f"{cand.get('norm_max_h2', 0.0):.2f}",
                "✓" if cand["confirmed_h3"] else "✗",
                "✓" if cand["confirmed_h2"] else "✗",
                cand["barcode"],
            ]
            row_id_bg = "#FFD700" if cand.get("is_winner") else "#DDDDDD"
            c = [
                row_id_bg, "#EEEEEE",
                COLOR_DISPLAY.get(cand["color_h4"], "#FFFFFF"), "#EEEEEE", "#EEEEEE",
                COLOR_DISPLAY.get(cand["color_h3"], "#FFFFFF"), "#EEEEEE",
                COLOR_DISPLAY.get(cand["color_h2"], "#FFFFFF"), "#EEEEEE",
                "#90EE90" if cand["confirmed_h3"] else "#FFB6C1",
                "#90EE90" if cand["confirmed_h2"] else "#FFB6C1",
                "#EEEEEE",
            ]
            table_data.append(row)
            cell_colors.append(c)

        t = ax_table.table(
            cellText=table_data, colLabels=col_lbls,
            cellLoc="center", loc="upper center", cellColours=cell_colors,
        )
        t.auto_set_font_size(False)
        t.set_fontsize(7)
        t.scale(1, 1.4)
        if n_hidden > 0:
            ax_table.text(0.5, 0.01,
                f"… {n_hidden} more candidates not shown",
                ha="center", va="bottom", fontsize=6, color="#888888",
                transform=ax_table.transAxes)

    winner = next((c for c in candidates if c.get("is_winner")), None)
    if winner:
        w_idx       = candidates.index(winner) + 1
        winner_str  = f"  ★ Winner: #{w_idx} {winner['barcode']}"
    else:
        winner_str = "  ★ Winner: None"

    fig.suptitle(
        f"Nucleus {nid}  |  centroid=({cx}, {cy})  |  "
        f"{n_cands} candidate(s){winner_str}  "
        f"[Big-FISH σ={LOG_KERNEL_SZ}px  SNR≥{MIN_BLOB_SNR}  "
        f"color=spectral_purity  purity≥{MIN_PURITY}  abs≥{MIN_ABS_SIGNAL}ADU]",
        fontsize=9, fontweight="bold", y=0.99,
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 65)
    logger.info("Puncta Big-FISH Parallel Pipeline (v6-bigfish): START")
    logger.info(f"  Detection  : Big-FISH detect_spots()  σ={LOG_KERNEL_SZ}px  "
                f"min_dist={MIN_DISTANCE}px  SNR≥{MIN_BLOB_SNR}")
    logger.info(f"  Color call : spectral_purity  purity≥{MIN_PURITY}  "
                f"abs≥{MIN_ABS_SIGNAL}ADU")
    logger.info(f"  (replaces v5 signal/p25_bg; bleedthrough filtered by purity=best/second)")
    logger.info("=" * 65)

    images   = load_all_images()
    regs     = load_registrations()
    labels   = np.load(str(LABELS_PATH))
    df_props = pd.read_csv(str(PROPS_PATH))

    logger.info("Pre-computing shifted label masks for Hyb3 and Hyb2...")
    labels_shifted_all = {rnd: shift_labels(labels, *regs[rnd]) for rnd in ROUNDS}

    nucleus_ids = df_props["nucleus_id"].tolist()
    logger.info(f"Processing {len(nucleus_ids)} nuclei...")

    all_candidates: list[dict] = []
    all_summaries:  list[dict] = []

    for i, nid in enumerate(nucleus_ids):
        nid = int(nid)
        nucleus_mask = (labels == nid)

        # Background still computed (kept in CSV for comparison with v5)
        nucleus_bg = compute_nucleus_background(images, labels_shifted_all, nid)

        # ── Step 1: Big-FISH detection in Hyb4 (per channel) ────────────
        positions = detect_per_channel_bigfish_hyb4(images["Hyb4"], nucleus_mask)

        # ── Step 2 & 3: Cross-round signals + spectral purity color calling ───
        candidates: list[dict] = []
        for cand_idx, (y_h4, x_h4, sigma, snr) in enumerate(positions):
            signals = measure_round_signals(images, regs, y_h4, x_h4)

            # Spectral purity color calling: uses already-computed search-window
            # max signals. No additional image access needed.
            color_h4, max_h4, norm_h4 = call_color_spectral(signals["Hyb4"])
            color_h3, max_h3, norm_h3 = call_color_spectral(signals["Hyb3"])
            color_h2, max_h2, norm_h2 = call_color_spectral(signals["Hyb2"])

            confirmed_h3 = (color_h3 != "None")
            confirmed_h2 = (color_h2 != "None")

            if confirmed_h3 and confirmed_h2:
                barcode = f"{color_h4}-{color_h3}-{color_h2}"
            else:
                barcode = "unconfirmed"

            snr_9 = measure_per_channel_snr(
                images, labels_shifted_all, regs, nid, y_h4, x_h4, sigma
            )

            cand = dict(
                nucleus_id   = nid,
                candidate_id = cand_idx + 1,
                y_h4         = y_h4,
                x_h4         = x_h4,
                sigma_h4     = round(sigma, 2),
                snr_h4       = round(snr, 2),
                ch1_h4 = signals["Hyb4"]["Ch1_AF647"],
                ch2_h4 = signals["Hyb4"]["Ch2_AF590"],
                ch3_h4 = signals["Hyb4"]["Ch3_AF488"],
                ch1_h3 = signals["Hyb3"]["Ch1_AF647"],
                ch2_h3 = signals["Hyb3"]["Ch2_AF590"],
                ch3_h3 = signals["Hyb3"]["Ch3_AF488"],
                ch1_h2 = signals["Hyb2"]["Ch1_AF647"],
                ch2_h2 = signals["Hyb2"]["Ch2_AF590"],
                ch3_h2 = signals["Hyb2"]["Ch3_AF488"],
                bg_ch1_h4 = round(nucleus_bg["Hyb4"]["Ch1_AF647"], 1),
                bg_ch2_h4 = round(nucleus_bg["Hyb4"]["Ch2_AF590"], 1),
                bg_ch3_h4 = round(nucleus_bg["Hyb4"]["Ch3_AF488"], 1),
                bg_ch1_h3 = round(nucleus_bg["Hyb3"]["Ch1_AF647"], 1),
                bg_ch2_h3 = round(nucleus_bg["Hyb3"]["Ch2_AF590"], 1),
                bg_ch3_h3 = round(nucleus_bg["Hyb3"]["Ch3_AF488"], 1),
                bg_ch1_h2 = round(nucleus_bg["Hyb2"]["Ch1_AF647"], 1),
                bg_ch2_h2 = round(nucleus_bg["Hyb2"]["Ch2_AF590"], 1),
                bg_ch3_h2 = round(nucleus_bg["Hyb2"]["Ch3_AF488"], 1),
                snr_ch1_h4 = round(snr_9["Hyb4"]["Ch1_AF647"], 2),
                snr_ch2_h4 = round(snr_9["Hyb4"]["Ch2_AF590"], 2),
                snr_ch3_h4 = round(snr_9["Hyb4"]["Ch3_AF488"], 2),
                snr_ch1_h3 = round(snr_9["Hyb3"]["Ch1_AF647"], 2),
                snr_ch2_h3 = round(snr_9["Hyb3"]["Ch2_AF590"], 2),
                snr_ch3_h3 = round(snr_9["Hyb3"]["Ch3_AF488"], 2),
                snr_ch1_h2 = round(snr_9["Hyb2"]["Ch1_AF647"], 2),
                snr_ch2_h2 = round(snr_9["Hyb2"]["Ch2_AF590"], 2),
                snr_ch3_h2 = round(snr_9["Hyb2"]["Ch3_AF488"], 2),
                # norm_max now = local SNR (not signal/p25 as in v5)
                color_h4     = color_h4,  max_h4 = max_h4,  norm_max_h4 = round(norm_h4, 2),
                color_h3     = color_h3,  max_h3 = max_h3,  norm_max_h3 = round(norm_h3, 2),
                color_h2     = color_h2,  max_h2 = max_h2,  norm_max_h2 = round(norm_h2, 2),
                confirmed_h3 = confirmed_h3,
                confirmed_h2 = confirmed_h2,
                barcode      = barcode,
            )
            candidates.append(cand)
            all_candidates.append(cand)

        # ── Summary row ──────────────────────────────────────────────────
        confirmed = [c for c in candidates if c["confirmed_h3"] and c["confirmed_h2"]]
        if confirmed:
            best         = max(confirmed, key=lambda c: c["max_h4"] + c["max_h3"] + c["max_h2"])
            best_barcode = best["barcode"]
            decoded_ok   = best_barcode not in ("unconfirmed",)
        else:
            best_barcode = "None"
            decoded_ok   = False

        for c in candidates:
            c["is_winner"] = False
        if confirmed:
            best["is_winner"] = True

        all_summaries.append(dict(
            nucleus_id    = nid,
            n_candidates  = len(candidates),
            n_confirmed   = len(confirmed),
            best_barcode  = best_barcode,
            decoded_ok    = decoded_ok,
        ))

        # ── QC figure ────────────────────────────────────────────────────
        if SAVE_FIGURES:
            fig = make_nucleus_figure(
                nid=nid, images=images, labels_shifted_all=labels_shifted_all,
                regs=regs, df_props=df_props, candidates=candidates,
            )
            fig.savefig(str(CROPS_DIR / f"nucleus_{nid:04d}.png"),
                        dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig)

        if (i + 1) % 100 == 0 or (i + 1) == len(nucleus_ids):
            logger.info(f"  {i+1}/{len(nucleus_ids)} nuclei processed...")

    # ── Save outputs ─────────────────────────────────────────────────────────
    df_candidates = pd.DataFrame(all_candidates)
    df_summary    = pd.DataFrame(all_summaries)
    df_candidates.to_csv(str(OUT_DIR / "anchor_candidates.csv"), index=False)
    df_summary.to_csv(str(OUT_DIR / "anchor_summary.csv"),    index=False)

    n_total   = len(df_summary)
    n_decoded = df_summary["decoded_ok"].sum()
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
    logger.info("Compare with v5: python_results/puncta_anchor/anchor_summary.csv")
    logger.info("  Key nuclei to check: v5 wrong_winner cases (qc_review.csv)")
    logger.info("  Expect: Purple now wins over Yellow at bleedthrough positions")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
