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

Algorithm (v3 — per-channel argmax detection)
----------------------------------------------
1. Per nucleus — find the single best punctum position in Hyb4:
       For EACH channel separately, find the brightest pixel position within
       the nucleus mask and compute its normalized signal (peak / nucleus_p25_bg).
       Pick the channel+position with the highest normalized signal.
       Guarantees exactly 0 or 1 candidate per nucleus (by design).

       Why not MIP detection (v1/v2)?
       MIP = max(Ch1, Ch2, Ch3) per pixel.  The detected position is where ANY
       channel peaks — e.g., a small Yellow pixel at (y1, x1) can dominate the MIP
       even if a larger Purple spot exists at (y2, x2).  We then measure Purple *at
       the Yellow position*, find it weak there, and incorrectly call Yellow.
       Per-channel detection finds each channel's own argmax, then picks the winner.

2. Per candidate position — cross-reference in Hyb3 and Hyb2:
       HybN position = (y_h4 − dy,  x_h4 − dx)        [registration formula]
       Measure max intensity in a search_radius window for each channel.
       call_color_normalized: peak / nucleus_p25_bg ≥ per-channel threshold
                              AND peak ≥ min_absolute_signal (absolute floor).
       confirmed_h3 / confirmed_h2 = both conditions met in that round.

3. Report the single candidate with QC figure for every nucleus.

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

SEARCH_RADIUS  = int(_CFG["validation"]["search_radius"])
FIG_DPI        = int(_CFG["output"].get("figure_dpi", 100))
SAVE_FIGURES   = bool(_CFG["output"].get("save_figures", True))

# LoG detection parameters (v5 — per-channel, multi-candidate, SNR-filtered)
LOG_MIN_SIGMA        = float(_CFG["detection"]["min_sigma"])
LOG_MAX_SIGMA        = float(_CFG["detection"]["max_sigma"])
LOG_THRESHOLD        = float(_CFG["detection"]["log_threshold"])
MIN_BLOB_SNR         = float(_CFG["detection"].get("min_blob_snr", 2.0))
MAX_BLOBS_PER_CH     = int(_CFG["detection"].get("max_blobs_per_channel", 3))

# Per-channel normalized signal threshold: max_in_window / nucleus_p25_background
_NORM_CFG = _CFG["validation"]["min_signal_normalized"]
MIN_SIGNAL_NORM = {
    "Ch1_AF647": float(_NORM_CFG.get("Ch1_AF647", 2.0)),
    "Ch2_AF590": float(_NORM_CFG.get("Ch2_AF590", 1.5)),
    "Ch3_AF488": float(_NORM_CFG.get("Ch3_AF488", 2.0)),
}

# Absolute signal floor: max_in_window must exceed this ADU value regardless of ratio.
# Prevents diffuse mCherry nuclear background (typically 200–400 ADU) from being
# amplified to a false-positive confirmation when the nucleus p25 background is very low.
MIN_ABS_SIGNAL = int(_CFG["validation"].get("min_absolute_signal", 300))

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


# ── SNR helper ────────────────────────────────────────────────────────────────
def _compute_snr(image: np.ndarray, mask: np.ndarray,
                 yc: int, xc: int, r_blob: float) -> float:
    """
    Compute inside/outside fluorescence ratio for a detected blob.

    Scientific logic
    ----------------
    A genuine smFISH punctum is a diffraction-limited bright focus.
    Its intensity should be highly concentrated (≥10× the local nuclear
    background).  Noise blobs (from texture, autofluorescence, or imaging
    artifacts) look similar in shape to real spots but lack this contrast.

    inside  = mean intensity within a disk of radius r_blob centred on (yc, xc),
              restricted to nucleus pixels.
    outside = mean intensity in the annulus (r_blob, 2×r_blob), restricted to
              nucleus pixels.  This is the local nuclear background immediately
              surrounding the punctum.
    Fallback: if the annulus contains no nucleus pixels (e.g. blob near the
              nucleus edge), use all nucleus pixels outside the blob disk.

    Parameters
    ----------
    image  : 2D float array, max-projection within nucleus (zeros outside mask)
    mask   : 2D bool array, True = nucleus pixel
    yc, xc : blob centre in image/crop coordinates
    r_blob : blob radius = sqrt(2) × sigma from LoG

    Returns
    -------
    float — inside/outside ratio (0 if undetermined, inf if outside_mean ≈ 0)
    """
    H, W = image.shape
    ys_grid, xs_grid = np.ogrid[:H, :W]
    dist = np.sqrt((ys_grid - yc) ** 2 + (xs_grid - xc) ** 2)

    inside_mask  = (dist <= r_blob) & mask
    annulus_mask = (dist > r_blob) & (dist <= 2.0 * r_blob) & mask

    if not inside_mask.any():
        return 0.0

    # Peak-to-background ratio: standard in smFISH / FISH literature.
    # Using the peak (max) inside instead of mean avoids the dilution problem
    # where background pixels within the blob disk drag the mean down.
    # A genuine diffraction-limited punctum has a very bright single pixel
    # at its centre; mean-inside would underestimate this by 3-5×.
    peak_inside = float(image[inside_mask].max())

    if annulus_mask.any():
        outside_mean = float(image[annulus_mask].mean())
    else:
        # Fallback: all nucleus pixels not inside the blob
        fallback_mask = mask & ~inside_mask
        if not fallback_mask.any():
            return 0.0
        outside_mean = float(image[fallback_mask].mean())

    if outside_mean < 1.0:
        return float("inf") if peak_inside > 0 else 0.0

    return peak_inside / outside_mean


# ── Step 1: Hyb4 detection (per-channel LoG, multi-candidate, SNR filter) ───
def detect_per_channel_log_hyb4(images_hyb4: dict,
                                  nucleus_mask: np.ndarray) -> list:
    """
    Run LoG blob detection independently on each Hyb4 channel.
    Return ALL blobs whose local SNR exceeds MIN_BLOB_SNR.
    Winner selection is deferred to the cross-round confirmation step in main().

    Scientific logic
    ----------------
    Per-channel LoG (not MIP): each channel is detected independently so that
    a bright spot in one channel does not suppress detection in another.

    Local SNR filter (peak inside blob / mean of surrounding annulus) removes
    diffuse background "bumps" (e.g. mCherry nuclear background) that LoG picks
    up at low threshold.  Background bumps have SNR ≈ 1 (annulus ≈ interior);
    real diffraction-limited smFISH puncta have SNR >> 1 (sharp peak above
    surrounding fluorescence).

    Winner selection (cross-round confirmation):
    Among all SNR-passing candidates, the one confirmed in the most rounds
    (Hyb3 + Hyb2) is selected.  Tiebreak: highest total signal across rounds.
    This makes cross-round biological consistency — not single-round brightness
    or blob size — the primary decision criterion.

    Parameters
    ----------
    images_hyb4  : {ch: 2D array}  — full-image Hyb4 channels (ADU)
    nucleus_mask : bool 2D array   — True = pixels belonging to this nucleus

    Returns
    -------
    List of N tuples: [(y_global, x_global, sigma, snr), ...]
    N = number of blobs passing SNR filter (may be 0 or many).
    """
    ys, xs = np.where(nucleus_mask)
    if len(ys) == 0:
        return []

    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    mask_crop = nucleus_mask[r0:r1, c0:c1]

    passing: list[tuple] = []

    for ch in CHANNELS:
        ch_crop  = images_hyb4[ch][r0:r1, c0:c1].astype(np.float64)
        ch_in    = np.where(mask_crop, ch_crop, 0.0)
        nuc_vals = ch_crop[mask_crop]
        if len(nuc_vals) == 0:
            continue

        # Normalize within nucleus pixels (p2/p99.5 stretch → [0, 1])
        lo = np.percentile(nuc_vals, 2)
        hi = np.percentile(nuc_vals, 99.5)
        if hi <= lo:
            continue
        norm_crop = np.where(mask_crop,
                             np.clip((ch_in - lo) / (hi - lo), 0.0, 1.0),
                             0.0)

        blobs = blob_log(norm_crop,
                         min_sigma=LOG_MIN_SIGMA,
                         max_sigma=LOG_MAX_SIGMA,
                         threshold=LOG_THRESHOLD,
                         overlap=0.5)

        # Score every in-mask blob by local SNR
        ch_blobs: list[tuple] = []
        for blob in blobs:
            y_c, x_c, sigma = int(blob[0]), int(blob[1]), float(blob[2])
            if not mask_crop[y_c, x_c]:
                continue
            snr = _compute_snr(norm_crop, mask_crop, y_c, x_c,
                               np.sqrt(2.0) * sigma)
            if snr < MIN_BLOB_SNR:
                continue
            ch_blobs.append((y_c + r0, x_c + c0, sigma, snr))

        # Keep at most MAX_BLOBS_PER_CH highest-SNR blobs per channel
        ch_blobs.sort(key=lambda b: b[3], reverse=True)
        passing.extend(ch_blobs[:MAX_BLOBS_PER_CH])

    return passing


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


# ── Step 2b: Per-channel per-round SNR ───────────────────────────────────────
def measure_per_channel_snr(images: dict,
                             labels_shifted_all: dict,
                             regs: dict,
                             nid: int,
                             y_h4: int, x_h4: int,
                             sigma: float) -> dict:
    """
    Compute per-channel, per-round SNR at the Hyb4-anchored position.

    Scientific logic
    ----------------
    The same punctum position (anchored from Hyb4) is sampled independently
    in each channel of each round using the peak-to-background ratio:
        SNR = peak_inside / mean_outside

    This gives a 3-round × 3-channel matrix (9 values) that reveals:
      - Which channels have genuine signal above local background
      - Whether Ch2_AF590 (mCherry-affected) has a systematically different
        SNR distribution from Ch1/Ch3 (important for per-channel thresholding)
      - Whether SNR differs between rounds (signal vs carry-over rounds)

    Parameters
    ----------
    images             : {round: {ch: 2D array}}
    labels_shifted_all : {round: 2D label array in that round's frame}
    regs               : {round: (dy, dx)}
    nid                : nucleus ID
    y_h4, x_h4        : candidate position in global Hyb4 frame
    sigma              : LoG sigma; blob radius = sqrt(2) * sigma

    Returns
    -------
    {round: {ch: snr_float}}  — 9 values total
    """
    r_blob = np.sqrt(2.0) * sigma
    snr_all = {}

    for rnd in ROUNDS:
        dy, dx = regs[rnd]
        y_rnd = int(round(y_h4 - dy))
        x_rnd = int(round(x_h4 - dx))

        # Nucleus mask in this round's image frame
        lbl_rnd   = labels_shifted_all[rnd]
        nuc_mask  = (lbl_rnd == nid)
        ys, xs    = np.where(nuc_mask)

        if len(ys) == 0:
            snr_all[rnd] = {ch: 0.0 for ch in CHANNELS}
            continue

        r0, r1 = int(ys.min()), int(ys.max()) + 1
        c0, c1 = int(xs.min()), int(xs.max()) + 1
        mask_crop = nuc_mask[r0:r1, c0:c1]

        # Position in bounding-box local coordinates
        y_local = y_rnd - r0
        x_local = x_rnd - c0

        snr_all[rnd] = {}
        for ch in CHANNELS:
            ch_crop   = images[rnd][ch][r0:r1, c0:c1].astype(np.float64)
            ch_masked = np.where(mask_crop, ch_crop, 0.0)
            snr_all[rnd][ch] = _compute_snr(
                ch_masked, mask_crop, y_local, x_local, r_blob
            )

    return snr_all


# ── Step 2c: Per-nucleus per-channel background ───────────────────────────────
def compute_nucleus_background(images: dict,
                                labels_shifted_all: dict,
                                nid: int) -> dict:
    """
    Compute per-channel per-round nuclear background for one nucleus.

    Scientific logic
    ----------------
    Background = 25th percentile of all pixels within the nucleus mask,
    measured independently for each channel and each round.

    Rationale for p25 (not median or mean):
      - The nucleus contains a few bright puncta; the median and mean are
        pulled upward, overestimating background.
      - p25 sits below most puncta signal and above detector read noise,
        giving a stable, robust floor estimate.
      - Per-nucleus: removes cell-to-cell variation in baseline fluorescence
        (autofluorescence, expression level, imaging depth variation).
      - Per-channel: each dye has a different offset (mCherry Ch2_AF590
        creates diffuse nuclear background not present in Ch1/Ch3).
      - Per-round: the same nucleus may differ across Hyb4/Hyb3/Hyb2 due
        to photobleaching or reagent washout between rounds.

    Parameters
    ----------
    images             : {round: {ch: 2D array}}
    labels_shifted_all : {round: 2D label array in that round's image frame}
    nid                : nucleus ID (integer label value)

    Returns
    -------
    {round: {ch: float}}  — 9 background estimates (p25 of nucleus pixels)
    """
    bg = {}
    for rnd in ROUNDS:
        nuc_mask = (labels_shifted_all[rnd] == nid)
        bg[rnd] = {}
        for ch in CHANNELS:
            pixels = images[rnd][ch][nuc_mask]
            if len(pixels) >= 4:
                bg[rnd][ch] = float(np.percentile(pixels, 25))
            else:
                bg[rnd][ch] = 1.0   # fallback for tiny / edge nuclei
    return bg


def call_color_normalized(signals_rnd: dict, bg_rnd: dict) -> tuple:
    """
    Color call using per-nucleus per-channel normalized signal.

    Scientific logic
    ----------------
    normalized_signal[ch] = max_in_search_window[ch] / nucleus_p25_background[ch]

    This is a cell-independent criterion: a ratio of 2.0 means "this spot is
    2× brighter than this cell's own baseline fluorescence in that channel",
    regardless of the absolute ADU level.  Per-channel thresholds account for
    the fact that Ch2_AF590 (mCherry) creates a diffuse nuclear background that
    systematically lowers SNR relative to Ch1/Ch3.

    Parameters
    ----------
    signals_rnd : {ch: float} — raw max pixel value in search window, per channel
    bg_rnd      : {ch: float} — nucleus 25th-percentile background, per channel

    Returns
    -------
    (color_name, max_abs_signal, max_norm_signal)
      color_name     : "Purple" / "Blue" / "Yellow" / "None"
      max_abs_signal : raw ADU value of the winning channel
      max_norm_signal: normalized ratio of the winning channel
    """
    norm_signals = {}
    for ch in CHANNELS:
        bg = max(bg_rnd.get(ch, 1.0), 1.0)   # floor at 1.0 to avoid division by zero
        norm_signals[ch] = signals_rnd[ch] / bg

    max_ch   = max(norm_signals, key=norm_signals.get)
    max_norm = norm_signals[max_ch]
    max_abs  = signals_rnd[max_ch]
    threshold = MIN_SIGNAL_NORM[max_ch]

    # Dual confirmation gate:
    #   1. Normalized ratio ≥ per-channel threshold (cell-independent criterion)
    #   2. Absolute signal ≥ MIN_ABS_SIGNAL (prevents diffuse mCherry background
    #      from being amplified to a false positive when nucleus p25 is very low)
    if max_norm < threshold or max_abs < MIN_ABS_SIGNAL:
        return "None", max_abs, max_norm
    return CH_COLOR_NAME[max_ch], max_abs, max_norm


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
        is_winner = cand.get("is_winner", False)

        # Circle radius reflects actual LoG blob size: r = √2 × sigma (px)
        # Clamp to [2, 15] px so circles remain visible but accurate.
        sigma = cand.get("sigma_h4", 3.0)
        circle_r = float(np.clip(np.sqrt(2.0) * sigma, 2, 15))

        if rnd == "Hyb4":
            y_plot = y_h4 - r0_req
            x_plot = x_h4 - c0_req
            edge_color = "#FFD700" if is_winner else "white"   # gold = winner
        else:
            y_rnd, x_rnd = hybn_position(y_h4, x_h4, dy, dx)
            y_plot = y_rnd - r0_req
            x_plot = x_rnd - c0_req
            if is_winner:
                edge_color = "#FFD700"   # gold follows winner through all rounds
            else:
                edge_color = "#00EE44" if cand.get(confirmed_key, False) else "#FF3333"

        # Winner: solid thick circle; others: dashed thin
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

        # ── Merged RGB panel (Purple + Blue + Yellow using actual display colors) ──
        # Each channel contributes its true display color (not raw R/G/B mapping).
        # This matches the per-channel panels so colors are visually consistent.
        ax_merge = fig.add_subplot(gs[r_idx, 3])
        merged = np.zeros((*norm_crops["Ch1_AF647"].shape, 3), dtype=np.float64)
        for _ch in CHANNELS:
            _nc    = norm_crops[_ch]
            _color = CH_COLORS_RGB[_ch]
            merged += np.stack([_nc * _color[0], _nc * _color[1], _nc * _color[2]],
                               axis=-1)
        merged = np.clip(merged, 0.0, 1.0)
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
            "#", "Pos (y,x)", "H4 color", "H4 norm", "H4 max",
            "H3 color", "H3 norm", "H2 color", "H2 norm",
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
            row_id_bg = "#FFD700" if cand.get("is_winner") else "#DDDDDD"  # gold # cell = winner
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
                f"(lower norm thresholds in config/puncta_anchor.yaml to reduce detections)",
                ha="center", va="bottom", fontsize=6, color="#888888",
                transform=ax_table.transAxes
            )

    winner = next((c for c in candidates if c.get("is_winner")), None)
    if winner:
        w_idx = candidates.index(winner) + 1
        winner_str = f"  ★ Winner: #{w_idx} {winner['barcode']}"
    else:
        winner_str = "  ★ Winner: None"

    fig.suptitle(
        f"Nucleus {nid}  |  centroid=({cx}, {cy})  |  "
        f"{n_cands} candidate(s){winner_str}  "
        f"[per-ch LoG σ=[{LOG_MIN_SIGMA}–{LOG_MAX_SIGMA}] thr={LOG_THRESHOLD} "
        f"SNR≥{MIN_BLOB_SNR}  "
        f"norm≥Ch1/Ch3:{MIN_SIGNAL_NORM['Ch1_AF647']:.1f} Ch2:{MIN_SIGNAL_NORM['Ch2_AF590']:.1f}  "
        f"abs≥{MIN_ABS_SIGNAL}ADU  search_r={SEARCH_RADIUS}px]",
        fontsize=9, fontweight="bold", y=0.99,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 65)
    logger.info("Puncta Anchor Validation Pipeline: START")
    logger.info(f"  Detection : per-channel LoG  σ=[{LOG_MIN_SIGMA}–{LOG_MAX_SIGMA}]  "
                f"thr={LOG_THRESHOLD}  SNR≥{MIN_BLOB_SNR}  "
                f"top{MAX_BLOBS_PER_CH}/ch  multi-candidate")
    logger.info(f"  Validation: search_radius={SEARCH_RADIUS}px  "
                f"abs_floor={MIN_ABS_SIGNAL}ADU")
    logger.info(f"  Norm thresholds: Ch1≥{MIN_SIGNAL_NORM['Ch1_AF647']:.1f}  "
                f"Ch2≥{MIN_SIGNAL_NORM['Ch2_AF590']:.1f}  "
                f"Ch3≥{MIN_SIGNAL_NORM['Ch3_AF488']:.1f}  "
                f"(peak_in_window / nucleus_p25_background)")
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

        # ── Per-nucleus background (used for cross-round confirmation) ──────
        # nucleus_bg[rnd][ch] = 25th percentile of nucleus pixels = background floor
        nucleus_bg = compute_nucleus_background(images, labels_shifted_all, nid)

        # ── Step 1: Detect best punctum in Hyb4 (per-channel LoG, size-first) ──
        positions = detect_per_channel_log_hyb4(images["Hyb4"], nucleus_mask)

        # ── Step 2 & 3: Measure signals in all rounds per candidate ───────
        candidates: list[dict] = []
        for cand_idx, (y_h4, x_h4, sigma, snr) in enumerate(positions):
            signals = measure_round_signals(images, regs, y_h4, x_h4)

            # Normalized color calls: max_in_window / nucleus_p25_background
            color_h4, max_h4, norm_h4 = call_color_normalized(
                signals["Hyb4"], nucleus_bg["Hyb4"])
            color_h3, max_h3, norm_h3 = call_color_normalized(
                signals["Hyb3"], nucleus_bg["Hyb3"])
            color_h2, max_h2, norm_h2 = call_color_normalized(
                signals["Hyb2"], nucleus_bg["Hyb2"])

            confirmed_h3 = (color_h3 != "None")
            confirmed_h2 = (color_h2 != "None")

            if confirmed_h3 and confirmed_h2:
                # Barcode = experimental order: Hyb4 (first imaged) → Hyb3 → Hyb2
                barcode = f"{color_h4}-{color_h3}-{color_h2}"
            else:
                barcode = "unconfirmed"

            # Per-channel per-round SNR (9 values: 3 channels × 3 rounds)
            snr_9 = measure_per_channel_snr(
                images, labels_shifted_all, regs, nid, y_h4, x_h4, sigma
            )

            cand = dict(
                nucleus_id   = nid,
                candidate_id = cand_idx + 1,
                y_h4         = y_h4,
                x_h4         = x_h4,
                sigma_h4     = round(sigma, 2),   # LoG scale; blob radius = √2 × sigma
                snr_h4       = round(snr, 2),      # max-proj SNR used for LoG filtering
                # Raw channel signals (search-window max, ADU)
                ch1_h4 = signals["Hyb4"]["Ch1_AF647"],
                ch2_h4 = signals["Hyb4"]["Ch2_AF590"],
                ch3_h4 = signals["Hyb4"]["Ch3_AF488"],
                ch1_h3 = signals["Hyb3"]["Ch1_AF647"],
                ch2_h3 = signals["Hyb3"]["Ch2_AF590"],
                ch3_h3 = signals["Hyb3"]["Ch3_AF488"],
                ch1_h2 = signals["Hyb2"]["Ch1_AF647"],
                ch2_h2 = signals["Hyb2"]["Ch2_AF590"],
                ch3_h2 = signals["Hyb2"]["Ch3_AF488"],
                # Per-nucleus per-channel per-round background (p25 of nucleus pixels)
                bg_ch1_h4 = round(nucleus_bg["Hyb4"]["Ch1_AF647"], 1),
                bg_ch2_h4 = round(nucleus_bg["Hyb4"]["Ch2_AF590"], 1),
                bg_ch3_h4 = round(nucleus_bg["Hyb4"]["Ch3_AF488"], 1),
                bg_ch1_h3 = round(nucleus_bg["Hyb3"]["Ch1_AF647"], 1),
                bg_ch2_h3 = round(nucleus_bg["Hyb3"]["Ch2_AF590"], 1),
                bg_ch3_h3 = round(nucleus_bg["Hyb3"]["Ch3_AF488"], 1),
                bg_ch1_h2 = round(nucleus_bg["Hyb2"]["Ch1_AF647"], 1),
                bg_ch2_h2 = round(nucleus_bg["Hyb2"]["Ch2_AF590"], 1),
                bg_ch3_h2 = round(nucleus_bg["Hyb2"]["Ch3_AF488"], 1),
                # Per-channel per-round SNR (peak-inside / local annulus, 9 values)
                snr_ch1_h4 = round(snr_9["Hyb4"]["Ch1_AF647"], 2),
                snr_ch2_h4 = round(snr_9["Hyb4"]["Ch2_AF590"], 2),
                snr_ch3_h4 = round(snr_9["Hyb4"]["Ch3_AF488"], 2),
                snr_ch1_h3 = round(snr_9["Hyb3"]["Ch1_AF647"], 2),
                snr_ch2_h3 = round(snr_9["Hyb3"]["Ch2_AF590"], 2),
                snr_ch3_h3 = round(snr_9["Hyb3"]["Ch3_AF488"], 2),
                snr_ch1_h2 = round(snr_9["Hyb2"]["Ch1_AF647"], 2),
                snr_ch2_h2 = round(snr_9["Hyb2"]["Ch2_AF590"], 2),
                snr_ch3_h2 = round(snr_9["Hyb2"]["Ch3_AF488"], 2),
                # Color calls and confirmation (based on normalized threshold)
                color_h4     = color_h4,  max_h4 = max_h4,  norm_max_h4 = round(norm_h4, 2),
                color_h3     = color_h3,  max_h3 = max_h3,  norm_max_h3 = round(norm_h3, 2),
                color_h2     = color_h2,  max_h2 = max_h2,  norm_max_h2 = round(norm_h2, 2),
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

        # Mark winner flag for QC figure highlighting (must happen before figure)
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
    logger.info("  2. If cross-round thresholding is off: adjust norm thresholds or min_absolute_signal in config")
    logger.info("  3. If cross-round misses real signals: increase search_radius or lower min_absolute_signal")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
