"""
Enhanced QC — Module 5 & 6 Visualizations
==========================================
Usage:
    python run_qc_enhanced.py

Generates the following QC figures into python_results/qc/:

  MODULE 5 QC:
    qc_m5_channel_scatter.png   — Ch1 vs Ch3 scatter per round, colored by decoded call
    qc_m5_snr_distribution.png  — SNR (max/2nd-max) violin per round; low SNR = unreliable
    qc_m5_intensity_heatmap.png — per-nucleus × channel intensity heatmap (sampled)

  MODULE 6 QC:
    qc_m6_spot_overlay_HybN.png — fluorescence composite + nucleus masks colored by
                                   decoded call (one figure per round)
    qc_m6_barcode_counts.png    — bar chart of top barcode counts
    qc_m6_confidence_map.png    — spatial map: marker size = SNR, color = decoded_ok

Prerequisites: run_module1.py → run_module5.py → run_module6.py must have been run.
"""

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from loguru import logger
from scipy.ndimage import shift as ndimage_shift
from skimage.segmentation import find_boundaries

# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
QC_DIR = PROJECT_ROOT / "python_results" / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

# Project color palette
PALETTE = {
    "Purple": "#9B59B6",
    "Blue":   "#3498DB",
    "Yellow": "#F4D03F",
    "None":   "#95A5A6",
}
CHANNELS = ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]
DUAL_HIGH_PCT = 80   # percentile threshold for "dual-high" flag (both Ch1 & Ch3 above this)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data():
    df_int = pd.read_csv(PROJECT_ROOT / "python_results/module5/spot_intensities.csv")
    df_bc  = pd.read_csv(PROJECT_ROOT / "python_results/module6/barcodes.csv")
    labels = np.load(PROJECT_ROOT / "python_results/module4/nucleus_labels.npy")

    with open(PROJECT_ROOT / "python_results/module1/crop_coords.json") as f:
        crop = json.load(f)
    with open(PROJECT_ROOT / "python_results/module3/registration_hyb2_to_hyb4.json") as f:
        reg2 = json.load(f)
    with open(PROJECT_ROOT / "python_results/module3/registration_hyb3_to_hyb4.json") as f:
        reg3 = json.load(f)

    return df_int, df_bc, labels, crop, reg2, reg3


def load_hyb4_channels():
    return {
        "Ch1_AF647": np.load(PROJECT_ROOT / "python_results/module1/hyb4_crop_Ch1_AF647.npy"),
        "Ch2_AF590": np.load(PROJECT_ROOT / "python_results/module1/hyb4_crop_Ch2_AF590.npy"),
        "Ch3_AF488": np.load(PROJECT_ROOT / "python_results/module1/hyb4_crop_Ch3_AF488.npy"),
    }


def load_icc_channels(icc_path: Path, crop: dict) -> dict:
    """Load ICC_Processed TIF, extract 3 fluorescence channels, crop to Hyb4 extent."""
    y0, x0 = crop["y0"], crop["x0"]
    h, w = crop["crop_h"], crop["crop_w"]
    z_map = {"Ch1_AF647": 3, "Ch2_AF590": 1, "Ch3_AF488": 2}
    img = AICSImage(str(icc_path))
    return {
        ch: img.get_image_data("TCZYX")[0, 0, z][y0:y0+h, x0:x0+w]
        for ch, z in z_map.items()
    }


def compute_snr(df_int: pd.DataFrame) -> pd.DataFrame:
    """
    For each nucleus × round, compute SNR = max_channel / mean_of_other_two.
    High SNR → clear winner; low SNR → ambiguous call.
    """
    rows = []
    for _, row in df_int.iterrows():
        vals = {ch: float(row[ch]) for ch in CHANNELS}
        sorted_vals = sorted(vals.values(), reverse=True)
        max_v = sorted_vals[0]
        second_v = sorted_vals[1] if sorted_vals[1] > 0 else 1.0
        rows.append({
            "nucleus_id": int(row["nucleus_id"]),
            "round": row["round"],
            "snr": max_v / second_v,
        })
    return pd.DataFrame(rows)


def find_dual_high(df_int: pd.DataFrame, pct: int = DUAL_HIGH_PCT) -> dict:
    """
    Identify nuclei that are SIMULTANEOUSLY high in BOTH Ch1_AF647 AND Ch3_AF488
    in each hybridisation round.

    Scientific logic:
      Each cell should carry signal in exactly ONE channel per round (by design).
      Nuclei with BOTH Ch1 and Ch3 elevated above the Nth percentile are candidate
      carry-over artefacts (incomplete washing), autofluorescence hotspots, or
      registration edge effects where the nucleus mask overlaps a neighbour.

    Parameters:
      df_int : Module 5 spot_intensities.csv (one row per nucleus × round)
      pct    : percentile cutoff (default 80)

    Returns:
      dict keyed by round name, each value = {
          "nids"      : set of dual-high nucleus_ids,
          "thresh_ch1": float ADU threshold for Ch1,
          "thresh_ch3": float ADU threshold for Ch3,
      }
    """
    result = {}
    for rnd in ["Hyb2", "Hyb3", "Hyb4"]:
        df_r = df_int[df_int["round"] == rnd].copy()
        if len(df_r) == 0:
            continue
        thresh_ch1 = float(np.percentile(df_r["Ch1_AF647"].values, pct))
        thresh_ch3 = float(np.percentile(df_r["Ch3_AF488"].values, pct))
        dual_mask = (df_r["Ch1_AF647"] > thresh_ch1) & (df_r["Ch3_AF488"] > thresh_ch3)
        nids = set(df_r.loc[dual_mask, "nucleus_id"].values.tolist())
        result[rnd] = {
            "nids":       nids,
            "thresh_ch1": thresh_ch1,
            "thresh_ch3": thresh_ch3,
        }
        logger.info(
            f"  Dual-high [{rnd}]: {len(nids)} nuclei "
            f"(Ch1>{thresh_ch1:.0f} AND Ch3>{thresh_ch3:.0f} ADU, p{pct})"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Channel scatter per round (Ch1 vs Ch3, colored by decoded call)
# ─────────────────────────────────────────────────────────────────────────────

def fig_channel_scatter(df_int: pd.DataFrame, df_bc: pd.DataFrame,
                        dual_high_per_round: dict = None):
    """
    Ch1_AF647 vs Ch3_AF488 scatter per round.
    Each dot = one nucleus, colored by its decoded color for that round.
    Scientific purpose: shows whether channel separation is clean.
    A good call: the argmax channel should be far above the others → elongated cluster.

    If dual_high_per_round is provided (from find_dual_high()):
      - Dashed vertical line at Ch1 p80 threshold
      - Dashed horizontal line at Ch3 p80 threshold
      - Orange hollow rings overlaid on dual-high nuclei
      - Text annotation with dual-high count
    """
    logger.info("Generating M5 channel scatter (with dual-high highlights)...")
    rounds = ["Hyb2", "Hyb3", "Hyb4"]
    col_map = {"Hyb2": "color_hyb2", "Hyb3": "color_hyb3", "Hyb4": "color_hyb4"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "M5 QC — Channel Separation (Ch1_AF647 vs Ch3_AF488)\n"
        "Each dot = one nucleus; color = decoded assignment for that round\n"
        f"Orange rings = dual-high (both Ch1 & Ch3 > p{DUAL_HIGH_PCT}; "
        "likely carry-over / washing artefact)",
        fontsize=11
    )

    for ax, rnd in zip(axes, rounds):
        df_r = df_int[df_int["round"] == rnd].copy()
        color_col = col_map[rnd]
        df_r = df_r.merge(df_bc[["nucleus_id", color_col]], on="nucleus_id", how="left")
        df_r[color_col] = df_r[color_col].fillna("None")

        # --- Base scatter: all nuclei colored by decoded call ---
        for color_label, grp in df_r.groupby(color_col):
            ax.scatter(
                grp["Ch1_AF647"], grp["Ch3_AF488"],
                c=PALETTE.get(color_label, "#95A5A6"),
                s=10, alpha=0.6, label=f"{color_label} (n={len(grp)})"
            )

        # --- Diagonal guide: y=x ---
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], "k--", lw=0.5, alpha=0.3)

        # --- Dual-high overlay ---
        if dual_high_per_round and rnd in dual_high_per_round:
            dh = dual_high_per_round[rnd]
            thresh_ch1 = dh["thresh_ch1"]
            thresh_ch3 = dh["thresh_ch3"]
            nids_dh    = dh["nids"]

            # Threshold lines to delineate top-right quadrant
            ax.axvline(thresh_ch1, color="darkorange", lw=1.2, linestyle="--", alpha=0.85,
                       label=f"Ch1 p{DUAL_HIGH_PCT}={thresh_ch1:.0f} ADU")
            ax.axhline(thresh_ch3, color="saddlebrown", lw=1.2, linestyle="--", alpha=0.85,
                       label=f"Ch3 p{DUAL_HIGH_PCT}={thresh_ch3:.0f} ADU")

            # Orange hollow rings on dual-high nuclei
            df_dh = df_r[df_r["nucleus_id"].isin(nids_dh)]
            if len(df_dh) > 0:
                ax.scatter(
                    df_dh["Ch1_AF647"], df_dh["Ch3_AF488"],
                    facecolors="none", edgecolors="darkorange",
                    linewidths=1.5, s=70, zorder=5,
                    label=f"Dual-high (n={len(df_dh)})"
                )
                # Annotation box in top-right corner
                ax.text(
                    0.97, 0.96, f"dual-high: n={len(df_dh)}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="darkorange",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="darkorange", alpha=0.85)
                )

        ax.set_xlabel("Ch1_AF647 (raw ADU)", fontsize=9)
        ax.set_ylabel("Ch3_AF488 (raw ADU)", fontsize=9)
        ax.set_title(f"{rnd}", fontsize=10)
        ax.legend(fontsize=7, markerscale=2)

    plt.tight_layout()
    out = QC_DIR / "qc_m5_channel_scatter.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1b: Dual-high spatial map (companion to channel scatter)
# ─────────────────────────────────────────────────────────────────────────────

def fig_dual_high_spatial(df_bc: pd.DataFrame, dual_high_per_round: dict):
    """
    Spatial scatter of all nucleus centroids per round (colored by decoded call),
    with dual-high nuclei highlighted as large orange hollow circles.

    Scientific purpose:
      - If dual-high nuclei cluster at the FOV edge → likely registration artefact
        (nucleus mask bleeds into a neighbour carrying a different oligo).
      - If scattered uniformly → autofluorescence hotspot or incomplete washing.
      - Cross-check with Method Y: dual-high cells that have Method Y puncta in
        both channels confirm biological carry-over; those with no puncta confirm
        autofluorescence or background elevation.
    """
    logger.info("Generating dual-high spatial map...")
    rounds  = ["Hyb2", "Hyb3", "Hyb4"]
    col_map = {"Hyb2": "color_hyb2", "Hyb3": "color_hyb3", "Hyb4": "color_hyb4"}

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(
        f"Dual-High Population — Spatial Map  (p{DUAL_HIGH_PCT} threshold, Ch1_AF647 AND Ch3_AF488)\n"
        "Orange circles = dual-high nuclei | Interpretation: edge-cluster → registration; "
        "scatter → carry-over / autofluorescence",
        fontsize=10
    )

    for ax, rnd in zip(axes, rounds):
        color_col = col_map[rnd]
        if color_col not in df_bc.columns:
            ax.set_visible(False)
            continue

        # Background: all nuclei colored by decoded call
        for color_label, grp in df_bc.groupby(color_col):
            ax.scatter(
                grp["centroid_x"], grp["centroid_y"],
                c=PALETTE.get(color_label, "#95A5A6"),
                s=5, alpha=0.45, label=f"{color_label} (n={len(grp)})"
            )

        # Overlay: dual-high nuclei as orange hollow circles
        dh     = dual_high_per_round.get(rnd, {})
        nids_dh = dh.get("nids", set())
        df_dh  = df_bc[df_bc["nucleus_id"].isin(nids_dh)]
        n_dh   = len(df_dh)

        if n_dh > 0:
            ax.scatter(
                df_dh["centroid_x"], df_dh["centroid_y"],
                facecolors="none", edgecolors="darkorange",
                linewidths=2.0, s=90, zorder=6,
                label=f"Dual-high (n={n_dh})"
            )
            # Label individual nucleus IDs for manual inspection (if few)
            if n_dh <= 30:
                for _, row in df_dh.iterrows():
                    ax.annotate(
                        str(int(row["nucleus_id"])),
                        (row["centroid_x"], row["centroid_y"]),
                        fontsize=5, color="darkorange", ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points"
                    )

        ax.set_title(f"{rnd}  |  dual-high: n={n_dh}", fontsize=10)
        ax.set_xlabel("x (px, Hyb4 frame)", fontsize=8)
        ax.set_ylabel("y (px, Hyb4 frame)", fontsize=8)
        ax.invert_yaxis()
        ax.legend(fontsize=7, markerscale=2)

    plt.tight_layout()
    out = QC_DIR / "qc_m5_dual_high_spatial.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: SNR distribution violin per round
# ─────────────────────────────────────────────────────────────────────────────

def fig_snr_distribution(df_int: pd.DataFrame):
    """
    SNR = max_channel / 2nd_max_channel, per nucleus per round.
    SNR < 1.5 → the top two channels are nearly equal → unreliable argmax call.
    SNR > 3   → clear dominant channel → high-confidence call.
    """
    logger.info("Generating M5 SNR distribution...")
    df_snr = compute_snr(df_int)

    rounds = ["Hyb2", "Hyb3", "Hyb4"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("M5 QC — SNR Distribution (max / 2nd-max channel intensity)\n"
                 "SNR > 3 = high-confidence call | SNR < 1.5 = ambiguous", fontsize=11)

    # Violin per round
    ax = axes[0]
    data = [df_snr[df_snr["round"] == rnd]["snr"].clip(0, 20).values for rnd in rounds]
    parts = ax.violinplot(data, positions=range(len(rounds)), showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#5DADE2")
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels(rounds)
    ax.axhline(1.5, color="red", lw=1, linestyle="--", label="SNR=1.5 (ambiguous)")
    ax.axhline(3.0, color="green", lw=1, linestyle="--", label="SNR=3.0 (confident)")
    ax.set_ylabel("SNR (max / 2nd-max)", fontsize=9)
    ax.set_title("SNR violin per round", fontsize=9)
    ax.legend(fontsize=8)

    # Fraction of nuclei with SNR > threshold, per round
    ax2 = axes[1]
    thresholds = [1.2, 1.5, 2.0, 3.0, 5.0]
    x = np.arange(len(thresholds))
    width = 0.25
    colors_bar = ["#E74C3C", "#E67E22", "#2ECC71"]
    for i, rnd in enumerate(rounds):
        snr_vals = df_snr[df_snr["round"] == rnd]["snr"].values
        fracs = [np.mean(snr_vals > t) for t in thresholds]
        ax2.bar(x + i * width, fracs, width, label=rnd, color=colors_bar[i], alpha=0.8)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f">{t}" for t in thresholds], fontsize=9)
    ax2.set_ylabel("Fraction of nuclei", fontsize=9)
    ax2.set_title("Fraction of nuclei above SNR threshold", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    out = QC_DIR / "qc_m5_snr_distribution.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Intensity heatmap (sampled nuclei)
# ─────────────────────────────────────────────────────────────────────────────

def fig_intensity_heatmap(df_int: pd.DataFrame, df_bc: pd.DataFrame, n_sample: int = 200):
    """
    Heatmap: rows = nuclei (sampled), columns = round × channel.
    Nuclei sorted by their barcode string for visual grouping.
    Scientific purpose: shows whether barcoded cells have coherent channel patterns.
    """
    logger.info("Generating M5 intensity heatmap...")
    rounds = ["Hyb2", "Hyb3", "Hyb4"]

    # Pivot intensity table: one row per nucleus, columns = round_channel
    pivot_cols = {}
    for rnd in rounds:
        df_r = df_int[df_int["round"] == rnd].set_index("nucleus_id")
        for ch in CHANNELS:
            pivot_cols[f"{rnd}\n{ch.split('_')[1]}"] = df_r[ch]

    df_wide = pd.DataFrame(pivot_cols)

    # Merge barcode for sorting
    df_wide = df_wide.join(df_bc.set_index("nucleus_id")[["barcode", "decoded_ok"]], how="left")
    df_wide = df_wide[df_wide["decoded_ok"] == True].dropna()
    df_wide = df_wide.sort_values("barcode")

    # Sample evenly across barcodes
    per_bc = max(1, n_sample // df_wide["barcode"].nunique())
    sampled = (
        df_wide.groupby("barcode", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_bc), random_state=42))
    )
    if len(sampled) > n_sample:
        sampled = sampled.sample(n_sample, random_state=42)
    sampled = sampled.sort_values("barcode").reset_index(drop=True)

    feat_cols = [c for c in sampled.columns if c not in ["barcode", "decoded_ok"]]
    mat = sampled[feat_cols].values.astype(float)

    # Log-scale for better contrast
    mat_log = np.log10(mat + 1)

    # Normalise per column (0-1)
    col_min = mat_log.min(axis=0, keepdims=True)
    col_max = mat_log.max(axis=0, keepdims=True)
    mat_norm = (mat_log - col_min) / (col_max - col_min + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(mat_norm, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels(feat_cols, fontsize=7, rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_ylabel(f"Nuclei (n={len(sampled)}, sorted by barcode)", fontsize=9)
    ax.set_title(f"M5 QC — Log-normalised Intensity Heatmap\n"
                 f"(decoded_ok nuclei only, sorted by barcode, n={len(sampled)})", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.02, label="log10(intensity+1), col-normalised")

    # Barcode boundary lines
    barcode_counts = sampled["barcode"].value_counts()[sampled["barcode"].unique()]
    cumsum = 0
    for bc, cnt in barcode_counts.items():
        cumsum += cnt
        ax.axhline(cumsum - 0.5, color="white", lw=0.5)

    plt.tight_layout()
    out = QC_DIR / "qc_m5_intensity_heatmap.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Spot overlay per round (THE key validation figure)
# ─────────────────────────────────────────────────────────────────────────────

def make_fluorescence_composite(channels: dict) -> np.ndarray:
    """
    Build a pseudo-colour RGB composite for display:
      R = Ch1_AF647 (Purple → use Red for visibility)
      G = Ch3_AF488 (Yellow → Green)
      B = Ch2_AF590 (Blue → Blue)
    Each channel: log-stretch, clip [0,1].
    """
    def stretch(arr):
        arr_f = arr.astype(np.float32)
        lo, hi = np.percentile(arr_f[arr_f > 0], [2, 99.5]) if arr_f.max() > 0 else (0, 1)
        return np.clip((arr_f - lo) / (hi - lo + 1e-8), 0, 1)

    rgb = np.zeros((*list(channels.values())[0].shape, 3), dtype=np.float32)
    rgb[:, :, 0] = stretch(channels["Ch1_AF647"])   # R = AF647
    rgb[:, :, 1] = stretch(channels["Ch3_AF488"])   # G = AF488
    rgb[:, :, 2] = stretch(channels["Ch2_AF590"])   # B = AF590
    return rgb


def make_mask_color_image(labels: np.ndarray, nid_to_color: dict) -> np.ndarray:
    """
    Build a float32 RGBA image where each nucleus pixel is filled with
    the decoded color (semi-transparent fill).
    nid_to_color: {nucleus_id (int) → hex color string}

    Vectorised via a lookup table indexed by label value → ~100x faster than
    looping over individual nucleus masks.
    """
    from matplotlib.colors import to_rgba

    max_id = int(labels.max())
    # LUT: index 0 = background (transparent), 1..max_id = nucleus colors
    lut = np.zeros((max_id + 1, 4), dtype=np.float32)
    for nid, hex_color in nid_to_color.items():
        if nid <= max_id:
            r, g, b, _ = to_rgba(hex_color)
            lut[nid] = [r, g, b, 0.55]   # fill alpha

    return lut[labels]   # fancy indexing: (H, W) int → (H, W, 4) float


def fig_spot_overlay(labels: np.ndarray, df_bc: pd.DataFrame,
                     channels_hyb4: dict,
                     channels_hyb3: dict, reg3: dict,
                     channels_hyb2: dict, reg2: dict):
    """
    For each round: show fluorescence composite (background) with nucleus masks
    colored by decoded call. Side-by-side: fluorescence | overlay | zoom patch.

    Scientific validation: If the algorithm decodes nucleus X as "Purple",
    that nucleus should be visibly bright in the Red (AF647) channel of the
    composite. Mismatches indicate registration errors or mis-calls.
    """
    logger.info("Generating M6 spot overlay figures...")

    ds = 4      # downsample factor for full-image panels
    zoom_size = 400  # pixels (downsampled) for zoom patch

    round_info = [
        ("Hyb4", channels_hyb4, 0.0, 0.0,  "color_hyb4"),
        ("Hyb3", channels_hyb3, reg3["dy"], reg3["dx"], "color_hyb3"),
        ("Hyb2", channels_hyb2, reg2["dy"], reg2["dx"], "color_hyb2"),
    ]

    for rnd, channels, dy, dx, color_col in round_info:
        logger.info(f"  Building overlay for {rnd}...")

        # --- Fluorescence composite (bring into Hyb4 space if needed) ---
        if abs(dy) > 0.5 or abs(dx) > 0.5:
            # Shift HybN fluorescence into Hyb4 coordinate frame
            channels_aligned = {
                ch: ndimage_shift(arr.astype(np.float32), shift=(dy, dx),
                                  order=1, mode="constant", cval=0)
                for ch, arr in channels.items()
            }
        else:
            channels_aligned = channels

        fluor = make_fluorescence_composite(channels_aligned)   # (H, W, 3) float32

        # --- Nucleus mask: shift to match fluorescence if needed ---
        # labels are in Hyb4 space; fluorescence is already in Hyb4 space → no shift needed
        labels_use = labels

        # --- Build color lookup: nucleus_id → decoded color ---
        nid_to_color = {}
        for _, row in df_bc.iterrows():
            nid = int(row["nucleus_id"])
            color_label = row[color_col] if pd.notna(row[color_col]) else "None"
            nid_to_color[nid] = PALETTE.get(color_label, PALETTE["None"])

        # --- Build RGB color LUT for boundary pixels ---
        # Maps nucleus_id → RGB float (no fill, boundaries only)
        from matplotlib.colors import to_rgb
        max_id = int(labels_use.max())
        lut_rgb = np.zeros((max_id + 1, 3), dtype=np.float32)
        for nid, hex_color in nid_to_color.items():
            if nid <= max_id:
                lut_rgb[nid] = to_rgb(hex_color)

        # --- Downsample for display ---
        fluor_ds  = fluor[::ds, ::ds]
        labels_ds = labels_use[::ds, ::ds].astype(np.int32)

        # No fill — start from pure fluorescence composite
        overlay_ds = fluor_ds.copy()

        # Draw 1-px inner boundary outline, colored by decoded call
        # mode='inner' keeps outlines inside each nucleus → adjacent same-color
        # nuclei still have a visible gap between them.
        boundary_ds   = find_boundaries(labels_ds, mode="inner")
        boundary_nids = labels_ds[boundary_ds]       # nucleus ID at each boundary px
        overlay_ds[boundary_ds] = lut_rgb[boundary_nids]  # color = decoded call

        # --- Choose a representative zoom region (centre of image) ---
        H_ds, W_ds = fluor_ds.shape[:2]
        cy, cx = H_ds // 2, W_ds // 2
        hy, hx = zoom_size // 2, zoom_size // 2
        y1 = max(0, cy - hy); y2 = min(H_ds, cy + hy)
        x1 = max(0, cx - hx); x2 = min(W_ds, cx + hx)

        # --- Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        fig.suptitle(
            f"M6 QC — Spot Overlay: {rnd}\n"
            f"Background: fluorescence composite (R=AF647, G=AF488, B=AF590)\n"
            f"Nucleus outlines: decoded color  |  "
            f"Purple=AF647, Yellow=AF488, Blue=AF590, Gray=None",
            fontsize=10
        )

        axes[0].imshow(fluor_ds)
        axes[0].set_title(f"Fluorescence composite\n{rnd} (R=AF647, G=AF488, B=AF590)",
                          fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(overlay_ds)
        axes[1].set_title(f"Overlay: decoded color outlines\n{rnd}", fontsize=9)
        axes[1].axis("off")
        # Rectangle to mark zoom region
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), x2-x1, y2-y1,
                         linewidth=1.5, edgecolor="white", facecolor="none")
        axes[1].add_patch(rect)

        axes[2].imshow(overlay_ds[y1:y2, x1:x2])
        axes[2].set_title(f"Zoom (centre region)\n{rnd}", fontsize=9)
        axes[2].axis("off")

        # Legend
        handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in PALETTE.items()]
        axes[2].legend(handles=handles, loc="lower right", fontsize=8,
                       framealpha=0.8, title="Decoded color")

        plt.tight_layout()
        out = QC_DIR / f"qc_m6_spot_overlay_{rnd}.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"    Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Barcode count bar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_barcode_counts(df_bc: pd.DataFrame):
    """
    Horizontal bar chart of nuclei per barcode (decoded_ok only).
    Scientific purpose: the ratio of cells per barcode should reflect
    the expected cell pool mixing ratios. A dominant barcode that's
    unexpected → contamination or a segmentation artefact.
    """
    logger.info("Generating M6 barcode count chart...")
    counts = df_bc[df_bc["decoded_ok"]]["barcode"].value_counts()
    top = counts.head(20)

    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.4)))
    colors = []
    for bc in top.index:
        parts = bc.split("-")
        # Color by Hyb4 color (last component)
        colors.append(PALETTE.get(parts[-1], "#BDC3C7"))

    bars = ax.barh(range(len(top)), top.values, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of nuclei", fontsize=10)
    ax.set_title(f"M6 QC — Top Barcodes (decoded_ok nuclei only)\n"
                 f"Total decoded: {df_bc['decoded_ok'].sum()} / {len(df_bc)}\n"
                 f"Bar color = Hyb4 round color", fontsize=10)

    for bar, val in zip(bars, top.values):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2, str(val),
                va="center", fontsize=8)

    plt.tight_layout()
    out = QC_DIR / "qc_m6_barcode_counts.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Decoding confidence spatial map
# ─────────────────────────────────────────────────────────────────────────────

def fig_confidence_map(df_int: pd.DataFrame, df_bc: pd.DataFrame):
    """
    Spatial scatter: centroid of each nucleus, colored by mean SNR across all rounds.
    Marker size scales with SNR.
    Scientific purpose: low-confidence regions may indicate registration issues
    or areas with less reliable FISH signal (e.g., edge of FOV).
    """
    logger.info("Generating M6 confidence spatial map...")

    df_snr = compute_snr(df_int)
    df_mean_snr = df_snr.groupby("nucleus_id")["snr"].mean().reset_index()
    df_mean_snr.columns = ["nucleus_id", "mean_snr"]

    df_plot = df_bc.merge(df_mean_snr, on="nucleus_id", how="left")
    df_plot["mean_snr"] = df_plot["mean_snr"].clip(0, 10)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("M6 QC — Decoding Confidence Spatial Map\n"
                 "SNR = max_channel / 2nd_max  |  Left: mean SNR  |  Right: decoded_ok status",
                 fontsize=11)

    # Left: SNR heatmap
    sc = axes[0].scatter(
        df_plot["centroid_x"], df_plot["centroid_y"],
        c=df_plot["mean_snr"], cmap="RdYlGn",
        s=8, alpha=0.8, vmin=1, vmax=8
    )
    plt.colorbar(sc, ax=axes[0], label="Mean SNR (all rounds)", fraction=0.03)
    axes[0].set_title("Mean SNR per nucleus", fontsize=10)
    axes[0].set_xlabel("x (px, Hyb4 frame)", fontsize=8)
    axes[0].set_ylabel("y (px, Hyb4 frame)", fontsize=8)
    axes[0].invert_yaxis()

    # Right: decoded_ok spatial map
    for ok, grp in df_plot.groupby("decoded_ok"):
        color = "#2ECC71" if ok else "#E74C3C"
        label = f"decoded_ok=True (n={len(grp)})" if ok else f"decoded_ok=False (n={len(grp)})"
        axes[1].scatter(grp["centroid_x"], grp["centroid_y"],
                        c=color, s=8, alpha=0.7, label=label)
    axes[1].set_title("decoded_ok spatial distribution", fontsize=10)
    axes[1].set_xlabel("x (px, Hyb4 frame)", fontsize=8)
    axes[1].set_ylabel("y (px, Hyb4 frame)", fontsize=8)
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8, markerscale=2)

    plt.tight_layout()
    out = QC_DIR / "qc_m6_confidence_map.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Enhanced QC — Module 5 & 6: START")
    logger.info("=" * 60)

    # Check prerequisites
    for p in [
        PROJECT_ROOT / "python_results/module5/spot_intensities.csv",
        PROJECT_ROOT / "python_results/module6/barcodes.csv",
        PROJECT_ROOT / "python_results/module4/nucleus_labels.npy",
    ]:
        if not p.exists():
            logger.error(f"Missing: {p}"); sys.exit(1)

    df_int, df_bc, labels, crop, reg2, reg3 = load_all_data()

    # ── Dual-high population detection (done once, shared across figures) ────
    logger.info("─" * 40)
    logger.info(f"Computing dual-high population (Ch1_AF647 AND Ch3_AF488 > p{DUAL_HIGH_PCT})")
    dual_high = find_dual_high(df_int)

    # Save dual-high nucleus IDs to CSV for manual biological inspection
    dh_rows = []
    for rnd, dh in dual_high.items():
        for nid in sorted(dh["nids"]):
            dh_rows.append({
                "round":      rnd,
                "nucleus_id": int(nid),
                "thresh_ch1": dh["thresh_ch1"],
                "thresh_ch3": dh["thresh_ch3"],
            })
    if dh_rows:
        df_dh_export = pd.DataFrame(dh_rows)
        dh_csv_path  = QC_DIR / "dual_high_nucleus_ids.csv"
        df_dh_export.to_csv(str(dh_csv_path), index=False)
        logger.info(f"  Dual-high IDs saved → {dh_csv_path.name}  ({len(dh_rows)} rows)")

    # M5 QC figures
    logger.info("─" * 40)
    logger.info("M5 QC figures")
    fig_channel_scatter(df_int, df_bc, dual_high_per_round=dual_high)
    fig_dual_high_spatial(df_bc, dual_high)
    fig_snr_distribution(df_int)
    fig_intensity_heatmap(df_int, df_bc)

    # M6 QC figures
    logger.info("─" * 40)
    logger.info("M6 QC figures")
    logger.info("Loading fluorescence channels (this may take ~10 s)...")

    channels_hyb4 = load_hyb4_channels()

    hyb3_path = PROJECT_ROOT / "IMAGES/20260123_psy_redo_3 hyb3_B7-20x-3x3-Z-FOVB_ICC_Processed001.tif"
    hyb2_path = PROJECT_ROOT / "IMAGES/20260123_psy_redo_3 hyb2_B7-20x-3x3-Z-FOVB_ICC_Processed001.tif"
    channels_hyb3 = load_icc_channels(hyb3_path, crop)
    channels_hyb2 = load_icc_channels(hyb2_path, crop)

    fig_spot_overlay(labels, df_bc, channels_hyb4, channels_hyb3, reg3, channels_hyb2, reg2)
    fig_barcode_counts(df_bc)
    fig_confidence_map(df_int, df_bc)

    logger.info("=" * 60)
    logger.info("Enhanced QC COMPLETE")
    logger.info(f"All figures saved to: python_results/qc/")
    logger.info("=" * 60)

    print("\n" + "=" * 50)
    print("QC COMPLETE — figures saved to python_results/qc/")
    print("=" * 50)
    print("  M5 QC:")
    print("    qc_m5_channel_scatter.png    — channel separation + dual-high overlay")
    print("    qc_m5_dual_high_spatial.png  — spatial map of dual-high nuclei")
    print("    qc_m5_snr_distribution.png   — call confidence (max/2nd-max)")
    print("    qc_m5_intensity_heatmap.png  — nucleus × channel intensity")
    print("    dual_high_nucleus_ids.csv    — nucleus IDs for manual inspection")
    print("  M6 QC:")
    print("    qc_m6_spot_overlay_Hyb{2,3,4}.png — fluorescence + decoded overlay + boundaries")
    print("    qc_m6_barcode_counts.png     — top barcode counts")
    print("    qc_m6_confidence_map.png     — SNR spatial distribution")
    print("=" * 50)


if __name__ == "__main__":
    main()
