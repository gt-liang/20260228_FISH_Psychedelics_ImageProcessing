"""
Puncta Detection Disagreement QC Exporter  (v2 – Color + Merged)
=================================================================
For every nucleus where ≥2 methods give different barcode calls,
exports a multi-panel figure for manual visual inspection.

Figure layout per nucleus (3 rounds × 4 columns):
    ┌──────────┬──────────┬──────────┬──────────┐
    │ Hyb4 Ch1 │ Hyb4 Ch2 │ Hyb4 Ch3 │ Hyb4 RGB │
    │ (Purple) │ (Blue)   │ (Yellow) │ (Merged) │
    ├──────────┼──────────┼──────────┼──────────┤
    │ Hyb3 Ch1 │ Hyb3 Ch2 │ Hyb3 Ch3 │ Hyb3 RGB │
    ├──────────┼──────────┼──────────┼──────────┤
    │ Hyb2 Ch1 │ Hyb2 Ch2 │ Hyb2 Ch3 │ Hyb2 RGB │
    └──────────┴──────────┴──────────┴──────────┘
    [Method call table: rows=rounds, cols=methods]

Crop strategy:
    Centroid-based with radius = sqrt(area_px / π) + PAD_PX.
    For HybN, nucleus centroid shifts by registration offset:
        cy_rnd = cy − dy,  cx_rnd = cx − dx
    (because shift_labels applies ndimage_shift(labels, shift=(−dy, −dx)),
     moving the nucleus centroid from (cy, cx) to (cy−dy, cx−dx)).

Merged RGB panel:
    R = Ch1_AF647 (Purple fluorophore)
    G = Ch3_AF488 (Yellow fluorophore)
    B = Ch2_AF590 (Blue fluorophore)
    White nucleus boundary overlay.

Filename convention (sortable by disagreement level):
    nucleus_XXXX_ndiff_N.png
    where N = number of distinct barcode calls (higher = more disagreement)

Output:
    python_results/puncta_comparison/disagreement_crops/
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from loguru import logger
from scipy.ndimage import shift as ndimage_shift
from skimage.segmentation import find_boundaries

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Config ──────────────────────────────────────────────────────────────────
CHANNELS = ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]
CH_LABELS = {
    "Ch1_AF647": "Ch1\n(Purple)",
    "Ch2_AF590": "Ch2\n(Blue)",
    "Ch3_AF488": "Ch3\n(Yellow)",
}

# RGB color vectors per channel (0–1 scale), matching fluorophore display colours
CH_COLORS_RGB = {
    "Ch1_AF647": np.array([0.608, 0.349, 0.714]),  # Purple  #9B59B6
    "Ch2_AF590": np.array([0.204, 0.596, 0.859]),  # Blue    #3498DB
    "Ch3_AF488": np.array([0.957, 0.816, 0.247]),  # Yellow  #F4D03F
}

ROUNDS  = ["Hyb4", "Hyb3", "Hyb2"]   # display order: top = Hyb4 (first imaged)
METHODS = ["X", "Y", "Z", "W", "T", "P"]
METHOD_LABELS_SHORT = {
    "X": "X (maxpx)",
    "Y": "Y (thresh)",
    "Z": "Z (LoG)",
    "W": "W (DoG)",
    "T": "T (TrackPy)",
    "P": "P (peak)",
}
COLOR_DISPLAY = {
    "Purple": "#9B59B6",
    "Blue":   "#3498DB",
    "Yellow": "#F4D03F",
    "None":   "#95A5A6",
}

ICC_CHANNELS = {"Ch1_AF647": 3, "Ch2_AF590": 1, "Ch3_AF488": 2}
PAD_PX = 25      # extra padding around nucleus radius

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
COMP_TABLE  = PROJECT_ROOT / "python_results/puncta_comparison/comparison_table.csv"
OUT_DIR     = PROJECT_ROOT / "python_results/puncta_comparison/disagreement_crops"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")


# ── Image loading ─────────────────────────────────────────────────────────────
def load_all_images():
    """Load all 3 rounds × 3 channels into memory."""
    with open(CROP_COORDS) as f:
        cc = json.load(f)
    y0, x0, h, w = cc["y0"], cc["x0"], cc["crop_h"], cc["crop_w"]

    images = {}   # {round: {ch: 2D array}}

    # Hyb4 — pre-cropped .npy
    logger.info("Loading Hyb4 channels (npy)...")
    images["Hyb4"] = {ch: np.load(str(HYB4_NPY[ch])) for ch in CHANNELS}

    # Hyb3 / Hyb2 — ICC TIF
    for rnd_key, tif_path in [("Hyb3", HYBNPATH["hyb3"]), ("Hyb2", HYBNPATH["hyb2"])]:
        logger.info(f"Loading {rnd_key} ICC_Processed TIF...")
        img  = AICSImage(str(tif_path))
        data = img.get_image_data("TCZYX")[0, 0]   # shape: (5, H, W)
        images[rnd_key] = {
            ch: data[z_idx, y0:y0+h, x0:x0+w]
            for ch, z_idx in ICC_CHANNELS.items()
        }

    return images


def load_registrations():
    regs = {}
    for rnd, path in [("Hyb2", REG_HYB2), ("Hyb3", REG_HYB3)]:
        with open(path) as f:
            d = json.load(f)
        regs[rnd] = (float(d["dy"]), float(d["dx"]))
    regs["Hyb4"] = (0.0, 0.0)
    return regs


def shift_labels(labels, dy, dx):
    if abs(dy) < 0.01 and abs(dx) < 0.01:
        return labels.astype(np.int32)
    return ndimage_shift(
        labels.astype(np.float32), shift=(-dy, -dx),
        order=0, mode="constant", cval=0
    ).astype(np.int32)


def _norm_crop(raw_crop: np.ndarray) -> np.ndarray:
    """Percentile normalisation (1st–99th percentile), returns float [0, 1]."""
    p1, p99 = np.percentile(raw_crop, [1, 99])
    return np.clip((raw_crop.astype(np.float64) - p1) / max(p99 - p1, 1.0), 0.0, 1.0)


# ── Per-nucleus figure ────────────────────────────────────────────────────────
def make_nucleus_figure(
    nid: int,
    n_unique: int,
    images: dict,
    labels: np.ndarray,
    regs: dict,
    row_comp: pd.Series,
    df_props: pd.DataFrame,
) -> plt.Figure:
    """
    Build a 3 rounds × 4 columns figure for one nucleus.

    Columns: Ch1 (Purple) | Ch2 (Blue) | Ch3 (Yellow) | RGB Merged
    Crop:    centroid-based, fixed radius = sqrt(area_px / π) + PAD_PX.
    For HybN: centroid in image = (cy − dy, cx − dx).
    """
    prop_row = df_props[df_props["nucleus_id"] == nid].iloc[0]
    cy = int(prop_row["centroid_y"])
    cx = int(prop_row["centroid_x"])

    # Nucleus crop radius derived from segmented area
    if "area_px" in prop_row.index:
        radius = int(np.sqrt(float(prop_row["area_px"]) / np.pi) + PAD_PX)
    elif "area" in prop_row.index:
        radius = int(np.sqrt(float(prop_row["area"]) / np.pi) + PAD_PX)
    else:
        radius = PAD_PX * 3   # conservative fallback

    H, W = labels.shape

    fig = plt.figure(figsize=(18, 10))
    # 3 image rows + 1 table row; 4 columns (Ch1, Ch2, Ch3, Merged)
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.55],
                          hspace=0.35, wspace=0.12)

    col_headers = [CH_LABELS[ch] for ch in CHANNELS] + ["Merged\n(RGB)"]

    for r_idx, rnd in enumerate(ROUNDS):   # Hyb4, Hyb3, Hyb2
        dy, dx = regs[rnd]

        # ── Centroid in this round's image frame ──────────────────────────
        # shift_labels applies ndimage_shift(labels, shift=(−dy, −dx)):
        #   a point at (cy, cx) in Hyb4 maps to (cy−dy, cx−dx) in HybN.
        cy_rnd = int(round(cy - dy))
        cx_rnd = int(round(cx - dx))

        r0 = max(0, cy_rnd - radius)
        r1 = min(H, cy_rnd + radius)
        c0 = max(0, cx_rnd - radius)
        c1 = min(W, cx_rnd + radius)

        # Nucleus mask / boundary for overlay
        labels_shifted = shift_labels(labels, dy, dx)
        mask_crop = (labels_shifted[r0:r1, c0:c1] == nid)
        boundary  = find_boundaries(mask_crop, mode="inner")

        # Normalise each channel once — reuse for colored panels and merge
        norm_crops = {ch: _norm_crop(images[rnd][ch][r0:r1, c0:c1])
                      for ch in CHANNELS}

        # ── 3 colored channel panels ───────────────────────────────────────
        for c_idx, ch in enumerate(CHANNELS):
            ax    = fig.add_subplot(gs[r_idx, c_idx])
            color = CH_COLORS_RGB[ch]
            norm  = norm_crops[ch]

            # Tinted RGB image: multiply normalised intensity by channel color
            colored = np.stack(
                [norm * color[0], norm * color[1], norm * color[2]], axis=-1
            )
            colored = np.clip(colored, 0.0, 1.0)
            ax.imshow(colored, interpolation="nearest")

            # Cyan nucleus boundary overlay
            bd_rgba = np.zeros((*boundary.shape, 4), dtype=np.float32)
            bd_rgba[boundary] = [0.0, 1.0, 1.0, 0.9]
            ax.imshow(bd_rgba, interpolation="nearest")

            if r_idx == 0:
                ax.set_title(col_headers[c_idx], fontsize=8, pad=3)
            if c_idx == 0:
                ax.set_ylabel(rnd, fontsize=8, labelpad=3)
            ax.set_xticks([]); ax.set_yticks([])

        # ── Merged RGB composite panel (4th column) ────────────────────────
        ax_merge = fig.add_subplot(gs[r_idx, 3])
        merged = np.stack([
            norm_crops["Ch1_AF647"],   # R — Purple (AF647)
            norm_crops["Ch3_AF488"],   # G — Yellow (AF488)
            norm_crops["Ch2_AF590"],   # B — Blue   (AF590)
        ], axis=-1)
        merged = np.clip(merged, 0.0, 1.0)
        ax_merge.imshow(merged, interpolation="nearest")

        # White nucleus boundary overlay on merged panel
        bd_rgba_w = np.zeros((*boundary.shape, 4), dtype=np.float32)
        bd_rgba_w[boundary] = [1.0, 1.0, 1.0, 0.9]
        ax_merge.imshow(bd_rgba_w, interpolation="nearest")

        if r_idx == 0:
            ax_merge.set_title(col_headers[3], fontsize=8, pad=3)
        ax_merge.set_xticks([]); ax_merge.set_yticks([])

    # ── Method call table ─────────────────────────────────────────────────────
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis("off")

    col_labels  = ["Round"] + [METHOD_LABELS_SHORT[m] for m in METHODS]
    table_data  = []
    cell_colors = []   # one row per data row (colLabels handles header row)

    for rnd in ROUNDS:
        row        = [rnd]
        row_colors = ["#DDDDDD"]
        for m in METHODS:
            col  = f"color_{rnd.lower()}_{m}"
            call = str(row_comp.get(col, "?"))
            row.append(call)
            row_colors.append(COLOR_DISPLAY.get(call, "#FFFFFF"))
        table_data.append(row)
        cell_colors.append(row_colors)

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.6)

    fig.suptitle(
        f"Nucleus {nid}  |  centroid=({cx}, {cy})  |  "
        f"{n_unique} distinct barcode calls across {len(METHODS)} methods",
        fontsize=9, fontweight="bold", y=0.98,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("Puncta Disagreement QC v2 (Color + Merged): START")
    logger.info("=" * 60)

    df_comp = pd.read_csv(str(COMP_TABLE))
    bc_cols = [c for c in df_comp.columns if c.startswith("barcode_")]
    df_comp["n_unique"] = df_comp[bc_cols].nunique(axis=1)

    df_disagree = df_comp[df_comp["n_unique"] > 1].copy()
    df_disagree = df_disagree.sort_values("n_unique", ascending=False)
    logger.info(f"Disagreement nuclei: {len(df_disagree)} / {len(df_comp)}")
    logger.info(
        f"  n_unique distribution:\n"
        f"{df_disagree['n_unique'].value_counts().sort_index(ascending=False).to_string()}"
    )

    images   = load_all_images()
    regs     = load_registrations()
    labels   = np.load(str(LABELS_PATH))
    df_props = pd.read_csv(str(PROPS_PATH))

    logger.info(f"Exporting {len(df_disagree)} figures to {OUT_DIR.name}/...")

    # Expand barcode_M → color_hybN_M columns for table colouring
    for m in METHODS:
        bc_col = f"barcode_{m}"
        if bc_col not in df_comp.columns:
            continue
        splits = df_comp[bc_col].str.split("-", expand=True)
        if splits.shape[1] >= 3:
            df_comp[f"color_hyb2_{m}"] = splits[0]
            df_comp[f"color_hyb3_{m}"] = splits[1]
            df_comp[f"color_hyb4_{m}"] = splits[2]
        else:
            for rnd in ["hyb2", "hyb3", "hyb4"]:
                df_comp[f"color_{rnd}_{m}"] = "?"

    # Refresh disagreement subset after column expansion
    df_disagree = df_comp[df_comp["n_unique"] > 1].copy()
    df_disagree = df_disagree.sort_values("n_unique", ascending=False)

    for i, (_, row) in enumerate(df_disagree.iterrows()):
        nid      = int(row["nucleus_id"])
        n_unique = int(row["n_unique"])

        fig = make_nucleus_figure(
            nid=nid,
            n_unique=n_unique,
            images=images,
            labels=labels,
            regs=regs,
            row_comp=row,
            df_props=df_props,
        )

        fname = OUT_DIR / f"nucleus_{nid:04d}_ndiff_{n_unique}.png"
        fig.savefig(str(fname), dpi=120, bbox_inches="tight")
        plt.close(fig)

        if (i + 1) % 50 == 0 or (i + 1) == len(df_disagree):
            logger.info(f"  {i+1}/{len(df_disagree)} saved...")

    logger.info("=" * 60)
    logger.info(f"COMPLETE — {len(df_disagree)} figures in {OUT_DIR}")
    logger.info("Sort by filename descending to review highest-disagreement cells first.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
