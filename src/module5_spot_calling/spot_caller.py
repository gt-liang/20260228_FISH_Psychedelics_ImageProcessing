"""
Module 5 — Spot Calling (Method X: Max Pixel Intensity per Nucleus)
====================================================================
For every nucleus × every hybridization round × every fluorescence channel,
extract the maximum pixel intensity within the nucleus mask.

Scientific Logic (Method X):
    Each cell pool has exactly ONE fluorescent spot in ONE channel per round.
    We do not need to detect spots explicitly. Instead, we sample the max
    pixel intensity within each nucleus for each channel — the channel with
    the highest max intensity is the one carrying the real spot.

    This approach is robust to:
      - Exact spot morphology (no size/shape assumptions)
      - Moderate background differences across channels
    It can be fooled by:
      - Very bright debris or carry-over signal (addressed in Module 6 by
        a background threshold: if max < threshold → None)

Coordinate Mapping:
    Nucleus labels are in Hyb4 space. For Hyb2 and Hyb3, cells have
    shifted positions (stage drift, ~15–40 px). To sample correctly:
      - From M3: shift (dy, dx) = how much to shift HybN to align to Hyb4.
      - To sample from HybN at the position corresponding to a Hyb4 nucleus:
        we apply shift (-dy, -dx) to the nucleus mask before sampling.
      - Equivalently: for a nucleus centroid (y, x) in Hyb4 space,
        sample from HybN at (y + dy, x + dx).

Inputs:
    - Hyb4 fluorescence crops (M1): hyb4_crop_Ch{1,2,3}.npy
    - Hyb2/Hyb3 ICC_Processed TIFs (full 20x tiled)
    - M3 registration offsets: registration_hyb{2,3}_to_hyb4.json
    - Nucleus labels: nucleus_labels.npy  (M4)
    - Crop coordinates: crop_coords.json  (M1)

Output:
    - spot_intensities.csv: columns = nucleus_id, round, ch1_af647, ch2_af590, ch3_af488
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from aicsimageio import AICSImage
from loguru import logger
from scipy import ndimage as sp_ndimage
from scipy.ndimage import shift as ndimage_shift


class SpotCaller:
    """Extracts max-pixel-intensity per nucleus per channel per round (Method X)."""

    def __init__(self, config_path: str, project_root: str = "."):
        self.project_root = Path(project_root)
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.results_dir = self.project_root / self.cfg["output"]["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load M1 crop coordinates (needed to crop Hyb2/Hyb3 ICC images)
        coords_path = self.project_root / self.cfg["input"]["crop_coords_path"]
        with open(coords_path) as f:
            self.crop_coords = json.load(f)

        logger.info(f"SpotCaller initialized | project_root={self.project_root}")
        logger.info(f"Results dir: {self.results_dir}")

    # ------------------------------------------------------------------
    # Step 1: Load nucleus labels and properties
    # ------------------------------------------------------------------
    def load_nucleus_data(self):
        """Load M4 label image and nucleus properties CSV."""
        labels_path = self.project_root / self.cfg["input"]["nucleus_labels_path"]
        props_path = self.project_root / self.cfg["input"]["nucleus_props_path"]

        logger.info(f"[M5 Step 1] Loading nucleus labels: {labels_path.name}")
        labels = np.load(str(labels_path))
        df_props = pd.read_csv(str(props_path))

        logger.info(f"  Labels shape: {labels.shape}, unique nuclei: {int(labels.max())}")
        logger.info(f"  Properties: {len(df_props)} rows")
        return labels, df_props

    # ------------------------------------------------------------------
    # Step 2: Load fluorescence images for all rounds
    # ------------------------------------------------------------------
    def load_hyb4_channels(self) -> dict:
        """
        Load Hyb4 fluorescence channels (already cropped by M1).
        Returns dict: {"Ch1_AF647": arr, "Ch2_AF590": arr, "Ch3_AF488": arr}
        """
        ch_map = {
            "Ch1_AF647": "hyb4_ch1_af647_path",
            "Ch2_AF590": "hyb4_ch2_af590_path",
            "Ch3_AF488": "hyb4_ch3_af488_path",
        }
        channels = {}
        for ch_name, cfg_key in ch_map.items():
            path = self.project_root / self.cfg["input"][cfg_key]
            arr = np.load(str(path))
            channels[ch_name] = arr
            logger.info(f"  Hyb4 {ch_name}: shape={arr.shape}, "
                        f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
        return channels

    def load_icc_channels(self, icc_path: Path, label: str) -> dict:
        """
        Load fluorescence channels from a full ICC_Processed TIF,
        crop to Hyb4 spatial extent using M1 crop coordinates.

        Returns dict: {"Ch1_AF647": arr, "Ch2_AF590": arr, "Ch3_AF488": arr}
        """
        y0 = self.crop_coords["y0"]
        x0 = self.crop_coords["x0"]
        crop_h = self.crop_coords["crop_h"]
        crop_w = self.crop_coords["crop_w"]
        icc_ch = self.cfg["processing"]["icc_channels"]

        logger.info(f"  Loading {label} ICC_Processed: {icc_path.name}")
        img = AICSImage(str(icc_path))

        channels = {}
        for ch_name, z_idx in icc_ch.items():
            arr_full = img.get_image_data("TCZYX")[0, 0, z_idx]
            arr_crop = arr_full[y0:y0 + crop_h, x0:x0 + crop_w]
            channels[ch_name] = arr_crop
            logger.info(f"    {ch_name} (Z={z_idx}): shape={arr_crop.shape}, "
                        f"min={arr_crop.min()}, max={arr_crop.max()}, mean={arr_crop.mean():.1f}")
        return channels

    def load_registration(self, reg_path: Path) -> tuple:
        """Load M3 registration result. Returns (dy, dx)."""
        with open(reg_path) as f:
            reg = json.load(f)
        return float(reg["dy"]), float(reg["dx"])

    # ------------------------------------------------------------------
    # Step 3: Sample max intensity per nucleus per channel
    # ------------------------------------------------------------------
    def sample_max_per_nucleus(self, labels: np.ndarray, channels: dict,
                               dy: float, dx: float, label: str) -> pd.DataFrame:
        """
        For each nucleus in `labels` (Hyb4 space), sample the max pixel
        intensity in each channel and compute a per-nucleus background estimate.

        Background subtraction (per nucleus per channel):
            background = median of ALL pixel values within the nucleus mask
            corrected  = max(max_intensity − background, 0)

            Scientific rationale:
              A punctum covers ~0.7% of mean nucleus area (14 px² / 2113 px²),
              so the median is essentially unaffected by the bright punctum and
              faithfully estimates the "pedestal" (autofluorescence + carry-over
              + non-specific binding). Subtracting this pedestal removes the
              consistent channel bias that can cause mis-assignment.

        Output columns (one row per nucleus):
            Ch1_AF647, Ch2_AF590, Ch3_AF488          — raw max intensities (ADU)
            Ch1_AF647_bg, Ch2_AF590_bg, Ch3_AF488_bg — median background (ADU)
            Ch1_AF647_corr, Ch2_AF590_corr, Ch3_AF488_corr — max − bg, ≥ 0

        Shift convention:
            (dy, dx) from M3 = "shift HybN by this to align to Hyb4"
            → shift the mask by (−dy, −dx) to sample from HybN image.
        """
        logger.info(f"  Sampling {label}: shift=({dy:+.1f}, {dx:+.1f}) px")

        # Shift mask into HybN coordinate frame
        if abs(dy) > 0.01 or abs(dx) > 0.01:
            labels_shifted = ndimage_shift(
                labels.astype(np.float32), shift=(-dy, -dx),
                order=0,   # nearest-neighbour — preserves integer label values
                mode="constant", cval=0,
            ).astype(np.int32)
        else:
            labels_shifted = labels.astype(np.int32)

        unique_ids = np.unique(labels_shifted)
        unique_ids = unique_ids[unique_ids > 0]
        nid_max = int(unique_ids.max())

        # Precompute flattened label array once (shared across channels)
        flat_labels = labels_shifted.ravel()
        nucleus_px_mask = flat_labels > 0    # boolean mask: within any nucleus
        nucleus_labels_flat = flat_labels[nucleus_px_mask]

        # --- Build one record per nucleus (wide format directly — no pivot needed) ---
        # First pass: compute max and background for all channels
        ch_max  = {}   # ch_name → array of length nid_max+1
        ch_bg   = {}   # ch_name → array of length nid_max+1

        for ch_name, arr in channels.items():
            arr_f = arr.astype(np.float64)

            # Max per nucleus (vectorised with np.maximum.at)
            nucleus_arr_flat = arr_f.ravel()[nucleus_px_mask]
            max_vals = np.zeros(nid_max + 1, dtype=np.float64)
            np.maximum.at(max_vals, nucleus_labels_flat, nucleus_arr_flat)
            ch_max[ch_name] = max_vals

            # Median per nucleus (background estimate)
            # scipy.ndimage.median with index= computes over each labeled region
            bg_list = sp_ndimage.median(arr_f, labels=labels_shifted,
                                        index=unique_ids.tolist())
            bg_vals = np.zeros(nid_max + 1, dtype=np.float64)
            for nid, bg in zip(unique_ids, bg_list):
                bg_vals[int(nid)] = float(bg)
            ch_bg[ch_name] = bg_vals

        # Second pass: assemble one record per nucleus
        records = []
        for nid in unique_ids:
            rec = {"nucleus_id": int(nid), "round": label}
            for ch_name in ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]:
                raw  = float(ch_max[ch_name][nid])
                bg   = float(ch_bg[ch_name][nid])
                corr = max(raw - bg, 0.0)
                rec[ch_name]              = raw
                rec[f"{ch_name}_bg"]      = bg
                rec[f"{ch_name}_corr"]    = corr
            records.append(rec)

        df_wide = pd.DataFrame(records)
        raw_cols  = ["Ch1_AF647",      "Ch2_AF590",      "Ch3_AF488"]
        bg_cols   = ["Ch1_AF647_bg",   "Ch2_AF590_bg",   "Ch3_AF488_bg"]
        corr_cols = ["Ch1_AF647_corr", "Ch2_AF590_corr", "Ch3_AF488_corr"]

        logger.info(f"    {label}: {len(df_wide)} nuclei sampled")
        for ch in raw_cols:
            bg_ch = f"{ch}_bg"; corr_ch = f"{ch}_corr"
            logger.info(
                f"    {ch}: raw_max={df_wide[ch].mean():.0f} | "
                f"bg(median)={df_wide[bg_ch].mean():.0f} | "
                f"corr={df_wide[corr_ch].mean():.0f}"
            )

        out_cols = ["nucleus_id", "round"] + raw_cols + bg_cols + corr_cols
        return df_wide[out_cols]

    # ------------------------------------------------------------------
    # Step 3b: Cross-round background correction (all 3 channels)
    # ------------------------------------------------------------------
    def _apply_cross_round_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each nucleus × channel, subtract the per-nucleus minimum corrected
        intensity observed across ALL 3 hybridization rounds.

        Scientific rationale:
            Each cell has genuine FISH signal in at most ONE round per channel.
            The other rounds carry only background: autofluorescence, carry-over,
            and persistent protein fluorescence (e.g., mCherry in Ch2_AF590).
            The cross-round minimum is therefore a conservative, per-nucleus
            estimate of the background pedestal for each channel.

            Two-layer correction:
              Layer 1 (_corr): spatial autofluorescence (per-nucleus median subtraction)
              Layer 2 (_xr):   carry-over + persistent fluorescence (cross-round minimum)

        Formula (per nucleus i, per channel ch):
            xr_bg_i_ch = min( Ch_corr_Hyb2_i, Ch_corr_Hyb3_i, Ch_corr_Hyb4_i )
            Ch_xr_i_rnd = max( Ch_corr_i_rnd − xr_bg_i_ch, 0 )

        New columns added:
            Ch1_AF647_xr_bg, Ch2_AF590_xr_bg, Ch3_AF488_xr_bg  — cross-round bg (ADU)
            Ch1_AF647_xr,    Ch2_AF590_xr,    Ch3_AF488_xr     — doubly-corrected signal
        """
        logger.info("[M5 Step 3b] Cross-round background correction (all 3 channels)...")
        channels  = ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]
        corr_cols = [f"{ch}_corr" for ch in channels]
        rounds    = sorted(df["round"].unique())

        # Pivot: one row per nucleus, columns = Ch_corr_<round>
        pivot_frames = []
        for rnd in rounds:
            df_r = (df[df["round"] == rnd][["nucleus_id"] + corr_cols]
                    .rename(columns={c: f"{c}_{rnd}" for c in corr_cols}))
            pivot_frames.append(df_r.set_index("nucleus_id"))
        df_wide = pd.concat(pivot_frames, axis=1)

        # Per-nucleus minimum across rounds → cross-round background
        xr_bg_series = {}
        for ch in channels:
            rnd_cols = [f"{ch}_corr_{rnd}" for rnd in rounds
                        if f"{ch}_corr_{rnd}" in df_wide.columns]
            xr_bg = df_wide[rnd_cols].min(axis=1)
            xr_bg_series[f"{ch}_xr_bg"] = xr_bg
            logger.info(f"  {ch} cross-round bg: "
                        f"median={xr_bg.median():.0f} | "
                        f"p95={xr_bg.quantile(0.95):.0f} | "
                        f"max={xr_bg.max():.0f} ADU")

        df_xr_bg = pd.DataFrame(xr_bg_series).reset_index()
        df = df.merge(df_xr_bg, on="nucleus_id", how="left")

        for ch in channels:
            df[f"{ch}_xr"] = (df[f"{ch}_corr"] - df[f"{ch}_xr_bg"]).clip(lower=0)

        logger.info("  Cross-round corrected means:")
        for ch in channels:
            logger.info(f"    {ch}_xr: mean={df[f'{ch}_xr'].mean():.0f} | "
                        f"median={df[f'{ch}_xr'].median():.0f} ADU")
        return df

    # ------------------------------------------------------------------
    # Step 4: QC visualization — per-round intensity distributions
    # ------------------------------------------------------------------
    def visualize_result(self, df: pd.DataFrame):
        """
        Generate per-round violin plots for all three correction layers:
        raw, spatially corrected (_corr), and cross-round corrected (_xr).
        """
        logger.info(f"[M5 Step 4] Generating spot calling QC visualization")

        rounds   = sorted(df["round"].unique())
        channels = ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]
        ch_colors = {"Ch1_AF647": "#9B59B6", "Ch2_AF590": "#3498DB", "Ch3_AF488": "#F4D03F"}

        # Determine which column sets are available
        has_corr = "Ch1_AF647_corr" in df.columns
        has_xr   = "Ch1_AF647_xr"   in df.columns

        # One figure per correction layer, side by side
        n_layers = 1 + has_corr + has_xr
        fig, axes = plt.subplots(n_layers, len(rounds),
                                 figsize=(6 * len(rounds), 4 * n_layers),
                                 squeeze=False)

        layer_info = [("raw",  channels,                         "Raw max intensity (ADU)")]
        if has_corr:
            layer_info.append(("corr", [f"{c}_corr" for c in channels], "Spatially corrected (ADU)"))
        if has_xr:
            layer_info.append(("xr",   [f"{c}_xr"   for c in channels], "Cross-round corrected (ADU)"))

        for row_idx, (layer_key, col_names, y_label) in enumerate(layer_info):
            for col_idx, rnd in enumerate(rounds):
                ax = axes[row_idx, col_idx]
                df_round = df[df["round"] == rnd]
                data = [df_round[c].clip(lower=0).dropna().values for c in col_names]
                parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=True)
                for pc, ch in zip(parts["bodies"], channels):
                    pc.set_facecolor(ch_colors[ch])
                    pc.set_alpha(0.7)
                ax.set_xticks([1, 2, 3])
                ax.set_xticklabels(["AF647\n(Purple)", "AF590\n(Blue)", "AF488\n(Yellow)"],
                                   fontsize=8)
                if row_idx == 0:
                    ax.set_title(f"{rnd}  (n={len(df_round)})", fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(y_label, fontsize=8)

        fig.suptitle(
            "Module 5 QC — Max Intensity per Nucleus per Channel\n"
            "Row 1: raw | Row 2: spatial bg subtracted | Row 3: cross-round corrected\n"
            "Signal channel should dominate clearly in the corrected rows",
            fontsize=10
        )
        plt.tight_layout()
        out_path = self.results_dir / "module5_spot_calling_QC.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  QC image saved → {out_path.name}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """
        Run spot calling for Hyb2, Hyb3, and Hyb4.

        Returns:
            df (pd.DataFrame): nucleus_id × round × {Ch1, Ch2, Ch3} max intensities
        """
        logger.info("=" * 60)
        logger.info("Module 5 — Spot Calling: START")
        logger.info("=" * 60)
        t_total = time.time()

        labels, df_props = self.load_nucleus_data()
        logger.info(f"[M5 Step 2] Loading fluorescence channels for all rounds")

        round_dfs = []

        # --- Hyb4 (no shift — reference frame) ---
        logger.info("[M5] Processing Hyb4")
        hyb4_ch = self.load_hyb4_channels()
        df_hyb4 = self.sample_max_per_nucleus(labels, hyb4_ch, dy=0.0, dx=0.0, label="Hyb4")
        round_dfs.append(df_hyb4)

        # --- Hyb3 ---
        logger.info("[M5] Processing Hyb3")
        hyb3_icc_path = self.project_root / self.cfg["input"]["hyb3_icc_path"]
        reg_hyb3_path = self.project_root / self.cfg["input"]["reg_hyb3_path"]
        dy3, dx3 = self.load_registration(reg_hyb3_path)
        hyb3_ch = self.load_icc_channels(hyb3_icc_path, "Hyb3")
        df_hyb3 = self.sample_max_per_nucleus(labels, hyb3_ch, dy=dy3, dx=dx3, label="Hyb3")
        round_dfs.append(df_hyb3)

        # --- Hyb2 ---
        logger.info("[M5] Processing Hyb2")
        hyb2_icc_path = self.project_root / self.cfg["input"]["hyb2_icc_path"]
        reg_hyb2_path = self.project_root / self.cfg["input"]["reg_hyb2_path"]
        dy2, dx2 = self.load_registration(reg_hyb2_path)
        hyb2_ch = self.load_icc_channels(hyb2_icc_path, "Hyb2")
        df_hyb2 = self.sample_max_per_nucleus(labels, hyb2_ch, dy=dy2, dx=dx2, label="Hyb2")
        round_dfs.append(df_hyb2)

        df_all = pd.concat(round_dfs, ignore_index=True)

        logger.info(f"[M5 Step 3] Combined table: {len(df_all)} rows "
                    f"({df_all['nucleus_id'].nunique()} nuclei × 3 rounds)")

        # Cross-round correction (Step 3b) — adds _xr_bg and _xr columns
        df_all = self._apply_cross_round_correction(df_all)

        self.visualize_result(df_all)

        out_path = self.results_dir / self.cfg["output"]["intensity_filename"]
        df_all.to_csv(str(out_path), index=False)
        logger.info(f"[M5 Step 5] Intensities saved → {out_path.name}")

        elapsed = time.time() - t_total
        logger.info("=" * 60)
        logger.info(f"Module 5 — Spot Calling: COMPLETE in {elapsed:.1f}s")
        logger.info(f"  Total rows (nuclei × rounds): {len(df_all)}")
        logger.info("=" * 60)

        return df_all
