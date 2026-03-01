"""
Module 5 — Spot Calling (Method Y: Ronan-style Puncta Area)
============================================================
For every nucleus × every hybridization round × every fluorescence channel,
apply a per-nucleus adaptive threshold and measure total surviving puncta area.

Scientific Logic (Method Y — Ronan's method):
    Each nucleus receives its OWN intensity threshold, computed from the
    pixel statistics within that nucleus:

        thresh = min(cell_mean + N×cell_std, 65 000)   N=6 by default

    This adaptive approach accounts for per-cell background differences
    that arise from:
      - uneven illumination across the FOV
      - variable nuclear auto-fluorescence
      - residual carry-over signal from previous rounds

    After thresholding:
      1. Connected-component label the binary image (skimage.measure.label).
      2. Remove components < 10 px (noise filter, skimage.morphology.remove_small_objects).
      3. Signal = total area of surviving puncta (px²).

    Interpretation:
      - area = 0   → no punctum detected in that channel for this nucleus
      - area > 0   → at least one punctum candidate survived the threshold + filter

    The method DOES NOT require explicit spot fitting or Gaussian modelling,
    making it robust to non-round puncta morphology.

Comparison with Method X (max pixel intensity):
    Method X:  fast, no morphology assumptions; susceptible to single-pixel hot pixels.
    Method Y:  slower, adaptive background; more tolerant of diffuse puncta.
    If both methods agree on which channel is the argmax → high-confidence call.
    Disagreement → flag for manual review.

Coordinate Mapping:
    Nucleus labels are in Hyb4 space. The same shift convention as Method X is
    used: apply shift (−dy, −dx) to the label image before sampling HybN.

Inputs:
    - module5_spot_calling.yaml (same config as Method X)
    - Hyb4 fluorescence crops (.npy), Hyb2/Hyb3 ICC_Processed TIFs

Output:
    python_results/module5/spot_intensities_methodY.csv
      Columns: nucleus_id, round, Ch1_AF647, Ch2_AF590, Ch3_AF488
      Values:  puncta area (px²) — comparable across channels within one row
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from aicsimageio import AICSImage
from loguru import logger
from scipy.ndimage import shift as ndimage_shift
from skimage import measure, morphology


class MethodYCaller:
    """
    Extracts total puncta area per nucleus per channel per round (Method Y).

    Threshold per nucleus per channel:
        thresh = min(cell_mean + n_sigma × cell_std, max_thresh_adu)
    Morphology:  label → remove_small_objects(min_size)
    Signal:      total surviving area (px²)
    """

    def __init__(self, config_path: str, project_root: str = "."):
        self.project_root = Path(project_root)
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.results_dir = self.project_root / self.cfg["output"]["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

        coords_path = self.project_root / self.cfg["input"]["crop_coords_path"]
        with open(coords_path) as f:
            self.crop_coords = json.load(f)

        # Method Y parameters (not in YAML — using research defaults)
        self.n_sigma         = 6.0      # threshold = mean + N×std
        self.max_thresh_adu  = 65000.0  # guard against div-by-zero at saturation
        self.min_puncta_size = 10       # px — discard components smaller than this

        logger.info(f"MethodYCaller initialized | project_root={self.project_root}")
        logger.info(f"  Threshold: cell_mean + {self.n_sigma}σ, cap={self.max_thresh_adu:.0f} ADU")
        logger.info(f"  Min puncta size: {self.min_puncta_size} px")

    # ------------------------------------------------------------------
    # Data loaders (identical interface to SpotCaller)
    # ------------------------------------------------------------------

    def load_nucleus_data(self):
        labels_path = self.project_root / self.cfg["input"]["nucleus_labels_path"]
        props_path  = self.project_root / self.cfg["input"]["nucleus_props_path"]
        labels      = np.load(str(labels_path))
        df_props    = pd.read_csv(str(props_path))
        logger.info(f"  Labels shape: {labels.shape}, nuclei: {int(labels.max())}")
        return labels, df_props

    def load_hyb4_channels(self) -> dict:
        ch_map = {
            "Ch1_AF647": "hyb4_ch1_af647_path",
            "Ch2_AF590": "hyb4_ch2_af590_path",
            "Ch3_AF488": "hyb4_ch3_af488_path",
        }
        channels = {}
        for ch_name, cfg_key in ch_map.items():
            path = self.project_root / self.cfg["input"][cfg_key]
            channels[ch_name] = np.load(str(path))
        return channels

    def load_icc_channels(self, icc_path: Path, label: str) -> dict:
        y0      = self.crop_coords["y0"]
        x0      = self.crop_coords["x0"]
        crop_h  = self.crop_coords["crop_h"]
        crop_w  = self.crop_coords["crop_w"]
        icc_ch  = self.cfg["processing"]["icc_channels"]

        logger.info(f"  Loading {label} ICC_Processed: {icc_path.name}")
        img = AICSImage(str(icc_path))
        channels = {}
        for ch_name, z_idx in icc_ch.items():
            arr_full = img.get_image_data("TCZYX")[0, 0, z_idx]
            channels[ch_name] = arr_full[y0:y0 + crop_h, x0:x0 + crop_w]
        return channels

    def load_registration(self, reg_path: Path) -> tuple:
        with open(reg_path) as f:
            reg = json.load(f)
        return float(reg["dy"]), float(reg["dx"])

    # ------------------------------------------------------------------
    # Method Y core
    # ------------------------------------------------------------------

    def sample_area_per_nucleus(self, labels: np.ndarray, channels: dict,
                                dy: float, dx: float, label: str) -> pd.DataFrame:
        """
        Adaptive-threshold puncta area per nucleus per channel.

        For each nucleus (bounding-box crop):
          1. Pixel values within nucleus mask → compute mean, std.
          2. thresh = min(mean + n_sigma × std, max_thresh_adu).
          3. Binary = (arr > thresh) & nucleus_mask.
          4. Label CCs; remove components < min_puncta_size px.
          5. area = cleaned.sum().

        Returns:
          DataFrame [nucleus_id, round, Ch1_AF647, Ch2_AF590, Ch3_AF488]
          Values = puncta area (px²).
        """
        logger.info(f"  Method Y sampling {label}: shift=({dy:+.1f}, {dx:+.1f}) px")

        # Shift labels into HybN coordinate frame (same convention as Method X)
        if abs(dy) > 0.01 or abs(dx) > 0.01:
            labels_shifted = ndimage_shift(
                labels.astype(np.float32), shift=(-dy, -dx),
                order=0, mode="constant", cval=0,
            ).astype(np.int32)
        else:
            labels_shifted = labels.astype(np.int32)

        # Precompute bounding boxes for all nuclei (fast regionprops scan)
        props    = measure.regionprops(labels_shifted)
        bbox_map = {p.label: p.bbox for p in props}   # {nid: (r0, c0, r1, c1)}

        unique_ids = sorted(bbox_map.keys())
        logger.info(f"    {len(unique_ids)} nuclei to process")

        records = []
        for ch_name, arr in channels.items():
            arr_f = arr.astype(np.float64)

            for nid in unique_ids:
                r0, c0, r1, c1 = bbox_map[nid]

                # Crop to nucleus bounding box → small array (typical ~50×50 px)
                local_labels = labels_shifted[r0:r1, c0:c1]
                local_arr    = arr_f[r0:r1, c0:c1]
                local_mask   = (local_labels == nid)

                pixel_vals = local_arr[local_mask]

                if len(pixel_vals) < 5:
                    area = 0
                else:
                    cell_mean = float(pixel_vals.mean())
                    cell_std  = float(pixel_vals.std())
                    thresh    = min(cell_mean + self.n_sigma * cell_std,
                                   self.max_thresh_adu)

                    # Binary: above threshold AND within nucleus
                    binary = (local_arr > thresh) & local_mask

                    if not binary.any():
                        area = 0
                    else:
                        labeled_cc = measure.label(binary, connectivity=2)
                        cleaned    = morphology.remove_small_objects(
                            labeled_cc > 0,
                            min_size=self.min_puncta_size,
                            connectivity=2,
                        )
                        area = int(cleaned.sum())

                records.append({
                    "nucleus_id":  int(nid),
                    "channel":     ch_name,
                    "puncta_area": area,
                })

        df = pd.DataFrame(records)
        df_wide = df.pivot(index="nucleus_id", columns="channel", values="puncta_area")
        df_wide.columns.name = None
        df_wide = df_wide.reset_index()

        for col in ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]:
            if col not in df_wide.columns:
                df_wide[col] = 0
        df_wide["round"] = label

        # Sanity check log
        n_any = (df_wide[["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]] > 0).any(axis=1).sum()
        logger.info(
            f"    {label}: {len(df_wide)} nuclei | "
            f"{n_any} ({100*n_any/max(len(df_wide),1):.0f}%) have ≥1 channel with punctum"
        )
        for ch in ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]:
            n_pos   = (df_wide[ch] > 0).sum()
            pos_val = df_wide.loc[df_wide[ch] > 0, ch]
            mean_a  = pos_val.mean() if len(pos_val) > 0 else 0.0
            logger.info(f"    {ch}: {n_pos} nuclei with puncta (mean area={mean_a:.1f} px²)")

        return df_wide[["nucleus_id", "round", "Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Run Method Y for Hyb2, Hyb3, and Hyb4.

        Returns:
          df_all (pd.DataFrame): nucleus_id × round × channel puncta areas (px²)
        """
        logger.info("=" * 60)
        logger.info("Module 5 — Spot Calling (Method Y): START")
        logger.info("=" * 60)
        t_total = time.time()

        labels, df_props = self.load_nucleus_data()

        round_dfs = []

        # Hyb4 (reference frame — no shift)
        logger.info("[MethodY] Processing Hyb4")
        hyb4_ch = self.load_hyb4_channels()
        df_hyb4 = self.sample_area_per_nucleus(labels, hyb4_ch, 0.0, 0.0, "Hyb4")
        round_dfs.append(df_hyb4)

        # Hyb3
        logger.info("[MethodY] Processing Hyb3")
        hyb3_icc_path = self.project_root / self.cfg["input"]["hyb3_icc_path"]
        reg_hyb3_path = self.project_root / self.cfg["input"]["reg_hyb3_path"]
        dy3, dx3      = self.load_registration(reg_hyb3_path)
        hyb3_ch       = self.load_icc_channels(hyb3_icc_path, "Hyb3")
        df_hyb3       = self.sample_area_per_nucleus(labels, hyb3_ch, dy3, dx3, "Hyb3")
        round_dfs.append(df_hyb3)

        # Hyb2
        logger.info("[MethodY] Processing Hyb2")
        hyb2_icc_path = self.project_root / self.cfg["input"]["hyb2_icc_path"]
        reg_hyb2_path = self.project_root / self.cfg["input"]["reg_hyb2_path"]
        dy2, dx2      = self.load_registration(reg_hyb2_path)
        hyb2_ch       = self.load_icc_channels(hyb2_icc_path, "Hyb2")
        df_hyb2       = self.sample_area_per_nucleus(labels, hyb2_ch, dy2, dx2, "Hyb2")
        round_dfs.append(df_hyb2)

        df_all = pd.concat(round_dfs, ignore_index=True)
        logger.info(f"Combined Method Y table: {len(df_all)} rows "
                    f"({df_all['nucleus_id'].nunique()} nuclei × 3 rounds)")

        out_path = self.results_dir / "spot_intensities_methodY.csv"
        df_all.to_csv(str(out_path), index=False)
        logger.info(f"Method Y intensities saved → {out_path.name}")

        elapsed = time.time() - t_total
        logger.info("=" * 60)
        logger.info(f"Module 5 — Spot Calling (Method Y): COMPLETE in {elapsed:.1f}s")
        logger.info("=" * 60)

        return df_all
