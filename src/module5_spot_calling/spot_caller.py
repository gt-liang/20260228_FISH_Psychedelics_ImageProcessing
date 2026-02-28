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
        intensity in each channel, accounting for the (dy, dx) shift between
        HybN and Hyb4.

        Shift convention:
            (dy, dx) from M3 = "shift HybN by this to align to Hyb4"
            → to map a Hyb4 mask pixel to HybN image: add (dy, dx) to coords
            → equivalently: shift the mask by (-dy, -dx) before sampling

        Sampling is done by shifting the entire label image (float shift → nearest
        neighbour remap) and then using np.where per label. This avoids looping
        over individual pixel coordinates and is vectorised over all nuclei at once.
        """
        logger.info(f"  Sampling {label}: shift=({dy:+.1f}, {dx:+.1f}) px")

        # Shift mask into HybN coordinate frame
        if abs(dy) > 0.01 or abs(dx) > 0.01:
            labels_shifted = ndimage_shift(
                labels.astype(np.float32), shift=(-dy, -dx),
                order=0,            # nearest-neighbour — preserves integer label values
                mode="constant", cval=0,
            ).astype(np.uint32)
        else:
            labels_shifted = labels  # < 0.1 px shift → no resampling needed

        unique_ids = np.unique(labels_shifted)
        unique_ids = unique_ids[unique_ids > 0]

        records = []
        for ch_name, arr in channels.items():
            # For each nucleus, find the max pixel in this channel
            # np.maximum.reduceat requires sorted, contiguous labels → use bincount trick
            # Simple approach: flatten mask and image, group by label
            flat_labels = labels_shifted.ravel()
            flat_arr = arr.ravel()

            # Only consider pixels belonging to a nucleus (label > 0)
            nucleus_mask = flat_labels > 0
            nucleus_labels_flat = flat_labels[nucleus_mask]
            nucleus_arr_flat = flat_arr[nucleus_mask]

            # Compute max per label using np.maximum.at
            max_vals = np.zeros(int(unique_ids.max()) + 1, dtype=np.float64)
            np.maximum.at(max_vals, nucleus_labels_flat, nucleus_arr_flat.astype(np.float64))

            for nid in unique_ids:
                records.append({
                    "nucleus_id": int(nid),
                    "channel": ch_name,
                    "max_intensity": float(max_vals[nid]),
                })

        df = pd.DataFrame(records)
        # Pivot to wide format: one row per nucleus, columns per channel
        df_wide = df.pivot(index="nucleus_id", columns="channel", values="max_intensity")
        df_wide.columns.name = None
        df_wide = df_wide.reset_index()
        # Ensure consistent column order
        for col in ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]:
            if col not in df_wide.columns:
                df_wide[col] = np.nan
        df_wide["round"] = label

        logger.info(f"    {label}: {len(df_wide)} nuclei sampled")
        logger.info(f"    Ch1_AF647 — mean={df_wide['Ch1_AF647'].mean():.1f}, "
                    f"max={df_wide['Ch1_AF647'].max():.1f}")
        logger.info(f"    Ch2_AF590 — mean={df_wide['Ch2_AF590'].mean():.1f}, "
                    f"max={df_wide['Ch2_AF590'].max():.1f}")
        logger.info(f"    Ch3_AF488 — mean={df_wide['Ch3_AF488'].mean():.1f}, "
                    f"max={df_wide['Ch3_AF488'].max():.1f}")

        return df_wide[["nucleus_id", "round", "Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]]

    # ------------------------------------------------------------------
    # Step 4: QC visualization — per-round intensity distributions
    # ------------------------------------------------------------------
    def visualize_result(self, df: pd.DataFrame):
        """
        Generate per-round violin plots of max intensity per channel.
        Used to verify that signal channels are clearly above background.
        """
        logger.info(f"[M5 Step 4] Generating spot calling QC visualization")

        rounds = df["round"].unique()
        channels = ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]
        colors = {"Ch1_AF647": "#9B59B6", "Ch2_AF590": "#3498DB", "Ch3_AF488": "#F4D03F"}

        fig, axes = plt.subplots(1, len(rounds), figsize=(6 * len(rounds), 5), sharey=False)
        if len(rounds) == 1:
            axes = [axes]

        for ax, rnd in zip(axes, rounds):
            df_round = df[df["round"] == rnd]
            data = [df_round[ch].dropna().values for ch in channels]
            parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=True)
            for pc, ch in zip(parts["bodies"], channels):
                pc.set_facecolor(colors[ch])
                pc.set_alpha(0.7)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["AF647\n(Ch1)", "AF590\n(Ch2)", "AF488\n(Ch3)"], fontsize=9)
            ax.set_title(f"{rnd}\n(n={len(df_round)} nuclei)", fontsize=10)
            ax.set_ylabel("Max pixel intensity (raw ADU)", fontsize=8)

        fig.suptitle("Module 5 QC — Max Intensity per Nucleus per Channel\n"
                     "Signal channel should be clearly above the other two", fontsize=11)
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
