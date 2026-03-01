"""
Module 1 — FOV Mapping
======================
Locate the 10x Live FOV inside the 3x3 tiled 20x Hyb4 image using DAPI template matching.

Scientific Logic:
    - The 10x image covers a large physical FOV. To image the same cells at higher
      resolution (20x), the microscope performs a 3x3 tile scan. This tiled 20x image
      is larger than the 10x FOV in pixel space (due to overlap and tiling margins).
    - The 10x FOV is approximately (but not exactly) centered in the 3x3 tile.
    - BF images at both magnifications are nearly uniform (contrast <9%) and lack
      structural features for reliable template matching.
    - DAPI nuclear staining produces a sparse, high-contrast point pattern that is
      structurally recognizable across magnifications after 2x upscaling.
    - We upscale the Live DAPI by 2x (same camera, 10x→20x = 2:1 pixel ratio),
      then use normalized cross-correlation to find the precise crop position.
    - The identified crop is applied to all 5 Hyb4 channels (DAPI, Ch2, Ch3, Ch1, BF).

Inputs:
    - Live DAPI (C00): uint16, 12-bit ADC, shape (2032, 2432)
    - Hyb4 ICC_Processed: uint16, shape (1,1,5,H,W), Z-axis = 5 channels

Outputs:
    - crop_coords.json: {y0, x0, crop_h, crop_w, match_score}
    - hyb4_crop_<channel>.npy: cropped uint16 arrays for each channel
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from aicsimageio import AICSImage
from loguru import logger
from skimage.feature import match_template
from skimage.transform import resize


class FOVMapper:
    """Finds and extracts the 10x Live FOV from the 20x Hyb4 tiled image."""

    def __init__(self, config_path: str, project_root: str = "."):
        self.project_root = Path(project_root)
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.results_dir = self.project_root / self.cfg["output"]["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"FOVMapper initialized | project_root={self.project_root}")
        logger.info(f"Results dir: {self.results_dir}")

    # ------------------------------------------------------------------
    # Step 1: Load Live DAPI
    # ------------------------------------------------------------------
    def load_live_dapi(self) -> np.ndarray:
        """
        Load the 10x Live DAPI image (uint16, 12-bit ADC).

        Scientific note: DAPI stains DNA in nuclei, producing a sparse,
        high-contrast point pattern. This is structurally recognizable
        after 2x upscaling to match 20x Hyb4 resolution — making it
        far more suitable for template matching than the nearly uniform BF channel.
        """
        path = self.project_root / self.cfg["input"]["live_dapi_path"]
        logger.info(f"[M1 Step 1] Loading Live DAPI: {path.name}")

        img = AICSImage(str(path))
        arr = img.get_image_data("TCZYX")[0, 0, 0]  # → (H, W)

        logger.info(f"  Live DAPI shape: {arr.shape}, dtype: {arr.dtype}")
        logger.info(f"  Live DAPI range: min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
        nonzero_frac = float(np.count_nonzero(arr)) / arr.size
        logger.info(f"  Non-zero pixels: {nonzero_frac*100:.1f}% (sparse DAPI signal expected)")

        if arr.max() > 4095:
            logger.warning(f"  Live DAPI max ({arr.max()}) exceeds 12-bit range (4095) — "
                           f"verify bit depth setting in config.")
        return arr

    # ------------------------------------------------------------------
    # Step 2: Upscale Live DAPI to 20x resolution (template)
    # ------------------------------------------------------------------
    def upscale_to_template(self, live_dapi: np.ndarray) -> np.ndarray:
        """
        Resize Live DAPI by pixel_ratio (2.0) to match 20x pixel size.

        Scientific note: Both 10x and 20x use the same camera sensor.
        The pixel size ratio is exactly 2:1 (magnification ratio).
        Upscaling brings nuclear features to the correct spatial scale
        for matching against Hyb4 DAPI — nuclei will appear the same
        size in both images after upscaling.

        Normalization: clip at 99.9th percentile to preserve nuclear
        intensity peaks (not flattened by saturation artifacts).
        """
        ratio = self.cfg["processing"]["pixel_ratio"]
        bit_depth = self.cfg["processing"]["live_bit_depth"]
        clip_pct = self.cfg["processing"]["clip_percentile"]

        target_h = int(round(live_dapi.shape[0] * ratio))
        target_w = int(round(live_dapi.shape[1] * ratio))

        logger.info(f"[M1 Step 2] Upscaling Live DAPI: {live_dapi.shape} → ({target_h}, {target_w})")

        # Clip only extreme outliers — preserve nuclear signal peaks
        clip_val = np.percentile(live_dapi, clip_pct)
        live_clipped = np.clip(live_dapi, 0, clip_val)
        logger.info(f"  Clip at {clip_pct}th percentile = {clip_val:.0f} "
                    f"(12-bit max = {2**bit_depth - 1})")

        # Normalize to [0, 1]
        live_norm = live_clipped.astype(np.float32) / (2**bit_depth - 1)

        # Resize with anti-aliasing (bilinear interpolation)
        template = resize(live_norm, (target_h, target_w),
                          order=1, anti_aliasing=True, preserve_range=True)

        logger.info(f"  Template shape: {template.shape}, range: [{template.min():.3f}, {template.max():.3f}]")
        return template.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 3: Load Hyb4 BF from ICC_Processed
    # ------------------------------------------------------------------
    def load_hyb4_icc(self) -> np.ndarray:
        """
        Load all 5 channels from the Hyb4 ICC_Processed TIF.

        IMPORTANT: The ICC_Processed file has shape (T=1, C=1, Z=5, H, W).
        The Z-axis encodes 5 channels post-MIP (FIJI export quirk), NOT z-planes.
        Channel order: Z0=DAPI, Z1=Ch2/AF590, Z2=Ch3/AF488, Z3=Ch1/AF647, Z4=BF
        """
        path = self.project_root / self.cfg["input"]["hyb4_icc_path"]
        logger.info(f"[M1 Step 3] Loading Hyb4 ICC_Processed: {path.name}")
        logger.info(f"  NOTE: Z-axis = 5 channels (FIJI export format), not z-planes")

        t_start = time.time()
        img = AICSImage(str(path))
        arr = img.get_image_data("TCZYX")[0, 0]  # → (Z=5, H, W)
        elapsed = time.time() - t_start

        logger.info(f"  Loaded in {elapsed:.1f}s | shape: {arr.shape}, dtype: {arr.dtype}")
        logger.info(f"  Full image size: {arr.shape[1]} × {arr.shape[2]} px (H × W)")

        # Log per-channel stats for sanity check
        ch_names = self.cfg["processing"]["hyb4_channels"]
        for ch_name, z_idx in ch_names.items():
            ch = arr[z_idx]
            logger.info(f"  Z{z_idx} ({ch_name}): min={ch.min()}, max={ch.max()}, mean={ch.mean():.0f}")

        return arr  # uint16, shape (5, H, W)

    # ------------------------------------------------------------------
    # Step 4: Template matching to find crop position
    # ------------------------------------------------------------------
    def find_crop_position(self, template: np.ndarray, hyb4_arr: np.ndarray) -> dict:
        """
        Find the precise (y0, x0) top-left corner of the 10x FOV in the Hyb4 image.

        Uses normalized cross-correlation (match_template) on the DAPI channel.
        Expected crop size = template size (Live DAPI upscaled to 20x resolution).
        The search space is (H_hyb4 - crop_h) × (W_hyb4 - crop_w).

        Scientific note: DAPI nuclear patterns form a sparse point lattice that
        is geometrically preserved across magnifications (after 2x upscaling).
        Normalized cross-correlation finds the position where these nuclear
        patterns overlap best, giving sub-pixel-accurate crop coordinates.
        """
        dapi_z_idx = self.cfg["processing"]["hyb4_channels"]["DAPI"]
        clip_pct = self.cfg["processing"]["clip_percentile"]
        bit_depth = self.cfg["processing"]["hyb4_bit_depth"]

        hyb4_dapi_raw = hyb4_arr[dapi_z_idx].astype(np.float32)

        # Normalize Hyb4 DAPI to [0, 1] — same percentile clipping as Live DAPI
        clip_val = np.percentile(hyb4_dapi_raw, clip_pct)
        hyb4_bf = np.clip(hyb4_dapi_raw, 0, clip_val) / (2**bit_depth - 1)
        logger.info(f"  Using Hyb4 DAPI (Z={dapi_z_idx}) for template matching")

        crop_h, crop_w = template.shape
        img_h, img_w = hyb4_bf.shape
        search_h = img_h - crop_h
        search_w = img_w - crop_w

        logger.info(f"[M1 Step 4] Template matching")
        logger.info(f"  Template size: {crop_h} × {crop_w} px")
        logger.info(f"  Hyb4 BF size: {img_h} × {img_w} px")
        logger.info(f"  Search space: {search_h} × {search_w} px")
        expected_y0 = search_h // 2
        expected_x0 = search_w // 2
        logger.info(f"  Expected center crop position: y0={expected_y0}, x0={expected_x0}")
        logger.info(f"  Running match_template (this may take 30–90s for large images)...")

        t_start = time.time()
        corr_map = match_template(hyb4_bf, template, pad_input=False)
        elapsed = time.time() - t_start

        logger.info(f"  match_template completed in {elapsed:.1f}s")
        logger.info(f"  Correlation map shape: {corr_map.shape}")
        logger.info(f"  Correlation range: min={corr_map.min():.3f}, max={corr_map.max():.3f}")

        # Find best match
        ij = np.unravel_index(corr_map.argmax(), corr_map.shape)
        y0, x0 = int(ij[0]), int(ij[1])
        match_score = float(corr_map[y0, x0])

        offset_y = y0 - expected_y0
        offset_x = x0 - expected_x0

        logger.info(f"  Best match: y0={y0}, x0={x0} | score={match_score:.4f}")
        logger.info(f"  Offset from center: dy={offset_y:+d}, dx={offset_x:+d} px")

        # Sanity checks
        if match_score < 0.5:
            logger.warning(f"  LOW MATCH SCORE ({match_score:.3f} < 0.5) — "
                           f"possible debris artifact or data mismatch. Verify result visually.")
        else:
            logger.info(f"  Match quality: GOOD (score={match_score:.3f})")

        if abs(offset_y) > 300 or abs(offset_x) > 300:
            logger.warning(f"  LARGE OFFSET from center (dy={offset_y}, dx={offset_x}) — "
                           f"expected <300 px. Verify imaging setup.")
        else:
            logger.info(f"  Offset within expected range (<300 px) — OK")

        return {
            "y0": y0,
            "x0": x0,
            "crop_h": crop_h,
            "crop_w": crop_w,
            "match_score": match_score,
            "offset_from_center_dy": offset_y,
            "offset_from_center_dx": offset_x,
        }

    # ------------------------------------------------------------------
    # Step 5: Extract and save all cropped channels
    # ------------------------------------------------------------------
    def extract_and_save_channels(self, hyb4_arr: np.ndarray, crop_info: dict) -> dict:
        """
        Apply the identified crop to all 5 Hyb4 channels and save as .npy.

        Returns dict of {channel_name: cropped_array}.
        """
        y0 = crop_info["y0"]
        x0 = crop_info["x0"]
        crop_h = crop_info["crop_h"]
        crop_w = crop_info["crop_w"]
        ch_names = self.cfg["processing"]["hyb4_channels"]

        logger.info(f"[M1 Step 5] Extracting all 5 channels at crop [{y0}:{y0+crop_h}, {x0}:{x0+crop_w}]")

        cropped_channels = {}
        for ch_name, z_idx in ch_names.items():
            ch_crop = hyb4_arr[z_idx, y0:y0+crop_h, x0:x0+crop_w]
            cropped_channels[ch_name] = ch_crop

            logger.info(f"  {ch_name} (Z{z_idx}): shape={ch_crop.shape}, "
                        f"min={ch_crop.min()}, max={ch_crop.max()}, mean={ch_crop.mean():.0f}")

            if self.cfg["output"]["save_cropped_channels"]:
                prefix = self.cfg["output"]["cropped_channel_prefix"]
                out_path = self.results_dir / f"{prefix}{ch_name}.npy"
                np.save(str(out_path), ch_crop)
                logger.info(f"  Saved → {out_path.name}")

        return cropped_channels

    # ------------------------------------------------------------------
    # Step 6: Overlay visualization (QC)
    # ------------------------------------------------------------------
    def visualize_result(self, template: np.ndarray, hyb4_arr: np.ndarray,
                         crop_info: dict):
        """
        Generate a 3-panel QC figure:
          Left   — Live DAPI upscaled to 20x resolution
          Center — Hyb4 DAPI crop (same spatial region)
          Right  — Magenta/Green overlay (Live=magenta, Hyb4=green)
                   Good alignment → nuclei appear white/yellow
                   Misalignment  → nuclei appear red or green separately

        Images are downsampled 4x for display (large arrays → manageable figure).

        Scientific note: The overlay uses a two-colour scheme standard in
        microscopy QC (e.g., ImageJ "Merge Channels"). If nuclei appear
        white/grey in the overlay, the spatial registration is correct.
        If they appear separately coloured, there is a systematic offset
        that should be investigated before proceeding to Module 2.
        """
        logger.info("[M1 Step 6] Generating overlay visualization")

        dapi_z_idx = self.cfg["processing"]["hyb4_channels"]["DAPI"]
        bit_depth_hyb4 = self.cfg["processing"]["hyb4_bit_depth"]
        clip_pct = self.cfg["processing"]["clip_percentile"]

        y0, x0 = crop_info["y0"], crop_info["x0"]
        crop_h, crop_w = crop_info["crop_h"], crop_info["crop_w"]

        # --- Prepare Live DAPI (template already normalized [0,1]) ---
        live_norm = template  # (crop_h, crop_w), float32 [0,1]

        # --- Prepare Hyb4 DAPI crop, normalized to [0,1] ---
        hyb4_dapi_raw = hyb4_arr[dapi_z_idx, y0:y0+crop_h, x0:x0+crop_w].astype(np.float32)
        clip_val = np.percentile(hyb4_dapi_raw, clip_pct)
        hyb4_norm = np.clip(hyb4_dapi_raw, 0, clip_val) / (2**bit_depth_hyb4 - 1)

        # --- Downsample 4x for display ---
        ds = 4
        live_ds = live_norm[::ds, ::ds]
        hyb4_ds = hyb4_norm[::ds, ::ds]

        # Contrast stretch each independently for better visibility
        def stretch(arr):
            lo, hi = np.percentile(arr, 1), np.percentile(arr, 99.5)
            return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

        live_disp = stretch(live_ds)
        hyb4_disp = stretch(hyb4_ds)

        # --- Build RGB overlay: Live=magenta (R+B), Hyb4=green (G) ---
        h, w = live_disp.shape
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        overlay[:, :, 0] = live_disp           # R channel = Live
        overlay[:, :, 1] = hyb4_disp           # G channel = Hyb4
        overlay[:, :, 2] = live_disp           # B channel = Live
        # Result: Live-only → magenta, Hyb4-only → green, both → white

        # --- Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"Module 1 QC — FOV Mapping\n"
            f"Crop: y0={y0}, x0={x0} | size={crop_h}×{crop_w}px | "
            f"score={crop_info['match_score']:.3f} | "
            f"offset dy={crop_info['offset_from_center_dy']:+d}, dx={crop_info['offset_from_center_dx']:+d} px",
            fontsize=11
        )

        axes[0].imshow(live_disp, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Live DAPI\n(10x upscaled ×2)", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(hyb4_disp, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Hyb4 DAPI crop\n(20x, ICC_Processed)", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay\nMagenta=Live | Green=Hyb4 | White=aligned", fontsize=10)
        axes[2].axis("off")

        plt.tight_layout()
        out_path = self.results_dir / "module1_fov_overlay_QC.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Overlay saved → {out_path.name}")

    # ------------------------------------------------------------------
    # Step 7: Save crop coordinates
    # ------------------------------------------------------------------
    def save_crop_coords(self, crop_info: dict):
        """Save crop coordinates and match metadata to JSON."""
        fname = self.cfg["output"]["crop_coords_filename"]
        out_path = self.results_dir / fname
        with open(out_path, "w") as f:
            json.dump(crop_info, f, indent=2)
        logger.info(f"[M1 Step 7] Crop coords saved → {out_path}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        Run the full FOV mapping pipeline.

        Returns:
            crop_info (dict): y0, x0, crop_h, crop_w, match_score, offsets
        """
        logger.info("=" * 60)
        logger.info("Module 1 — FOV Mapping: START")
        logger.info("=" * 60)
        t_total = time.time()

        live_dapi = self.load_live_dapi()
        template = self.upscale_to_template(live_dapi)
        hyb4_arr = self.load_hyb4_icc()
        crop_info = self.find_crop_position(template, hyb4_arr)
        self.extract_and_save_channels(hyb4_arr, crop_info)
        self.visualize_result(template, hyb4_arr, crop_info)
        self.save_crop_coords(crop_info)

        elapsed = time.time() - t_total
        logger.info("=" * 60)
        logger.info(f"Module 1 — FOV Mapping: COMPLETE in {elapsed:.1f}s")
        logger.info(f"  Crop position : y0={crop_info['y0']}, x0={crop_info['x0']}")
        logger.info(f"  Crop size     : {crop_info['crop_h']} × {crop_info['crop_w']} px")
        logger.info(f"  Match score   : {crop_info['match_score']:.4f}")
        logger.info("=" * 60)

        return crop_info
