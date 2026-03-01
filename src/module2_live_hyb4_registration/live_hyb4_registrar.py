"""
Module 2 — Live → Hyb4 Global Registration
===========================================
Find the precise translation (dy, dx) between 10x Live DAPI and 20x Hyb4 DAPI crop
using phase correlation.

Scientific Logic:
    - After Module 1, the Hyb4 DAPI crop spatially covers the same physical area
      as the 10x Live FOV (at 20x resolution). However, there is still a residual
      sub-pixel-to-~tens-of-pixel translation between the two images due to:
        * Slight mis-centering not captured by template matching
        * Stage repositioning drift between Live and Hyb4 imaging sessions
        * Any mechanical shift during sample mounting/remounting
    - Phase correlation finds this global translation in the frequency domain.
      It is robust to: intensity differences (Live 12-bit vs Hyb4 16-bit),
      nuclear morphology changes (Live blurry vs Fixed sharp), and sparse debris.
    - No rotation or scaling is expected (same camera, same well, no remounting).
    - The output shift (dy, dx) is used in Module 6 to map Live nucleus IDs
      into the Hyb4 coordinate space for barcode assignment.

Inputs:
    - Live DAPI (C00): uint16, 12-bit ADC, shape (2032, 2432) → upscaled to (4064, 4864)
    - Hyb4 DAPI crop: uint16, shape (4064, 4864) — output of Module 1

Outputs:
    - registration_live_hyb4.json: {dy, dx, shift_magnitude_px, pearson_r}
    - module2_registration_QC.png: before/after overlay visualization
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from aicsimageio import AICSImage
from loguru import logger
from scipy.ndimage import shift as ndimage_shift
from skimage.registration import phase_cross_correlation
from skimage.transform import resize


class LiveHyb4Registrar:
    """Registers 10x Live DAPI to 20x Hyb4 DAPI crop via phase correlation."""

    def __init__(self, config_path: str, project_root: str = "."):
        self.project_root = Path(project_root)
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.results_dir = self.project_root / self.cfg["output"]["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LiveHyb4Registrar initialized | project_root={self.project_root}")
        logger.info(f"Results dir: {self.results_dir}")

    # ------------------------------------------------------------------
    # Step 1: Load and upscale Live DAPI
    # ------------------------------------------------------------------
    def load_live_dapi_upscaled(self) -> np.ndarray:
        """
        Load Live DAPI (12-bit uint16) and upscale 2x to match 20x pixel size.

        Scientific note: The same camera is used for 10x and 20x imaging.
        Upscaling by 2x brings the Live image to the same pixel scale as
        the Hyb4 crop — a prerequisite for pixel-accurate phase correlation.
        """
        path = self.project_root / self.cfg["input"]["live_dapi_path"]
        ratio = self.cfg["processing"]["pixel_ratio"]
        bit_depth = self.cfg["processing"]["live_bit_depth"]
        clip_pct = self.cfg["processing"]["clip_percentile"]

        logger.info(f"[M2 Step 1] Loading Live DAPI: {path.name}")
        arr = AICSImage(str(path)).get_image_data("TCZYX")[0, 0, 0]

        logger.info(f"  Shape: {arr.shape}, dtype: {arr.dtype}, "
                    f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")

        # Normalize to [0,1] with percentile clipping
        clip_val = np.percentile(arr, clip_pct)
        arr_norm = np.clip(arr, 0, clip_val).astype(np.float32) / (2**bit_depth - 1)

        # Upscale 2x
        target_h = int(round(arr.shape[0] * ratio))
        target_w = int(round(arr.shape[1] * ratio))
        arr_up = resize(arr_norm, (target_h, target_w),
                        order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)

        logger.info(f"  Upscaled: {arr.shape} → {arr_up.shape} | "
                    f"range: [{arr_up.min():.3f}, {arr_up.max():.3f}]")
        return arr_up

    # ------------------------------------------------------------------
    # Step 2: Load Hyb4 DAPI crop (Module 1 output)
    # ------------------------------------------------------------------
    def load_hyb4_dapi_crop(self) -> np.ndarray:
        """
        Load the Hyb4 DAPI crop saved by Module 1.

        This is the 20x Hyb4 DAPI channel, already cropped to the
        spatial extent of the 10x Live FOV.
        """
        path = self.project_root / self.cfg["input"]["hyb4_dapi_crop_path"]
        bit_depth = self.cfg["processing"]["hyb4_bit_depth"]
        clip_pct = self.cfg["processing"]["clip_percentile"]

        logger.info(f"[M2 Step 2] Loading Hyb4 DAPI crop: {path.name}")
        arr = np.load(str(path))

        logger.info(f"  Shape: {arr.shape}, dtype: {arr.dtype}, "
                    f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")

        # Normalize to [0,1]
        clip_val = np.percentile(arr, clip_pct)
        arr_norm = np.clip(arr, 0, clip_val).astype(np.float32) / (2**bit_depth - 1)

        logger.info(f"  Normalized range: [{arr_norm.min():.3f}, {arr_norm.max():.3f}]")
        return arr_norm

    # ------------------------------------------------------------------
    # Step 3: Phase correlation registration
    # ------------------------------------------------------------------
    def register(self, live_up: np.ndarray, hyb4_dapi: np.ndarray) -> dict:
        """
        Compute the translation (dy, dx) from Live DAPI to Hyb4 DAPI
        using phase cross-correlation.

        Scientific note: Phase correlation computes the normalized cross-power
        spectrum in Fourier space, then finds the dominant frequency shift.
        Result interpretation:
            shift = (dy, dx) means the Live image must be shifted by (dy, dx)
            to align with Hyb4. Equivalently, Hyb4 features appear at
            (y_live + dy, x_live + dx) in the Live coordinate system.

        upsample_factor=10 gives 0.1 px sub-pixel precision at modest cost.
        """
        upsample = self.cfg["processing"]["upsample_factor"]
        max_shift = self.cfg["processing"]["max_shift_px"]
        min_peak = self.cfg["processing"]["min_phase_peak"]

        logger.info(f"[M2 Step 3] Phase correlation registration")
        logger.info(f"  Reference: Hyb4 DAPI crop {hyb4_dapi.shape}")
        logger.info(f"  Moving   : Live DAPI upscaled {live_up.shape}")

        if live_up.shape != hyb4_dapi.shape:
            logger.warning(f"  SIZE MISMATCH: Live {live_up.shape} vs Hyb4 {hyb4_dapi.shape} — "
                           f"cropping to minimum overlap for phase correlation")
            h = min(live_up.shape[0], hyb4_dapi.shape[0])
            w = min(live_up.shape[1], hyb4_dapi.shape[1])
            live_up = live_up[:h, :w]
            hyb4_dapi = hyb4_dapi[:h, :w]
            logger.info(f"  Cropped to: {live_up.shape}")

        logger.info(f"  upsample_factor={upsample} → precision = {1/upsample:.2f} px")

        t_start = time.time()
        shift_vec, error, phasediff = phase_cross_correlation(
            hyb4_dapi, live_up,
            upsample_factor=upsample,
            normalization=None,   # standard cross-correlation; avoids numerical instability
        )                         # that normalization="phase" causes on sparse DAPI images
        elapsed = time.time() - t_start

        dy, dx = float(shift_vec[0]), float(shift_vec[1])
        magnitude = float(np.sqrt(dy**2 + dx**2))

        logger.info(f"  phase_cross_correlation completed in {elapsed:.2f}s")
        logger.info(f"  Shift: dy={dy:+.2f} px, dx={dx:+.2f} px")
        logger.info(f"  Magnitude: {magnitude:.2f} px")

        # --- Independent quality metric: Pearson r after applying shift ---
        # Apply the estimated shift to Live and measure spatial co-localization with Hyb4.
        # Pearson r close to 1.0 means nuclei overlap well → registration succeeded.
        live_shifted_q = ndimage_shift(live_up, shift=(dy, dx), order=1, mode="constant", cval=0)
        pearson_r = float(np.corrcoef(live_shifted_q.ravel(), hyb4_dapi.ravel())[0, 1])
        logger.info(f"  Pearson r (after shift, higher=better): {pearson_r:.4f}")

        # --- Quality guards ---
        if pearson_r < 0.1:
            logger.warning(f"  LOW CORRELATION (r={pearson_r:.4f} < 0.1) — "
                           f"registration may be unreliable. Check DAPI signal quality.")
        elif pearson_r < 0.3:
            logger.warning(f"  MODERATE CORRELATION (r={pearson_r:.4f}) — "
                           f"result is usable but worth verifying the QC overlay.")
        else:
            logger.info(f"  Registration quality: GOOD (Pearson r={pearson_r:.4f} > 0.3)")

        if abs(dy) > max_shift or abs(dx) > max_shift:
            logger.warning(f"  LARGE SHIFT detected (dy={dy:+.1f}, dx={dx:+.1f}) > {max_shift} px — "
                           f"verify that M1 crop is correct and Live DAPI is from the same FOV.")
        else:
            logger.info(f"  Shift within expected range (< {max_shift} px) — OK")

        return {
            "dy": dy,
            "dx": dx,
            "shift_magnitude_px": magnitude,
            "pearson_r": pearson_r,
            "upsample_factor": upsample,
        }

    # ------------------------------------------------------------------
    # Step 4: Overlay QC visualization (before / after)
    # ------------------------------------------------------------------
    def visualize_result(self, live_up: np.ndarray, hyb4_dapi: np.ndarray,
                         reg: dict):
        """
        Generate a 4-panel QC figure:
          Panel 1 — Live DAPI upscaled (before registration)
          Panel 2 — Hyb4 DAPI crop (reference)
          Panel 3 — Overlay BEFORE registration (shows residual offset)
          Panel 4 — Overlay AFTER applying shift (should show white nuclei)

        Scientific note: After registration, nuclei from both images should
        co-localize → white/grey in the magenta-green overlay. Remaining
        colour separation indicates non-translational misalignment
        (rotation/scaling) which is not expected in this experiment.
        """
        logger.info("[M2 Step 4] Generating registration QC visualization")

        dy, dx = reg["dy"], reg["dx"]
        ds = 4  # downsample factor for display

        # Apply shift to Live to get registered version
        live_shifted = ndimage_shift(live_up, shift=(dy, dx), order=1, mode="constant", cval=0)

        # Crop to valid overlap region (avoid shifted zeros at edges)
        pad_y = int(abs(dy)) + 5
        pad_x = int(abs(dx)) + 5
        h, w = live_up.shape
        y1, y2 = pad_y, h - pad_y
        x1, x2 = pad_x, w - pad_x

        def stretch(arr):
            lo, hi = np.percentile(arr, 1), np.percentile(arr, 99.5)
            return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

        def make_overlay(ch1, ch2):
            """Magenta/Green overlay: ch1=magenta (R+B), ch2=green (G)."""
            rgb = np.zeros((*ch1.shape, 3), dtype=np.float32)
            rgb[:, :, 0] = ch1
            rgb[:, :, 1] = ch2
            rgb[:, :, 2] = ch1
            return rgb

        # Downsampled patches for display
        live_ds = stretch(live_up[y1:y2:ds, x1:x2:ds])
        hyb4_ds = stretch(hyb4_dapi[y1:y2:ds, x1:x2:ds])
        live_shift_ds = stretch(live_shifted[y1:y2:ds, x1:x2:ds])

        overlay_before = make_overlay(live_ds, hyb4_ds)
        overlay_after = make_overlay(live_shift_ds, hyb4_ds)

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle(
            f"Module 2 QC — Live→Hyb4 Registration\n"
            f"shift: dy={dy:+.2f} px, dx={dx:+.2f} px | "
            f"magnitude={reg['shift_magnitude_px']:.2f} px | "
            f"Pearson r={reg['pearson_r']:.4f} (higher=better)",
            fontsize=11
        )

        axes[0].imshow(live_ds, cmap="gray")
        axes[0].set_title("Live DAPI\n(upscaled ×2, before shift)", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(hyb4_ds, cmap="gray")
        axes[1].set_title("Hyb4 DAPI crop\n(reference)", fontsize=9)
        axes[1].axis("off")

        axes[2].imshow(overlay_before)
        axes[2].set_title("Overlay BEFORE\nMagenta=Live | Green=Hyb4", fontsize=9)
        axes[2].axis("off")

        axes[3].imshow(overlay_after)
        axes[3].set_title("Overlay AFTER registration\n(should be white/grey)", fontsize=9)
        axes[3].axis("off")

        plt.tight_layout()
        out_path = self.results_dir / "module2_registration_QC.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  QC image saved → {out_path.name}")

    # ------------------------------------------------------------------
    # Step 5: Save registration result
    # ------------------------------------------------------------------
    def save_result(self, reg: dict):
        """Save shift vector and quality metrics to JSON."""
        fname = self.cfg["output"]["registration_filename"]
        out_path = self.results_dir / fname
        with open(out_path, "w") as f:
            json.dump(reg, f, indent=2)
        logger.info(f"[M2 Step 5] Registration result saved → {out_path.name}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        Run the full Live→Hyb4 registration pipeline.

        Returns:
            reg (dict): dy, dx, shift_magnitude_px, phase_peak_value
        """
        logger.info("=" * 60)
        logger.info("Module 2 — Live→Hyb4 Registration: START")
        logger.info("=" * 60)
        t_total = time.time()

        live_up = self.load_live_dapi_upscaled()
        hyb4_dapi = self.load_hyb4_dapi_crop()
        reg = self.register(live_up, hyb4_dapi)
        self.visualize_result(live_up, hyb4_dapi, reg)
        self.save_result(reg)

        elapsed = time.time() - t_total
        logger.info("=" * 60)
        logger.info(f"Module 2 — Live→Hyb4 Registration: COMPLETE in {elapsed:.1f}s")
        logger.info(f"  Shift : dy={reg['dy']:+.2f} px, dx={reg['dx']:+.2f} px")
        logger.info(f"  Magnitude : {reg['shift_magnitude_px']:.2f} px")
        logger.info(f"  Pearson r : {reg['pearson_r']:.4f} (higher=better)")
        logger.info("=" * 60)

        return reg
