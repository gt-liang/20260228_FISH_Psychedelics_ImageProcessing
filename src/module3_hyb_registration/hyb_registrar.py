"""
Module 3 — Hyb-to-Hyb Brightfield Registration
================================================
Register Hyb2 and Hyb3 BF channels to the Hyb4 BF crop (reference frame),
using phase cross-correlation.

Scientific Logic:
    - After Module 1, Hyb4 channels are cropped to the Live FOV extent.
      Hyb2 and Hyb3 were imaged in the same well, but buffer exchanges
      between rounds cause small stage repositioning shifts (typically < 50 px).
    - DAPI cannot be used for Hyb2/Hyb3: DNase treatment degrades DNA,
      so DAPI signal ≈ 0. Brightfield (BF) is unaffected and shows
      consistent cell body morphology across all rounds.
    - Each Hyb is registered independently to Hyb4 (direct, no chaining)
      to avoid error accumulation.
    - The resulting (dy, dx) per round is used in Module 5 to sample FISH
      spot intensities in a common coordinate system (Hyb4 space).

Inputs:
    - Hyb4 BF crop : python_results/module1/hyb4_crop_BF.npy (M1 output)
    - crop_coords  : python_results/module1/crop_coords.json  (M1 crop origin)
    - Hyb2 ICC_Processed TIF : full 20x tiled image
    - Hyb3 ICC_Processed TIF : full 20x tiled image

Outputs (per round):
    - registration_hybN_to_hyb4.json : {dy, dx, shift_magnitude_px, pearson_r}
    - module3_registration_QC_hybN.png : 4-panel QC overlay
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


class HybRegistrar:
    """Registers Hyb2 and Hyb3 BF to Hyb4 BF crop via phase cross-correlation."""

    def __init__(self, config_path: str, project_root: str = "."):
        self.project_root = Path(project_root)
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.results_dir = self.project_root / self.cfg["output"]["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load M1 crop coordinates (shared reference frame for all rounds)
        coords_path = self.project_root / self.cfg["input"]["crop_coords_path"]
        with open(coords_path) as f:
            self.crop_coords = json.load(f)

        logger.info(f"HybRegistrar initialized | project_root={self.project_root}")
        logger.info(f"Results dir: {self.results_dir}")
        logger.info(f"M1 crop origin: y0={self.crop_coords['y0']}, x0={self.crop_coords['x0']}, "
                    f"h={self.crop_coords['crop_h']}, w={self.crop_coords['crop_w']}")

    # ------------------------------------------------------------------
    # Step 1: Load Hyb4 BF crop (Module 1 output — reference)
    # ------------------------------------------------------------------
    def load_hyb4_bf_crop(self) -> np.ndarray:
        """
        Load the Hyb4 BF crop saved by Module 1 as the registration reference.

        Scientific note: Hyb4 is our reference frame because:
          1. It has usable DAPI for nuclear segmentation (Module 4).
          2. Module 2 already aligned Live to Hyb4.
          3. All downstream analysis (spot calling, decoding) is in Hyb4 space.
        """
        path = self.project_root / self.cfg["input"]["hyb4_bf_crop_path"]
        bit_depth = self.cfg["processing"]["hyb_bit_depth"]
        clip_pct = self.cfg["processing"]["clip_percentile"]

        logger.info(f"[M3] Loading Hyb4 BF crop (reference): {path.name}")
        arr = np.load(str(path))

        logger.info(f"  Shape: {arr.shape}, dtype: {arr.dtype}, "
                    f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")

        clip_val = np.percentile(arr, clip_pct)
        arr_norm = np.clip(arr, 0, clip_val).astype(np.float32) / (2**bit_depth - 1)

        logger.info(f"  Normalized range: [{arr_norm.min():.3f}, {arr_norm.max():.3f}]")
        return arr_norm

    # ------------------------------------------------------------------
    # Step 2: Load, extract BF, and crop for a given Hyb round
    # ------------------------------------------------------------------
    def load_and_crop_bf(self, icc_path: Path, label: str) -> np.ndarray:
        """
        Load an ICC_Processed TIF, extract BF channel (Z=4), and crop to
        the same spatial region as the Hyb4 crop (using M1 crop_coords).

        Scientific note: ICC_Processed files use a FIJI export format where
        the Z-axis encodes 5 channels (not focal planes):
            Z0=DAPI, Z1=AF590, Z2=AF488, Z3=AF647, Z4=BF
        We access: img.get_image_data("TCZYX")[0, 0, bf_idx, :, :]
        """
        bf_idx = self.cfg["processing"]["bf_channel_idx"]
        bit_depth = self.cfg["processing"]["hyb_bit_depth"]
        clip_pct = self.cfg["processing"]["clip_percentile"]

        y0 = self.crop_coords["y0"]
        x0 = self.crop_coords["x0"]
        crop_h = self.crop_coords["crop_h"]
        crop_w = self.crop_coords["crop_w"]

        logger.info(f"[M3] Loading {label} BF: {icc_path.name}")
        arr_full = AICSImage(str(icc_path)).get_image_data("TCZYX")[0, 0, bf_idx]

        logger.info(f"  Full shape: {arr_full.shape}, dtype: {arr_full.dtype}, "
                    f"min={arr_full.min()}, max={arr_full.max()}, mean={arr_full.mean():.1f}")

        # Apply same crop as M1 to bring into Hyb4 spatial frame
        arr_crop = arr_full[y0:y0 + crop_h, x0:x0 + crop_w]
        logger.info(f"  Cropped shape: {arr_crop.shape} "
                    f"(y0={y0}, x0={x0}, h={crop_h}, w={crop_w})")

        clip_val = np.percentile(arr_crop, clip_pct)
        arr_norm = np.clip(arr_crop, 0, clip_val).astype(np.float32) / (2**bit_depth - 1)

        logger.info(f"  Normalized range: [{arr_norm.min():.3f}, {arr_norm.max():.3f}]")
        return arr_norm

    # ------------------------------------------------------------------
    # Step 3: Phase correlation registration
    # ------------------------------------------------------------------
    def register(self, reference: np.ndarray, moving: np.ndarray, label: str) -> dict:
        """
        Compute translation (dy, dx) from a moving Hyb BF image to the Hyb4 BF reference.

        Scientific note: normalization=None is used (not "phase") because BF images,
        while not as sparse as DAPI, still benefit from amplitude-weighted correlation
        for numerical stability. Quality is assessed via Pearson r after applying
        the estimated shift.

        Result interpretation:
            shift (dy, dx) = how much to shift `moving` to align with `reference`
            In Module 5, FISH spot coordinates from HybN will be offset by (dy, dx)
            to map them into Hyb4 space for per-nucleus intensity sampling.
        """
        upsample = self.cfg["processing"]["upsample_factor"]
        max_shift = self.cfg["processing"]["max_shift_px"]

        logger.info(f"[M3] Phase correlation: {label} BF → Hyb4 BF")
        logger.info(f"  Reference: Hyb4 BF {reference.shape}")
        logger.info(f"  Moving   : {label} BF {moving.shape}")

        if reference.shape != moving.shape:
            logger.warning(f"  SIZE MISMATCH: reference {reference.shape} vs moving {moving.shape} — "
                           f"cropping to minimum overlap")
            h = min(reference.shape[0], moving.shape[0])
            w = min(reference.shape[1], moving.shape[1])
            reference = reference[:h, :w]
            moving = moving[:h, :w]
            logger.info(f"  Cropped to: {reference.shape}")

        logger.info(f"  upsample_factor={upsample} → precision = {1/upsample:.2f} px")

        t_start = time.time()
        shift_vec, error, phasediff = phase_cross_correlation(
            reference, moving,
            upsample_factor=upsample,
            normalization=None,
        )
        elapsed = time.time() - t_start

        dy, dx = float(shift_vec[0]), float(shift_vec[1])
        magnitude = float(np.sqrt(dy**2 + dx**2))

        logger.info(f"  phase_cross_correlation completed in {elapsed:.2f}s")
        logger.info(f"  Shift: dy={dy:+.2f} px, dx={dx:+.2f} px")
        logger.info(f"  Magnitude: {magnitude:.2f} px")

        # Independent quality metric: Pearson r after applying shift
        moving_shifted_q = ndimage_shift(moving, shift=(dy, dx), order=1, mode="constant", cval=0)
        pearson_r = float(np.corrcoef(moving_shifted_q.ravel(), reference.ravel())[0, 1])
        logger.info(f"  Pearson r (after shift, higher=better): {pearson_r:.4f}")

        # Quality guards
        if pearson_r < 0.1:
            logger.warning(f"  LOW CORRELATION (r={pearson_r:.4f} < 0.1) — "
                           f"registration may be unreliable. Check BF signal quality.")
        elif pearson_r < 0.3:
            logger.warning(f"  MODERATE CORRELATION (r={pearson_r:.4f}) — "
                           f"worth verifying QC overlay.")
        else:
            logger.info(f"  Registration quality: GOOD (Pearson r={pearson_r:.4f} > 0.3)")

        if abs(dy) > max_shift or abs(dx) > max_shift:
            logger.warning(f"  LARGE SHIFT detected (dy={dy:+.1f}, dx={dx:+.1f}) > {max_shift} px — "
                           f"verify that the {label} ICC image is from the same FOV as Hyb4.")
        else:
            logger.info(f"  Shift within expected range (< {max_shift} px) — OK")

        return {
            "label": label,
            "dy": dy,
            "dx": dx,
            "shift_magnitude_px": magnitude,
            "pearson_r": pearson_r,
            "upsample_factor": upsample,
        }

    # ------------------------------------------------------------------
    # Step 4: QC visualization
    # ------------------------------------------------------------------
    def visualize_result(self, reference: np.ndarray, moving: np.ndarray,
                         reg: dict):
        """
        Generate a 4-panel QC figure for one Hyb round:
          Panel 1 — Reference (Hyb4 BF crop)
          Panel 2 — Moving (HybN BF crop, before registration)
          Panel 3 — Overlay BEFORE registration (shows residual offset)
          Panel 4 — Overlay AFTER applying shift (should be white/grey cells)
        """
        label = reg["label"]
        dy, dx = reg["dy"], reg["dx"]
        ds = 4  # downsample factor for display

        logger.info(f"[M3] Generating QC visualization for {label}")

        moving_shifted = ndimage_shift(moving, shift=(dy, dx), order=1, mode="constant", cval=0)

        pad_y = int(abs(dy)) + 5
        pad_x = int(abs(dx)) + 5
        h, w = reference.shape
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

        ref_ds = stretch(reference[y1:y2:ds, x1:x2:ds])
        mov_ds = stretch(moving[y1:y2:ds, x1:x2:ds])
        mov_shift_ds = stretch(moving_shifted[y1:y2:ds, x1:x2:ds])

        overlay_before = make_overlay(mov_ds, ref_ds)
        overlay_after = make_overlay(mov_shift_ds, ref_ds)

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle(
            f"Module 3 QC — {label} BF → Hyb4 BF Registration\n"
            f"shift: dy={dy:+.2f} px, dx={dx:+.2f} px | "
            f"magnitude={reg['shift_magnitude_px']:.2f} px | "
            f"Pearson r={reg['pearson_r']:.4f} (higher=better)",
            fontsize=11
        )

        axes[0].imshow(ref_ds, cmap="gray")
        axes[0].set_title("Hyb4 BF crop\n(reference)", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(mov_ds, cmap="gray")
        axes[1].set_title(f"{label} BF crop\n(before registration)", fontsize=9)
        axes[1].axis("off")

        axes[2].imshow(overlay_before)
        axes[2].set_title("Overlay BEFORE\nMagenta=Moving | Green=Ref", fontsize=9)
        axes[2].axis("off")

        axes[3].imshow(overlay_after)
        axes[3].set_title("Overlay AFTER registration\n(should be white/grey)", fontsize=9)
        axes[3].axis("off")

        plt.tight_layout()
        out_path = self.results_dir / f"module3_registration_QC_{label.lower()}.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  QC image saved → {out_path.name}")

    # ------------------------------------------------------------------
    # Step 5: Save registration result
    # ------------------------------------------------------------------
    def save_result(self, reg: dict, filename: str):
        """Save shift vector and quality metrics to JSON."""
        out_path = self.results_dir / filename
        with open(out_path, "w") as f:
            json.dump(reg, f, indent=2)
        logger.info(f"[M3] Registration result saved → {out_path.name}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        Register Hyb2 and Hyb3 BF to Hyb4 BF.

        Returns:
            results (dict): {"hyb2": reg_dict, "hyb3": reg_dict}
        """
        logger.info("=" * 60)
        logger.info("Module 3 — Hyb-to-Hyb BF Registration: START")
        logger.info("=" * 60)
        t_total = time.time()

        hyb4_bf = self.load_hyb4_bf_crop()

        results = {}
        for label, icc_key, out_key in [
            ("Hyb3", "hyb3_icc_path", "reg_hyb3_filename"),
            ("Hyb2", "hyb2_icc_path", "reg_hyb2_filename"),
        ]:
            logger.info("-" * 40)
            icc_path = self.project_root / self.cfg["input"][icc_key]
            moving_bf = self.load_and_crop_bf(icc_path, label)
            reg = self.register(hyb4_bf, moving_bf, label)
            self.visualize_result(hyb4_bf, moving_bf, reg)
            self.save_result(reg, self.cfg["output"][out_key])
            results[label.lower()] = reg

        elapsed = time.time() - t_total
        logger.info("=" * 60)
        logger.info(f"Module 3 — Hyb-to-Hyb BF Registration: COMPLETE in {elapsed:.1f}s")
        for label, reg in results.items():
            logger.info(f"  {label.upper()} → Hyb4 : dy={reg['dy']:+.2f} px, "
                        f"dx={reg['dx']:+.2f} px | Pearson r={reg['pearson_r']:.4f}")
        logger.info("=" * 60)

        return results
