"""
Module 4 — Nuclear Segmentation (CellposeSAM)
=============================================
Segment individual nuclei in the Hyb4 DAPI crop using CellposeSAM (v4).

Scientific Logic:
    - Input: Hyb4 DAPI crop (20x, 16-bit, output of Module 1) — the only
      round with usable DAPI (Hyb2/Hyb3 are DNase-treated).
    - CellposeSAM uses a SAM (Segment Anything Model) backbone to detect
      cell boundaries from fluorescence images without requiring channel-
      specific training. It auto-detects nuclear diameter (diameter=0).
    - Output: an integer label image where each unique non-zero value
      identifies a single nucleus. Label 0 = background.
    - Post-filtering: nuclei outside [min_area, max_area] are removed
      (too small → debris or partial; too large → clumped nuclei).
    - The label image is the common coordinate reference for Module 5
      (spot calling) and Module 6 (decoding).

Inputs:
    - Hyb4 DAPI crop: python_results/module1/hyb4_crop_DAPI.npy

Outputs:
    - nucleus_labels.npy      : uint32 label image, shape = (H, W)
    - nucleus_properties.csv  : per-nucleus id, centroid_y, centroid_x, area_px
    - module4_segmentation_QC.png : overlay of labels on DAPI image
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from skimage.measure import regionprops_table


class NuclearSegmentor:
    """Segments nuclei from Hyb4 DAPI crop using CellposeSAM."""

    def __init__(self, config_path: str, project_root: str = "."):
        self.project_root = Path(project_root)
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.results_dir = self.project_root / self.cfg["output"]["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"NuclearSegmentor initialized | project_root={self.project_root}")
        logger.info(f"Results dir: {self.results_dir}")

    # ------------------------------------------------------------------
    # Step 1: Load Hyb4 DAPI crop
    # ------------------------------------------------------------------
    def load_dapi(self) -> np.ndarray:
        """
        Load Hyb4 DAPI crop and normalize to [0, 1] float32 for Cellpose.

        Scientific note: Cellpose expects float input in [0, 1] or uint8 [0, 255].
        We normalize with percentile clipping to bring nuclear signal into a
        robust range without losing bright nuclei to saturation.
        """
        path = self.project_root / self.cfg["input"]["hyb4_dapi_crop_path"]
        bit_depth = self.cfg["processing"]["hyb4_bit_depth"]
        clip_pct = self.cfg["processing"]["clip_percentile"]

        logger.info(f"[M4 Step 1] Loading Hyb4 DAPI crop: {path.name}")
        arr = np.load(str(path))

        logger.info(f"  Shape: {arr.shape}, dtype: {arr.dtype}, "
                    f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")

        clip_val = np.percentile(arr, clip_pct)
        arr_norm = np.clip(arr, 0, clip_val).astype(np.float32) / (2**bit_depth - 1)

        logger.info(f"  Normalized range: [{arr_norm.min():.3f}, {arr_norm.max():.3f}]")
        return arr_norm

    # ------------------------------------------------------------------
    # Step 2: Run CellposeSAM
    # ------------------------------------------------------------------
    def segment(self, dapi: np.ndarray) -> np.ndarray:
        """
        Run CellposeSAM on the DAPI image and return a raw label image.

        Scientific note: CellposeSAM (v4) uses a single universal model
        (SAM backbone). The `model_type` argument is deprecated. Key params:
          - diameter=0: auto-detect from image (recommended for unknown samples)
          - flow_threshold: controls how strict the boundary flow must be
            (lower → fewer but cleaner masks; default 0.4 is a good starting point)
          - cellprob_threshold: lower → more cells detected (default 0.0)
        """
        from cellpose import models

        use_gpu = self.cfg["processing"]["use_gpu"]
        diameter_cfg = self.cfg["processing"]["diameter"]
        # Cellpose v4 API: diameter=None triggers auto-detection; 0 is not valid
        diameter = None if diameter_cfg == 0 else diameter_cfg
        flow_threshold = self.cfg["processing"]["flow_threshold"]
        cellprob_threshold = self.cfg["processing"]["cellprob_threshold"]

        logger.info(f"[M4 Step 2] Running CellposeSAM")
        logger.info(f"  diameter={diameter} (None=auto), flow_threshold={flow_threshold}, "
                    f"cellprob_threshold={cellprob_threshold}, use_gpu={use_gpu}")

        # Convert to uint16 for Cellpose (it handles uint16 natively)
        dapi_u16 = (dapi * 65535).astype(np.uint16)

        model = models.CellposeModel(gpu=use_gpu)
        logger.info(f"  Model device: {'MPS/GPU' if use_gpu else 'CPU'}")

        t_start = time.time()
        masks, flows, styles = model.eval(
            dapi_u16,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        elapsed = time.time() - t_start

        n_raw = int(masks.max())
        logger.info(f"  Cellpose completed in {elapsed:.1f}s")
        logger.info(f"  Raw nuclei detected: {n_raw}")

        return masks.astype(np.uint32)

    # ------------------------------------------------------------------
    # Step 3: Post-filter by area
    # ------------------------------------------------------------------
    def filter_labels(self, masks: np.ndarray) -> np.ndarray:
        """
        Remove nuclei outside [min_area, max_area] and re-label sequentially.

        Scientific note:
          - Too small (< min_area): debris, partial nuclei at image edge,
            or Cellpose false positives on bright specks.
          - Too large (> max_area): clumped/touching nuclei not separated,
            or artefacts. In practice, max_area rarely triggers.
          After filtering, labels are re-numbered 1..N for clean indexing.
        """
        min_area = self.cfg["processing"]["min_nucleus_area_px"]
        max_area = self.cfg["processing"]["max_nucleus_area_px"]

        logger.info(f"[M4 Step 3] Filtering nuclei: area in [{min_area}, {max_area}] px²")

        filtered = masks.copy()
        props = regionprops_table(masks, properties=["label", "area"])
        for lbl, area in zip(props["label"], props["area"]):
            if area < min_area or area > max_area:
                filtered[filtered == lbl] = 0

        # Re-number surviving labels sequentially (1..N) WITHOUT merging.
        # Using sk_label(filtered > 0) would merge adjacent regions that share
        # a border after their separating pixel was zeroed out → wrong areas.
        # Instead: compact the existing label set while preserving boundaries.
        unique_labels = np.unique(filtered)
        unique_labels = unique_labels[unique_labels > 0]   # drop background
        filtered_relabeled = np.zeros_like(filtered, dtype=np.uint32)
        for new_id, old_id in enumerate(unique_labels, start=1):
            filtered_relabeled[filtered == old_id] = new_id

        n_before = int(masks.max())
        n_after = int(filtered_relabeled.max())
        logger.info(f"  Nuclei before filter: {n_before}")
        logger.info(f"  Nuclei after  filter: {n_after} "
                    f"(removed {n_before - n_after})")

        return filtered_relabeled

    # ------------------------------------------------------------------
    # Step 4: Extract per-nucleus properties
    # ------------------------------------------------------------------
    def extract_properties(self, labels: np.ndarray) -> pd.DataFrame:
        """
        Compute per-nucleus: id, centroid (y, x), area.

        The centroid (centroid_y, centroid_x) in Hyb4 pixel coordinates
        is used in Module 6 for barcode assignment display and QC.
        """
        logger.info(f"[M4 Step 4] Extracting nucleus properties")

        props = regionprops_table(
            labels,
            properties=["label", "centroid", "area"]
        )
        df = pd.DataFrame({
            "nucleus_id":   props["label"].astype(int),
            "centroid_y":   props["centroid-0"],
            "centroid_x":   props["centroid-1"],
            "area_px":      props["area"].astype(int),
        })

        logger.info(f"  {len(df)} nuclei | area: "
                    f"mean={df['area_px'].mean():.0f} px², "
                    f"min={df['area_px'].min()} px², "
                    f"max={df['area_px'].max()} px²")
        return df

    # ------------------------------------------------------------------
    # Step 5: QC visualization
    # ------------------------------------------------------------------
    def visualize_result(self, dapi: np.ndarray, labels: np.ndarray, df: pd.DataFrame):
        """
        Generate a 2-panel QC figure:
          Panel 1 — DAPI image
          Panel 2 — Segmentation overlay (random colors per nucleus)
                    with nucleus count and area stats
        """
        logger.info(f"[M4 Step 5] Generating segmentation QC visualization")

        ds = 4  # downsample for display

        def stretch(arr):
            lo, hi = np.percentile(arr, 1), np.percentile(arr, 99.5)
            return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

        dapi_ds = stretch(dapi[::ds, ::ds])
        labels_ds = labels[::ds, ::ds]

        # Random-color label image
        rng = np.random.default_rng(42)
        n_labels = int(labels_ds.max())
        color_map = np.zeros((n_labels + 1, 3), dtype=np.float32)
        color_map[1:] = rng.uniform(0.3, 1.0, size=(n_labels, 3))

        label_rgb = color_map[labels_ds]        # (H, W, 3)
        overlay = dapi_ds[:, :, None] * 0.5 + label_rgb * 0.5
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(
            f"Module 4 QC — Nuclear Segmentation (CellposeSAM)\n"
            f"Nuclei: {len(df)} | area mean={df['area_px'].mean():.0f} px² | "
            f"min={df['area_px'].min()} px² | max={df['area_px'].max()} px²",
            fontsize=11
        )

        axes[0].imshow(dapi_ds, cmap="gray")
        axes[0].set_title("Hyb4 DAPI crop\n(input)", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(f"Segmentation overlay\n{len(df)} nuclei (random colors)", fontsize=9)
        axes[1].axis("off")

        plt.tight_layout()
        out_path = self.results_dir / "module4_segmentation_QC.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  QC image saved → {out_path.name}")

    # ------------------------------------------------------------------
    # Step 6: Save outputs
    # ------------------------------------------------------------------
    def save_outputs(self, labels: np.ndarray, df: pd.DataFrame):
        """Save label image (.npy) and nucleus properties (.csv)."""
        label_path = self.results_dir / self.cfg["output"]["label_filename"]
        props_path = self.results_dir / self.cfg["output"]["props_filename"]

        np.save(str(label_path), labels)
        df.to_csv(str(props_path), index=False)

        logger.info(f"[M4 Step 6] Labels saved → {label_path.name} "
                    f"(dtype={labels.dtype}, unique IDs={int(labels.max())})")
        logger.info(f"[M4 Step 6] Properties saved → {props_path.name} "
                    f"({len(df)} rows)")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        Run the full nuclear segmentation pipeline.

        Returns:
            result (dict): n_nuclei, label_path, props_path
        """
        logger.info("=" * 60)
        logger.info("Module 4 — Nuclear Segmentation: START")
        logger.info("=" * 60)
        t_total = time.time()

        dapi = self.load_dapi()
        masks_raw = self.segment(dapi)
        labels = self.filter_labels(masks_raw)
        df = self.extract_properties(labels)
        self.visualize_result(dapi, labels, df)
        self.save_outputs(labels, df)

        elapsed = time.time() - t_total
        logger.info("=" * 60)
        logger.info(f"Module 4 — Nuclear Segmentation: COMPLETE in {elapsed:.1f}s")
        logger.info(f"  Nuclei detected : {len(df)}")
        logger.info(f"  Area (mean)     : {df['area_px'].mean():.0f} px²")
        logger.info(f"  Area (min/max)  : {df['area_px'].min()} / {df['area_px'].max()} px²")
        logger.info("=" * 60)

        return {
            "n_nuclei": len(df),
            "label_path": str(self.results_dir / self.cfg["output"]["label_filename"]),
            "props_path": str(self.results_dir / self.cfg["output"]["props_filename"]),
        }
