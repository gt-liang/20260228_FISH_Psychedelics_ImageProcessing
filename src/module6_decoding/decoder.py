"""
Module 6 — Barcode Decoding
============================
Decode a 3-round × 3-channel fluorescence barcode for each nucleus.

Scientific Logic:
    Each cell pool was labeled with a unique combination of fluorescent oligos
    across 3 hybridization rounds (Hyb2, Hyb3, Hyb4). In each round, exactly
    ONE channel (AF647, AF590, or AF488) carries a real spot.

    Decoding (per nucleus per round):
      1. Read max intensities from M5: (Ch1_AF647, Ch2_AF590, Ch3_AF488)
      2. If max(all three) < background_threshold → no signal → color = "None"
      3. Otherwise → argmax channel → color label
         Ch1_AF647 → Purple
         Ch2_AF590 → Blue
         Ch3_AF488 → Yellow

    The 3-round color sequence is the barcode:
      e.g. (Purple, Yellow, Blue) ↔ a specific drug or condition

    Background threshold:
      Set based on the M5 QC violin plot — the valley between the
      background distribution (~500-1000 ADU) and signal peaks.
      Default: 2000 ADU (conservative; adjust after visual inspection).

Inputs:
    - spot_intensities.csv (M5): nucleus_id × round × channel intensities
    - nucleus_properties.csv (M4): centroid coordinates for spatial QC

Outputs:
    - barcodes.csv: per nucleus — id, centroid, color per round, barcode string,
                    raw intensities, decoded_ok flag
    - module6_decoding_QC.png: spatial map of barcodes on the FOV
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from loguru import logger


# Colour palette for visualization (matches project convention)
DISPLAY_COLORS = {
    "Purple": "#9B59B6",
    "Blue":   "#3498DB",
    "Yellow": "#F4D03F",
    "None":   "#95A5A6",
}


class Decoder:
    """Decodes 3-round fluorescence barcodes from per-nucleus max intensities."""

    def __init__(self, config_path: str, project_root: str = "."):
        self.project_root = Path(project_root)
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.results_dir = self.project_root / self.cfg["output"]["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.color_map  = self.cfg["processing"]["color_map"]   # {ch_name: color_label}
        self.no_signal  = self.cfg["processing"]["no_signal_label"]
        self.use_corr   = self.cfg["processing"].get("use_background_subtracted", False)
        self.use_xr     = self.cfg["processing"].get("use_cross_round_correction", False)
        self.channels   = ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]  # canonical names

        # Column redirect: choose which correction layer to use per channel.
        #
        # use_cross_round_correction=true applies the cross-round minimum
        # subtraction (_xr) ONLY to Ch2_AF590 (Blue), where mCherry fluorescent
        # protein creates a persistent background across all rounds. Ch1 and Ch3
        # retain the spatial bg subtraction (_corr) because they do not have a
        # persistent-protein background problem — applying cross-round min to
        # those channels would incorrectly remove borderline Purple/Yellow signals.
        #
        #   Ch2_AF590: _xr  (spatial bg + cross-round mCherry removal)
        #   Ch1_AF647: _corr (spatial bg subtraction only)
        #   Ch3_AF488: _corr (spatial bg subtraction only)
        if self.use_xr and self.use_corr:
            self.ch_cols = {
                "Ch1_AF647": "Ch1_AF647_corr",  # spatial bg only — no persistent protein
                "Ch2_AF590": "Ch2_AF590_xr",    # mCherry cross-round correction
                "Ch3_AF488": "Ch3_AF488_corr",  # spatial bg only — no persistent protein
            }
            self.bg_thresh = self.cfg["processing"].get("cross_round_threshold", 200)
            mode_label     = "Ch2 mCherry-corrected (_xr), Ch1/Ch3 spatial bg (_corr)"
        elif self.use_corr:
            self.ch_cols   = {ch: f"{ch}_corr" for ch in self.channels}
            self.bg_thresh = self.cfg["processing"]["corrected_background_threshold"]
            mode_label     = "spatial bg subtracted (_corr)"
        else:
            self.ch_cols   = {ch: ch for ch in self.channels}
            self.bg_thresh = self.cfg["processing"]["background_threshold"]
            mode_label     = "raw max intensity"

        logger.info(f"Decoder initialized | project_root={self.project_root}")
        logger.info(f"Results dir: {self.results_dir}")
        logger.info(f"Mode: {mode_label}")
        logger.info(f"Background threshold: {self.bg_thresh} ADU")

    # ------------------------------------------------------------------
    # Step 1: Load M5 intensities and M4 properties
    # ------------------------------------------------------------------
    def load_data(self):
        intensity_path = self.project_root / self.cfg["input"]["intensity_path"]
        props_path = self.project_root / self.cfg["input"]["nucleus_props_path"]

        logger.info(f"[M6 Step 1] Loading intensity table: {intensity_path.name}")
        df_int = pd.read_csv(str(intensity_path))
        logger.info(f"  {len(df_int)} rows, {df_int['nucleus_id'].nunique()} nuclei, "
                    f"rounds: {sorted(df_int['round'].unique())}")

        logger.info(f"[M6 Step 1] Loading nucleus properties: {props_path.name}")
        df_props = pd.read_csv(str(props_path))

        return df_int, df_props

    # ------------------------------------------------------------------
    # Step 2: Decode one round for all nuclei
    # ------------------------------------------------------------------
    def decode_round(self, df_round: pd.DataFrame, round_name: str) -> pd.Series:
        """
        For each nucleus in df_round, assign a color label:
          - If max(channels) < background_threshold → "None"
          - Otherwise → color of argmax channel

        Returns a Series indexed by nucleus_id with color labels.
        """
        colors = {}
        for _, row in df_round.iterrows():
            nid = int(row["nucleus_id"])
            # self.ch_cols maps canonical channel name → CSV column to read
            # (raw: Ch1_AF647, or corrected: Ch1_AF647_corr)
            vals = {ch: float(row[self.ch_cols[ch]]) for ch in self.channels}
            max_val = max(vals.values())

            if max_val < self.bg_thresh:
                colors[nid] = self.no_signal
            else:
                best_ch = max(vals, key=vals.get)
                colors[nid] = self.color_map[best_ch]   # color_map keyed by canonical name

        return pd.Series(colors, name=round_name)

    # ------------------------------------------------------------------
    # Step 3: Decode all rounds and build barcode table
    # ------------------------------------------------------------------
    def decode(self, df_int: pd.DataFrame, df_props: pd.DataFrame) -> pd.DataFrame:
        """
        Decode all 3 rounds for all nuclei and assemble the barcode table.

        Barcode string format: "Color4-Color3-Color2"  (experimental order: Hyb4 → Hyb3 → Hyb2)
        e.g. "Purple-Yellow-Blue"

        A nucleus is flagged decoded_ok=False if ANY round is "None".
        """
        logger.info(f"[M6 Step 2] Decoding per round (background threshold={self.bg_thresh} ADU)")

        round_series = {}
        for rnd in ["Hyb2", "Hyb3", "Hyb4"]:
            df_rnd = df_int[df_int["round"] == rnd]
            if len(df_rnd) == 0:
                logger.warning(f"  No data for round {rnd} — skipping")
                continue
            color_series = self.decode_round(df_rnd, rnd)
            round_series[rnd] = color_series

            # Log color distribution for this round
            counts = color_series.value_counts()
            logger.info(f"  {rnd}: " +
                        " | ".join(f"{color}={cnt}" for color, cnt in counts.items()))

        # Merge all nucleus IDs across rounds
        all_ids = sorted(
            set().union(*[s.index for s in round_series.values()])
        )
        df_decode = pd.DataFrame({"nucleus_id": all_ids})

        for rnd, series in round_series.items():
            df_decode[f"color_{rnd.lower()}"] = df_decode["nucleus_id"].map(series).fillna(self.no_signal)

        # Build barcode string — experimental order: Hyb4 (first imaged) → Hyb3 → Hyb2
        round_cols = [f"color_{r.lower()}" for r in ["Hyb4", "Hyb3", "Hyb2"]
                      if f"color_{r.lower()}" in df_decode.columns]
        df_decode["barcode"] = df_decode[round_cols].apply(
            lambda row: "-".join(row.values), axis=1
        )
        df_decode["decoded_ok"] = df_decode[round_cols].apply(
            lambda row: self.no_signal not in row.values, axis=1
        )

        # Merge centroid coordinates from M4
        df_decode = df_decode.merge(
            df_props[["nucleus_id", "centroid_y", "centroid_x", "area_px"]],
            on="nucleus_id", how="left"
        )

        # Append raw, _bg, _corr, _xr_bg, _xr intensities for traceability
        for rnd in ["Hyb2", "Hyb3", "Hyb4"]:
            extra_suffixes = ["_bg", "_corr", "_xr_bg", "_xr"]
            trace_cols = ["nucleus_id"] + self.channels
            extra_cols = [f"{ch}{sfx}" for ch in self.channels for sfx in extra_suffixes]
            for c in extra_cols:
                if c in df_int.columns:
                    trace_cols.append(c)
            df_rnd = df_int[df_int["round"] == rnd][trace_cols].copy()
            rename_map = {ch: f"{rnd}_{ch}" for ch in self.channels}
            for c in extra_cols:
                if c in df_rnd.columns:
                    rename_map[c] = f"{rnd}_{c}"
            df_rnd = df_rnd.rename(columns=rename_map)
            df_decode = df_decode.merge(df_rnd, on="nucleus_id", how="left")

        n_total = len(df_decode)
        n_ok = df_decode["decoded_ok"].sum()
        n_none = n_total - n_ok
        logger.info(f"[M6 Step 3] Decoding summary:")
        logger.info(f"  Total nuclei   : {n_total}")
        logger.info(f"  Fully decoded  : {n_ok} ({100*n_ok/n_total:.1f}%)")
        logger.info(f"  Has None round : {n_none} ({100*n_none/n_total:.1f}%)")
        logger.info(f"  Unique barcodes: {df_decode['barcode'].nunique()}")

        # Show top barcodes
        top_barcodes = df_decode[df_decode["decoded_ok"]]["barcode"].value_counts().head(10)
        logger.info(f"  Top barcodes (decoded_ok only):")
        for bc, cnt in top_barcodes.items():
            logger.info(f"    {bc}: {cnt} nuclei")

        return df_decode

    # ------------------------------------------------------------------
    # Step 4: QC visualization — spatial barcode map
    # ------------------------------------------------------------------
    def visualize_result(self, df_decode: pd.DataFrame):
        """
        Scatter plot of nucleus centroids colored by Hyb4 color (or barcode).
        Gives a spatial overview of which barcodes are where in the FOV.
        """
        logger.info(f"[M6 Step 4] Generating barcode spatial QC visualization")

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        rounds = ["Hyb2", "Hyb3", "Hyb4"]

        for ax, rnd in zip(axes, rounds):
            col = f"color_{rnd.lower()}"
            if col not in df_decode.columns:
                ax.set_visible(False)
                continue

            for color_label, grp in df_decode.groupby(col):
                ax.scatter(
                    grp["centroid_x"], grp["centroid_y"],
                    c=DISPLAY_COLORS.get(color_label, "#95A5A6"),
                    s=8, alpha=0.8, label=f"{color_label} (n={len(grp)})"
                )
            ax.set_title(f"{rnd} color calls", fontsize=10)
            ax.set_xlabel("x (px, Hyb4 frame)", fontsize=8)
            ax.set_ylabel("y (px, Hyb4 frame)", fontsize=8)
            ax.invert_yaxis()   # image coordinates: y increases downward
            ax.legend(fontsize=7, markerscale=2)

        fig.suptitle(
            f"Module 6 QC — Barcode Spatial Map\n"
            f"background threshold={self.bg_thresh} ADU | "
            f"decoded_ok: {df_decode['decoded_ok'].sum()}/{len(df_decode)} nuclei",
            fontsize=11
        )
        plt.tight_layout()
        out_path = self.results_dir / "module6_decoding_QC.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  QC image saved → {out_path.name}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """
        Run the full barcode decoding pipeline.

        Returns:
            df_barcodes (pd.DataFrame): one row per nucleus with barcode info
        """
        logger.info("=" * 60)
        logger.info("Module 6 — Barcode Decoding: START")
        logger.info("=" * 60)
        t_total = time.time()

        df_int, df_props = self.load_data()
        df_barcodes = self.decode(df_int, df_props)
        self.visualize_result(df_barcodes)

        out_path = self.results_dir / self.cfg["output"]["barcode_filename"]
        df_barcodes.to_csv(str(out_path), index=False)
        logger.info(f"[M6 Step 5] Barcodes saved → {out_path.name} ({len(df_barcodes)} rows)")

        elapsed = time.time() - t_total
        logger.info("=" * 60)
        logger.info(f"Module 6 — Barcode Decoding: COMPLETE in {elapsed:.1f}s")
        logger.info("=" * 60)

        return df_barcodes
