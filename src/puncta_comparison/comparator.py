"""
Puncta Detection Method Comparator
====================================
Runs all 6 puncta detection methods on the same smFISH data and
produces a unified comparison table + QC figures.

Pipeline per method:
  1. Load spatially bg-corrected image crops (same correction as Module 5)
  2. For each nucleus × round × channel: compute signal value via method
  3. Argmax over channels → colour call → barcode
  4. Compute: decoded rate, SNR, barcode distribution, pairwise agreement

Output (python_results/puncta_comparison/):
  comparison_table.csv          — per nucleus × round × channel × method signal + calls
  qc_decoded_rate.png           — bar chart: % decoded per method
  qc_snr_distribution.png       — violin: SNR per method
  qc_pairwise_agreement.png     — 6×6 agreement heatmap
  qc_barcode_distribution.png   — top barcode counts per method
  qc_disagreement_overlay.png   — FOV spatial map of cells where methods disagree
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yaml
from aicsimageio import AICSImage
from loguru import logger
from scipy.ndimage import shift as ndimage_shift
from skimage.measure import regionprops

from .methods import METHODS, METHOD_LABELS

CHANNELS   = ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]
ROUNDS     = ["Hyb4", "Hyb3", "Hyb2"]   # experimental order: Hyb4 first imaged → Hyb3 → Hyb2
COLOR_MAP  = {"Ch1_AF647": "Purple", "Ch2_AF590": "Blue", "Ch3_AF488": "Yellow"}
NO_SIGNAL  = "None"
THRESHOLD  = 500          # ADU — applied to _corr / _xr values before argmax

DISPLAY_COLORS = {
    "Purple": "#9B59B6",
    "Blue":   "#3498DB",
    "Yellow": "#F4D03F",
    "None":   "#95A5A6",
}

METHOD_COLORS = {
    "X": "#2C3E50",
    "Y": "#E74C3C",
    "Z": "#3498DB",
    "W": "#2ECC71",
    "T": "#E67E22",
    "P": "#9B59B6",
}


class PunctaComparator:
    """Orchestrates cross-method puncta detection comparison."""

    def __init__(self, config_path: str, project_root: str = "."):
        self.project_root = Path(project_root)
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.out_dir = self.project_root / self.cfg["output"]["results_dir"]
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Load crop coords (for ICC image cropping)
        coords_path = self.project_root / self.cfg["input"]["crop_coords_path"]
        with open(coords_path) as f:
            self.crop_coords = json.load(f)

        self.methods_to_run = self.cfg["processing"].get("methods", list(METHODS.keys()))
        self.method_params  = self.cfg["processing"].get("method_params", {})

        logger.info(f"PunctaComparator initialized | root={self.project_root}")
        logger.info(f"Methods: {self.methods_to_run}")
        logger.info(f"Output: {self.out_dir}")

    # ──────────────────────────────────────────────────────────────────
    # Data loading helpers
    # ──────────────────────────────────────────────────────────────────
    def _load_nucleus_data(self):
        labels = np.load(str(self.project_root / self.cfg["input"]["nucleus_labels_path"]))
        props  = pd.read_csv(str(self.project_root / self.cfg["input"]["nucleus_props_path"]))
        logger.info(f"Nucleus labels: {labels.shape}, {int(labels.max())} nuclei")
        return labels, props

    def _load_spot_intensities(self) -> pd.DataFrame:
        path = self.project_root / self.cfg["input"]["spot_intensities_path"]
        df   = pd.read_csv(str(path))
        logger.info(f"spot_intensities.csv: {len(df)} rows")
        return df

    def _load_hyb4_channels(self) -> dict:
        """Returns {ch: 2-D array} for Hyb4 (pre-cropped by M1)."""
        inp = self.cfg["input"]
        return {
            "Ch1_AF647": np.load(str(self.project_root / inp["hyb4_ch1_path"])),
            "Ch2_AF590": np.load(str(self.project_root / inp["hyb4_ch2_path"])),
            "Ch3_AF488": np.load(str(self.project_root / inp["hyb4_ch3_path"])),
        }

    def _load_icc_channels(self, icc_path: Path) -> dict:
        """Load and crop a HybN ICC_Processed TIF. Returns {ch: 2-D array}."""
        y0, x0 = self.crop_coords["y0"], self.crop_coords["x0"]
        h,  w  = self.crop_coords["crop_h"], self.crop_coords["crop_w"]
        icc_ch = self.cfg["processing"]["icc_channels"]

        img = AICSImage(str(icc_path))
        out = {}
        for ch_name, z_idx in icc_ch.items():
            full = img.get_image_data("TCZYX")[0, 0, z_idx]
            out[ch_name] = full[y0:y0 + h, x0:x0 + w]
        logger.info(f"  Loaded {icc_path.name}")
        return out

    def _load_registration(self, reg_path: Path) -> tuple:
        with open(reg_path) as f:
            d = json.load(f)
        return float(d["dy"]), float(d["dx"])

    def _shift_labels(self, labels: np.ndarray, dy: float, dx: float) -> np.ndarray:
        """Shift Hyb4 nucleus mask into HybN coordinate frame."""
        if abs(dy) < 0.01 and abs(dx) < 0.01:
            return labels.astype(np.int32)
        return ndimage_shift(
            labels.astype(np.float32), shift=(-dy, -dx),
            order=0, mode="constant", cval=0
        ).astype(np.int32)

    # ──────────────────────────────────────────────────────────────────
    # Core: compute signals for all image-based methods
    # ──────────────────────────────────────────────────────────────────
    def _compute_image_signals(
        self,
        labels: np.ndarray,
        channels: dict,
        df_bg: pd.DataFrame,        # background scalars from spot_intensities.csv
        rnd: str,
        dy: float,
        dx: float,
    ) -> pd.DataFrame:
        """
        For one hybridisation round, compute per-nucleus × per-channel signals
        for all image-based methods (Y, Z, W, T, P).

        Background subtraction (per nucleus per channel):
            corr_crop = max(raw_crop - bg_scalar, 0)
            where bg_scalar = _bg column from spot_intensities.csv
            This is identical to Module 5's _corr correction.

        Returns a DataFrame indexed by nucleus_id with columns:
            {method}_{channel}  for each (method, channel) pair.
        """
        labels_shifted = self._shift_labels(labels, dy, dx)
        props          = regionprops(labels_shifted)

        # Index df_bg by nucleus_id for fast lookup
        bg_idx = df_bg[df_bg["round"] == rnd].set_index("nucleus_id")

        image_methods = [m for m in self.methods_to_run if m != "X"]
        params        = self.method_params

        records = []
        for rp in props:
            nid = rp.label
            if nid not in bg_idx.index:
                continue

            r0, c0, r1, c1 = rp.bbox
            # Local mask within bounding box
            local_mask = (labels_shifted[r0:r1, c0:c1] == nid)

            rec = {"nucleus_id": nid}
            for ch in CHANNELS:
                arr    = channels[ch]
                bg_val = float(bg_idx.loc[nid, f"{ch}_bg"]) if f"{ch}_bg" in bg_idx.columns else 0.0
                raw_crop  = arr[r0:r1, c0:c1].astype(np.float64)
                corr_crop = np.maximum(raw_crop - bg_val, 0.0)

                for m_key in image_methods:
                    fn  = METHODS[m_key]
                    sig = fn(corr_crop, local_mask, **params.get(m_key, {}))
                    rec[f"{m_key}_{ch}"] = sig

            records.append(rec)

        return pd.DataFrame(records)

    # ──────────────────────────────────────────────────────────────────
    # Step 1: Build full signal table
    # ──────────────────────────────────────────────────────────────────
    def build_signal_table(self, labels, df_int) -> pd.DataFrame:
        """
        Assemble per-nucleus × round × channel signal values for all methods.
        Method X values are read directly from spot_intensities.csv (_corr/_xr).
        Methods Y/Z/W/T/P are computed from image crops.
        """
        inp = self.cfg["input"]

        # Registration offsets
        reg_hyb2 = self._load_registration(
            self.project_root / inp["reg_hyb2_path"])
        reg_hyb3 = self._load_registration(
            self.project_root / inp["reg_hyb3_path"])
        reg = {"Hyb4": (0.0, 0.0), "Hyb3": reg_hyb3, "Hyb2": reg_hyb2}

        # Images per round
        round_images = {
            "Hyb4": self._load_hyb4_channels(),
            "Hyb3": self._load_icc_channels(self.project_root / inp["hyb3_icc_path"]),
            "Hyb2": self._load_icc_channels(self.project_root / inp["hyb2_icc_path"]),
        }

        all_parts = []

        for rnd in ROUNDS:
            logger.info(f"  Processing {rnd}...")
            dy, dx = reg[rnd]
            channels = round_images[rnd]

            # ── Method X: from CSV ──────────────────────────────────────
            df_rnd = df_int[df_int["round"] == rnd].copy()
            # Use same column selection as Module 6:
            #   Ch1/Ch3 → _corr,  Ch2 → _xr
            x_cols = {
                "Ch1_AF647": "Ch1_AF647_corr",
                "Ch2_AF590": "Ch2_AF590_xr",
                "Ch3_AF488": "Ch3_AF488_corr",
            }
            df_x = df_rnd[["nucleus_id"] + list(x_cols.values())].copy()
            df_x = df_x.rename(columns={v: f"X_{k}" for k, v in x_cols.items()})
            df_x["round"] = rnd

            # ── Methods Y/Z/W/T/P: from image crops ────────────────────
            if any(m in self.methods_to_run for m in ["Y", "Z", "W", "T", "P"]):
                df_img = self._compute_image_signals(
                    labels, channels, df_rnd, rnd, dy, dx
                )
                df_img["round"] = rnd
                merged = df_x.merge(df_img, on=["nucleus_id", "round"], how="left")
            else:
                merged = df_x

            all_parts.append(merged)

        df_signals = pd.concat(all_parts, ignore_index=True)
        logger.info(f"Signal table: {len(df_signals)} rows × {len(df_signals.columns)} cols")
        return df_signals

    # ──────────────────────────────────────────────────────────────────
    # Step 2: Decode barcodes for each method
    # ──────────────────────────────────────────────────────────────────
    def decode_all(self, df_signals: pd.DataFrame, df_props: pd.DataFrame) -> dict:
        """
        For each method, decode colour calls and barcodes.
        Returns dict: {method_key → barcodes DataFrame}
        """
        results = {}

        for m_key in self.methods_to_run:
            logger.info(f"  Decoding Method {m_key}...")
            barcodes = self._decode_one_method(df_signals, m_key)
            barcodes = barcodes.merge(
                df_props[["nucleus_id", "centroid_y", "centroid_x"]],
                on="nucleus_id", how="left"
            )
            results[m_key] = barcodes

            n_ok  = barcodes["decoded_ok"].sum()
            n_tot = len(barcodes)
            logger.info(
                f"    {m_key}: {n_ok}/{n_tot} decoded ({100*n_ok/n_tot:.1f}%), "
                f"{barcodes[barcodes['decoded_ok']]['barcode'].nunique()} unique barcodes"
            )

        return results

    def _decode_one_method(self, df_signals: pd.DataFrame, m_key: str) -> pd.DataFrame:
        """Argmax decoding for one method — same logic as Module 6."""
        thresh = THRESHOLD

        round_series = {}
        for rnd in ROUNDS:
            df_rnd = df_signals[df_signals["round"] == rnd]
            if len(df_rnd) == 0:
                continue

            colors = {}
            for _, row in df_rnd.iterrows():
                nid  = int(row["nucleus_id"])
                vals = {ch: float(row.get(f"{m_key}_{ch}", 0.0)) for ch in CHANNELS}
                max_val = max(vals.values())
                if max_val < thresh:
                    colors[nid] = NO_SIGNAL
                else:
                    best_ch    = max(vals, key=vals.get)
                    colors[nid] = COLOR_MAP[best_ch]

            round_series[rnd] = pd.Series(colors, name=rnd)

        all_ids = sorted(set().union(*[s.index for s in round_series.values()]))
        df_bc   = pd.DataFrame({"nucleus_id": all_ids})
        for rnd, series in round_series.items():
            df_bc[f"color_{rnd.lower()}"] = df_bc["nucleus_id"].map(series).fillna(NO_SIGNAL)

        round_cols = [f"color_{r.lower()}" for r in ROUNDS if f"color_{r.lower()}" in df_bc.columns]
        df_bc["barcode"]    = df_bc[round_cols].apply(lambda r: "-".join(r.values), axis=1)
        df_bc["decoded_ok"] = df_bc[round_cols].apply(lambda r: NO_SIGNAL not in r.values, axis=1)
        return df_bc

    # ──────────────────────────────────────────────────────────────────
    # Step 3: Compute metrics
    # ──────────────────────────────────────────────────────────────────
    def compute_metrics(self, results: dict, df_signals: pd.DataFrame) -> dict:
        """
        Compute per-method: decoded rate, SNR distribution, barcode distribution.
        Also compute pairwise agreement matrix across all methods.
        """
        metrics = {}

        for m_key, df_bc in results.items():
            n_tot  = len(df_bc)
            n_ok   = int(df_bc["decoded_ok"].sum())
            top_bc = df_bc[df_bc["decoded_ok"]]["barcode"].value_counts().head(10)

            # SNR: for each decoded nucleus × round, signal/max(other two channels)
            snr_vals = []
            for rnd in ROUNDS:
                df_rnd = df_signals[df_signals["round"] == rnd]
                col    = f"color_{rnd.lower()}"
                for _, row in df_rnd.iterrows():
                    nid = int(row["nucleus_id"])
                    bc_row = df_bc[df_bc["nucleus_id"] == nid]
                    if len(bc_row) == 0 or not bc_row.iloc[0]["decoded_ok"]:
                        continue
                    color = bc_row.iloc[0][col]
                    if color == NO_SIGNAL:
                        continue
                    # Reverse map: colour → channel
                    sig_ch  = [ch for ch, c in COLOR_MAP.items() if c == color]
                    if not sig_ch:
                        continue
                    sig_ch  = sig_ch[0]
                    other_ch = [ch for ch in CHANNELS if ch != sig_ch]
                    sig_val  = float(row.get(f"{m_key}_{sig_ch}", 0.0))
                    bg_vals  = [float(row.get(f"{m_key}_{ch}", 1.0)) for ch in other_ch]
                    denom    = max(max(bg_vals), 1.0)
                    snr_vals.append(sig_val / denom)

            metrics[m_key] = {
                "decoded_rate": 100.0 * n_ok / max(n_tot, 1),
                "n_decoded":    n_ok,
                "n_total":      n_tot,
                "snr_values":   snr_vals,
                "snr_median":   float(np.median(snr_vals)) if snr_vals else 0.0,
                "top_barcodes": top_bc,
            }

        # Pairwise agreement matrix
        all_keys = list(results.keys())
        agree_mat = pd.DataFrame(np.nan, index=all_keys, columns=all_keys)

        for i, ka in enumerate(all_keys):
            for j, kb in enumerate(all_keys):
                if i == j:
                    agree_mat.loc[ka, kb] = 100.0
                    continue
                df_a = results[ka][["nucleus_id", "barcode"]].rename(
                    columns={"barcode": "bc_a"})
                df_b = results[kb][["nucleus_id", "barcode"]].rename(
                    columns={"barcode": "bc_b"})
                merged = df_a.merge(df_b, on="nucleus_id")
                if len(merged) == 0:
                    agree_mat.loc[ka, kb] = np.nan
                else:
                    pct = 100.0 * (merged["bc_a"] == merged["bc_b"]).mean()
                    agree_mat.loc[ka, kb] = pct

        metrics["__agreement__"] = agree_mat
        return metrics

    # ──────────────────────────────────────────────────────────────────
    # Step 4: Visualisation
    # ──────────────────────────────────────────────────────────────────
    def visualize(self, results: dict, metrics: dict):
        agree_mat = metrics.pop("__agreement__")
        m_keys    = [m for m in self.methods_to_run if m in results]

        self._fig_decoded_rate(metrics, m_keys)
        self._fig_snr_violin(metrics, m_keys)
        self._fig_agreement(agree_mat, m_keys)
        self._fig_barcode_dist(metrics, m_keys)
        self._fig_disagreement_map(results, m_keys)

        metrics["__agreement__"] = agree_mat   # restore

    def _fig_decoded_rate(self, metrics, m_keys):
        fig, ax = plt.subplots(figsize=(8, 4))
        rates = [metrics[m]["decoded_rate"] for m in m_keys]
        bars  = ax.bar(m_keys, rates,
                       color=[METHOD_COLORS.get(m, "#888") for m in m_keys],
                       width=0.6, edgecolor="white", linewidth=1.2)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{rate:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Decoded rate (%)", fontsize=10)
        ax.set_title("Puncta Detection Method Comparison — Decoded Rate\n"
                     f"(threshold={THRESHOLD} ADU, n={metrics[m_keys[0]]['n_total']} nuclei)",
                     fontsize=10)
        ax.set_xticklabels(
            [METHOD_LABELS[m].replace("\n", " ") for m in m_keys], fontsize=8)
        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out = self.out_dir / "qc_decoded_rate.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved → {out.name}")

    def _fig_snr_violin(self, metrics, m_keys):
        data   = [metrics[m]["snr_values"] for m in m_keys]
        colors = [METHOD_COLORS.get(m, "#888") for m in m_keys]

        fig, ax = plt.subplots(figsize=(9, 4))
        parts = ax.violinplot(data, positions=range(len(m_keys)),
                              showmedians=True, showextrema=False)
        for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
            pc.set_facecolor(col); pc.set_alpha(0.7)
        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)

        ax.set_xticks(range(len(m_keys)))
        ax.set_xticklabels(
            [METHOD_LABELS[m].replace("\n", " ") for m in m_keys], fontsize=8)
        ax.set_ylabel("SNR (signal ch / max other ch)", fontsize=10)
        ax.set_title("SNR Distribution per Method (decoded nuclei only)", fontsize=10)
        ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out = self.out_dir / "qc_snr_distribution.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved → {out.name}")

    def _fig_agreement(self, agree_mat, m_keys):
        mat = agree_mat.loc[m_keys, m_keys].values.astype(float)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, vmin=60, vmax=100, cmap="YlGn")
        plt.colorbar(im, ax=ax, label="Barcode agreement (%)")
        ax.set_xticks(range(len(m_keys))); ax.set_yticks(range(len(m_keys)))
        ax.set_xticklabels(
            [METHOD_LABELS[m].replace("\n", " ") for m in m_keys],
            rotation=30, ha="right", fontsize=7)
        ax.set_yticklabels(
            [METHOD_LABELS[m].replace("\n", " ") for m in m_keys], fontsize=7)
        for i in range(len(m_keys)):
            for j in range(len(m_keys)):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            fontsize=7, color="black" if val < 85 else "white")
        ax.set_title("Pairwise Barcode Agreement Matrix", fontsize=10)
        plt.tight_layout()
        out = self.out_dir / "qc_pairwise_agreement.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved → {out.name}")

    def _fig_barcode_dist(self, metrics, m_keys):
        # Collect all unique barcodes across methods
        all_bc = set()
        for m in m_keys:
            all_bc |= set(metrics[m]["top_barcodes"].index)
        all_bc = sorted(all_bc)

        n_bc = len(all_bc)
        fig, ax = plt.subplots(figsize=(max(12, n_bc * 0.8), 5))
        x       = np.arange(n_bc)
        width   = 0.8 / len(m_keys)

        for i, m in enumerate(m_keys):
            counts = [metrics[m]["top_barcodes"].get(bc, 0) for bc in all_bc]
            ax.bar(x + i * width, counts, width,
                   label=METHOD_LABELS[m].replace("\n", " "),
                   color=METHOD_COLORS.get(m, "#888"), alpha=0.85)

        ax.set_xticks(x + width * (len(m_keys) - 1) / 2)
        ax.set_xticklabels(all_bc, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("Cell count", fontsize=10)
        ax.set_title("Barcode Distribution — All Methods (decoded_ok cells only)", fontsize=10)
        ax.legend(fontsize=7, ncol=3)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out = self.out_dir / "qc_barcode_distribution.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved → {out.name}")

    def _fig_disagreement_map(self, results, m_keys):
        """Spatial FOV map coloured by number of methods that agree on the call."""
        # For each nucleus: count how many unique barcodes are called
        nids = results[m_keys[0]]["nucleus_id"].values
        bc_df = pd.DataFrame({"nucleus_id": nids})
        for m in m_keys:
            bc_df = bc_df.merge(
                results[m][["nucleus_id", "barcode", "centroid_y", "centroid_x"]],
                on="nucleus_id", how="left", suffixes=("", f"_{m}")
            ).rename(columns={"barcode": f"bc_{m}"})

        bc_cols    = [f"bc_{m}" for m in m_keys]
        n_unique   = bc_df[bc_cols].nunique(axis=1)
        bc_df["n_unique_calls"] = n_unique

        # Get centroids (from any method)
        if "centroid_y" not in bc_df.columns:
            for m in m_keys:
                if f"centroid_y_{m}" in bc_df.columns:
                    bc_df["centroid_y"] = bc_df[f"centroid_y_{m}"]
                    bc_df["centroid_x"] = bc_df[f"centroid_x_{m}"]
                    break

        fig, ax = plt.subplots(figsize=(8, 7))
        scatter_df = bc_df.dropna(subset=["centroid_y", "centroid_x"])

        sc = ax.scatter(
            scatter_df["centroid_x"], scatter_df["centroid_y"],
            c=scatter_df["n_unique_calls"], cmap="RdYlGn_r",
            vmin=1, vmax=len(m_keys), s=6, alpha=0.85
        )
        plt.colorbar(sc, ax=ax, label="# distinct barcode calls across methods\n(1=full agreement, >1=disagreement)")
        ax.invert_yaxis()
        ax.set_xlabel("x (px, Hyb4 frame)", fontsize=8)
        ax.set_ylabel("y (px, Hyb4 frame)", fontsize=8)
        ax.set_title("Spatial Map of Method Agreement\n"
                     "Green = all methods agree | Red = methods disagree", fontsize=9)

        n_agree    = (scatter_df["n_unique_calls"] == 1).sum()
        n_disagree = (scatter_df["n_unique_calls"] > 1).sum()
        ax.text(0.02, 0.98,
                f"Full agreement: {n_agree} ({100*n_agree/len(scatter_df):.1f}%)\n"
                f"Disagreement: {n_disagree} ({100*n_disagree/len(scatter_df):.1f}%)",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()
        out = self.out_dir / "qc_disagreement_map.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved → {out.name}")

    # ──────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────
    def run(self):
        logger.info("=" * 60)
        logger.info("Puncta Detection Comparison: START")
        logger.info("=" * 60)
        t0 = time.time()

        labels, df_props = self._load_nucleus_data()
        df_int           = self._load_spot_intensities()

        # Step 1: Signal table
        logger.info("[Step 1] Building signal table for all methods...")
        df_signals = self.build_signal_table(labels, df_int)
        df_signals.to_csv(str(self.out_dir / "signal_table.csv"), index=False)
        logger.info(f"  signal_table.csv saved ({len(df_signals)} rows)")

        # Step 2: Decode
        logger.info("[Step 2] Decoding barcodes for all methods...")
        results = self.decode_all(df_signals, df_props)

        # Step 3: Metrics
        logger.info("[Step 3] Computing metrics...")
        metrics = self.compute_metrics(results, df_signals)

        # Print summary table
        agree_mat = metrics["__agreement__"]
        logger.info("\n" + "─" * 55)
        logger.info(f"{'Method':<8} {'Decoded%':>9} {'SNR median':>11}")
        logger.info("─" * 55)
        for m in self.methods_to_run:
            if m in metrics:
                logger.info(
                    f"  {m:<6} {metrics[m]['decoded_rate']:>8.1f}%  "
                    f"{metrics[m]['snr_median']:>10.1f}"
                )
        logger.info("─" * 55)
        logger.info("Pairwise barcode agreement (%):")
        logger.info("\n" + agree_mat.to_string(float_format="{:.1f}".format))

        # Step 4: Save per-method barcodes + combined comparison CSV
        logger.info("[Step 4] Saving outputs...")
        for m, df_bc in results.items():
            df_bc.to_csv(str(self.out_dir / f"barcodes_{m}.csv"), index=False)

        # Wide comparison table: one row per nucleus, barcode call from each method
        nids = results[self.methods_to_run[0]]["nucleus_id"].values
        comp = pd.DataFrame({"nucleus_id": nids})
        for m, df_bc in results.items():
            sub = df_bc[["nucleus_id", "barcode", "decoded_ok",
                          "color_hyb2", "color_hyb3", "color_hyb4"]].copy()
            sub = sub.rename(columns={
                "barcode":     f"barcode_{m}",
                "decoded_ok":  f"decoded_ok_{m}",
                "color_hyb2":  f"color_hyb2_{m}",
                "color_hyb3":  f"color_hyb3_{m}",
                "color_hyb4":  f"color_hyb4_{m}",
            })
            comp = comp.merge(sub, on="nucleus_id", how="left")
        comp.to_csv(str(self.out_dir / "comparison_table.csv"), index=False)
        logger.info(f"  comparison_table.csv saved ({len(comp)} rows)")

        # Step 5: Figures
        logger.info("[Step 5] Generating QC figures...")
        self.visualize(results, metrics)

        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info(f"Puncta Detection Comparison: COMPLETE in {elapsed:.1f}s")
        logger.info(f"Outputs in: {self.out_dir}")
        logger.info("=" * 60)

        return results, metrics
