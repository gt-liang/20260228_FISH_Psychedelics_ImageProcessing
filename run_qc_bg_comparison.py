"""
QC — Background Subtraction Before vs After Comparison
======================================================
Usage:
    python run_qc_bg_comparison.py

Generates three comparison figures that quantify the effect of per-nucleus
background subtraction on channel signal quality and barcode decoding calls.

Scientific Purpose:
    Per-nucleus background subtraction (median pedestal removal) is applied in
    Module 5. This script isolates the effect by:
      1. Re-deriving "before" calls from RAW intensities with threshold=2000 ADU
      2. Using "after" calls from barcodes.csv (bg-subtracted, threshold=500 ADU)
    Then comparing which nuclei changed, in which direction, and whether the
    changes are biologically plausible.

Outputs (python_results/qc/):
  qc_bg_scatter_comparison.png  — channel scatter raw vs corrected per round
  qc_bg_background_magnitude.png — per-nucleus background & signal distributions
  qc_bg_call_changes.png        — decoding change analysis per round
"""

import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

QC_DIR = PROJECT_ROOT / "python_results" / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

# ─── Constants ───────────────────────────────────────────────────────────────
PALETTE = {
    "Purple": "#9B59B6",
    "Blue":   "#3498DB",
    "Yellow": "#F4D03F",
    "None":   "#95A5A6",
}
CHANNELS  = ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]
COLOR_MAP = {"Ch1_AF647": "Purple", "Ch2_AF590": "Blue", "Ch3_AF488": "Yellow"}
ROUNDS    = ["Hyb2", "Hyb3", "Hyb4"]

RAW_THRESH  = 2000   # original threshold on raw intensities
CORR_THRESH = 500    # threshold on background-subtracted intensities
CH_LABELS   = {"Ch1_AF647": "AF647\n(Purple)", "Ch2_AF590": "AF590\n(Blue)",
               "Ch3_AF488": "AF488\n(Yellow)"}


# ─── Helper: re-derive "before" calls from raw columns ───────────────────────

def derive_calls_raw(df_int: pd.DataFrame) -> pd.DataFrame:
    """
    Decode using RAW max intensities (threshold = RAW_THRESH).
    Returns df with [nucleus_id, round, color_before].
    Mirrors original Module 6 logic exactly.
    """
    rows = []
    for _, row in df_int.iterrows():
        vals    = {ch: float(row[ch]) for ch in CHANNELS}
        max_val = max(vals.values())
        if max_val < RAW_THRESH:
            color = "None"
        else:
            color = COLOR_MAP[max(vals, key=vals.get)]
        rows.append({"nucleus_id": int(row["nucleus_id"]),
                     "round":      row["round"],
                     "color_before": color})
    return pd.DataFrame(rows)


def derive_calls_corr(df_int: pd.DataFrame) -> pd.DataFrame:
    """
    Decode using CORRECTED intensities (threshold = CORR_THRESH).
    Values come from *_corr columns written by Module 5.
    """
    rows = []
    for _, row in df_int.iterrows():
        vals    = {ch: float(row[f"{ch}_corr"]) for ch in CHANNELS}
        max_val = max(vals.values())
        if max_val < CORR_THRESH:
            color = "None"
        else:
            color = COLOR_MAP[max(vals, key=vals.get)]
        rows.append({"nucleus_id": int(row["nucleus_id"]),
                     "round":      row["round"],
                     "color_after": color})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Channel Scatter: Raw vs Corrected (2 rows × 3 rounds)
# ─────────────────────────────────────────────────────────────────────────────

def fig_scatter_comparison(df_int: pd.DataFrame, df_calls: pd.DataFrame):
    """
    Row 1: Ch1_AF647 (raw) vs Ch3_AF488 (raw) — original axes
    Row 2: Ch1_AF647_corr vs Ch3_AF488_corr   — after background removal

    Both rows use the SAME color coding (post-correction decoded call) so you can
    see how the same nuclei redistribute in signal space after background removal.

    Key expectation:
      - In Row 1, Blue dots often cluster along the X=Y diagonal (background
        elevates both channels uniformly → looks like high signal in both).
      - In Row 2, Blue dots should move closer to origin in non-Blue channels,
        and Purple/Yellow dots should align more cleanly with their respective axis.
    """
    logger.info("Generating scatter comparison figure...")
    col_map = {"Hyb2": "color_after", "Hyb3": "color_after", "Hyb4": "color_after"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Background Subtraction — Channel Scatter Comparison\n"
        "Top row: RAW max intensities  |  Bottom row: Corrected (max − per-nucleus median)\n"
        "Dot color = post-correction decoded call  |  Expected: tighter clustering in bottom row",
        fontsize=12
    )

    for col_idx, rnd in enumerate(ROUNDS):
        df_r    = df_int[df_int["round"] == rnd].copy()
        df_col  = df_calls[df_calls["round"] == rnd][["nucleus_id", "color_after"]]
        df_r    = df_r.merge(df_col, on="nucleus_id", how="left")
        df_r["color_after"] = df_r["color_after"].fillna("None")

        for row_idx, (xcol, ycol, title) in enumerate([
            ("Ch1_AF647",       "Ch3_AF488",       "RAW max intensity (ADU)"),
            ("Ch1_AF647_corr",  "Ch3_AF488_corr",  "Corrected (max − bg median) (ADU)"),
        ]):
            ax = axes[row_idx, col_idx]

            for color_label, grp in df_r.groupby("color_after"):
                ax.scatter(
                    grp[xcol], grp[ycol],
                    c=PALETTE.get(color_label, "#95A5A6"),
                    s=10, alpha=0.6, label=f"{color_label} (n={len(grp)})"
                )

            # y=x guide
            lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([0, lim], [0, lim], "k--", lw=0.5, alpha=0.25)

            ax.set_xlabel(xcol.replace("_corr", " corr").replace("_", " "), fontsize=8)
            ax.set_ylabel(ycol.replace("_corr", " corr").replace("_", " "), fontsize=8)
            ax.legend(fontsize=6, markerscale=2)

            if row_idx == 0:
                ax.set_title(f"{rnd}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{title}\n\n{ycol.replace('_corr','').replace('_',' ')} corr" if row_idx else
                              f"{title}\n\n{ycol.replace('_',' ')} (raw)", fontsize=8)

    # Row labels
    for row_idx, lbl in enumerate(["RAW intensities", "BG-subtracted (corrected)"]):
        axes[row_idx, 0].set_ylabel(f"{lbl}\n\nCh3_AF488{'_corr' if row_idx else ''} (ADU)", fontsize=8)

    plt.tight_layout()
    out = QC_DIR / "qc_bg_scatter_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Background Magnitude per Channel per Round
# ─────────────────────────────────────────────────────────────────────────────

def fig_background_magnitude(df_int: pd.DataFrame):
    """
    For each round × channel: violin showing
      - Background (median per nucleus) — the pedestal to be removed
      - Raw max intensity — total signal before removal
      - Corrected max intensity — signal after pedestal removal

    Scientific interpretation:
      Channels where background/max is high have unreliable raw calls.
      After subtraction, only true punctum signal remains.
      Background/signal ratio > 30% indicates a problematic channel.
    """
    logger.info("Generating background magnitude figure...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Per-Nucleus Background Magnitude vs Signal per Channel\n"
        "Background = per-nucleus median intensity (pedestal: autofluorescence + carry-over)\n"
        "BG/Signal ratio > 30% → channel at risk of mis-assignment",
        fontsize=11
    )

    ch_colors = {
        "Ch1_AF647": PALETTE["Purple"],
        "Ch2_AF590": PALETTE["Blue"],
        "Ch3_AF488": PALETTE["Yellow"],
    }

    for ax, rnd in zip(axes, ROUNDS):
        df_r = df_int[df_int["round"] == rnd]

        positions = []
        data_bg   = []
        data_raw  = []
        data_corr = []
        tick_lbls = []
        bg_ratio  = []   # background / raw max (mean over nuclei)

        for i, ch in enumerate(CHANNELS):
            raw_vals  = df_r[ch].values
            bg_vals   = df_r[f"{ch}_bg"].values
            corr_vals = df_r[f"{ch}_corr"].values

            # Only include nuclei with detectable signal (raw > 500) for fair comparison
            sig_mask = raw_vals > 500
            positions.append(i + 1)
            data_bg.append(bg_vals[sig_mask])
            data_raw.append(raw_vals[sig_mask])
            data_corr.append(corr_vals[sig_mask])

            ratio = float(bg_vals[sig_mask].mean() / (raw_vals[sig_mask].mean() + 1e-8))
            bg_ratio.append(ratio)
            tick_lbls.append(f"{CH_LABELS[ch]}\nbg/raw={ratio:.0%}")

        # Violin: background distribution
        vp_bg = ax.violinplot(data_bg, positions=[p - 0.15 for p in positions],
                              widths=0.25, showmedians=True)
        for pc in vp_bg["bodies"]:
            pc.set_facecolor("#E8E8E8"); pc.set_alpha(0.8)
        vp_bg["cmedians"].set_color("gray")

        # Violin: corrected (post-subtraction)
        vp_corr = ax.violinplot(data_corr, positions=[p + 0.15 for p in positions],
                                widths=0.25, showmedians=True)
        for pc, ch in zip(vp_corr["bodies"], CHANNELS):
            pc.set_facecolor(ch_colors[ch]); pc.set_alpha(0.7)

        # Mark channels with high bg/signal ratio
        for i, (ratio, pos) in enumerate(zip(bg_ratio, positions)):
            if ratio > 0.30:
                ax.text(pos, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 500,
                        "⚠ high bg", ha="center", fontsize=7, color="red")

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_lbls, fontsize=7)
        ax.set_ylabel("Intensity (ADU)", fontsize=8)
        ax.set_title(f"{rnd}", fontsize=10)
        ax.set_yscale("log")

        # Legend (only on first panel)
        if rnd == "Hyb4":
            handles = [
                mpatches.Patch(color="#E8E8E8", label="Background (median)"),
                mpatches.Patch(color="#aaaaaa", label="Corrected signal (max−bg)"),
            ]
            ax.legend(handles=handles, fontsize=8, loc="upper left")

    plt.tight_layout()
    out = QC_DIR / "qc_bg_background_magnitude.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Decoding Call Changes: Before vs After
# ─────────────────────────────────────────────────────────────────────────────

def fig_call_changes(df_calls: pd.DataFrame):
    """
    For each round:
      Left panel:  horizontal stacked bars showing # nuclei whose call CHANGED
                   vs STAYED THE SAME, broken down by "before" color
      Right panel: transition matrix (before → after) as a heatmap
                   showing ONLY nuclei that changed

    Scientific interpretation:
      - Most cells that were called "None" (below threshold) before should now
        be called a real color → these were genuine signals below 2000 ADU raw
        but clearly above 500 ADU corrected.
      - Some "Blue" cells should flip to "Yellow" or "Purple" → these had their
        call driven by a high Blue background pedestal, not real Blue signal.
    """
    logger.info("Generating call change figure...")

    color_order = ["Purple", "Yellow", "Blue", "None"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Decoding Call Changes: Before (RAW, thresh=2000) vs After (Corrected, thresh=500)\n"
        "Top row: count of changed vs unchanged calls  |  "
        "Bottom row: transition heatmap for changed nuclei",
        fontsize=12
    )

    for col_idx, rnd in enumerate(ROUNDS):
        df_r = df_calls[df_calls["round"] == rnd].copy()
        n_total = len(df_r)

        # ── Top: Change count bar chart ──────────────────────────────────────
        ax_bar = axes[0, col_idx]

        df_r["changed"] = df_r["color_before"] != df_r["color_after"]
        n_changed   = df_r["changed"].sum()
        n_unchanged = n_total - n_changed

        # Break changed by "before" color
        changed_by_before = df_r[df_r["changed"]].groupby("color_before").size()

        bar_y   = 0
        bar_colors_all = []
        bar_widths = []
        bar_labels = []

        # Unchanged (gray)
        bar_widths.append(n_unchanged)
        bar_colors_all.append("#DDDDDD")
        bar_labels.append(f"Unchanged (n={n_unchanged})")

        # Changed — broken by previous color
        for c in color_order:
            if c in changed_by_before.index:
                n = int(changed_by_before[c])
                bar_widths.append(n)
                bar_colors_all.append(PALETTE[c])
                bar_labels.append(f"Changed from {c} (n={n})")

        lefts = [0]
        for w in bar_widths[:-1]:
            lefts.append(lefts[-1] + w)

        for w, left, col, lbl in zip(bar_widths, lefts, bar_colors_all, bar_labels):
            ax_bar.barh(0, w, left=left, color=col, edgecolor="white", height=0.5, label=lbl)
            if w > 15:
                ax_bar.text(left + w / 2, 0, str(w), ha="center", va="center",
                            fontsize=8, fontweight="bold", color="white")

        ax_bar.set_xlim(0, n_total)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("Number of nuclei", fontsize=8)
        ax_bar.set_title(f"{rnd}  |  {n_changed}/{n_total} changed ({100*n_changed/n_total:.0f}%)",
                         fontsize=10)
        ax_bar.legend(fontsize=6, loc="upper right")

        # ── Bottom: Transition heatmap ────────────────────────────────────────
        ax_heat = axes[1, col_idx]

        df_changed = df_r[df_r["changed"]]
        mat = np.zeros((len(color_order), len(color_order)), dtype=int)
        for _, row in df_changed.iterrows():
            r_idx = color_order.index(row["color_before"])
            c_idx = color_order.index(row["color_after"])
            mat[r_idx, c_idx] += 1

        im = ax_heat.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0)
        ax_heat.set_xticks(range(len(color_order)))
        ax_heat.set_xticklabels(color_order, fontsize=9)
        ax_heat.set_yticks(range(len(color_order)))
        ax_heat.set_yticklabels(color_order, fontsize=9)
        ax_heat.set_xlabel("Call AFTER correction", fontsize=8)
        ax_heat.set_ylabel("Call BEFORE correction", fontsize=8)
        ax_heat.set_title(f"{rnd} — transition matrix\n(changed nuclei only, n={len(df_changed)})",
                          fontsize=9)

        for i in range(len(color_order)):
            for j in range(len(color_order)):
                if mat[i, j] > 0:
                    ax_heat.text(j, i, str(mat[i, j]), ha="center", va="center",
                                 fontsize=9, fontweight="bold",
                                 color="white" if mat[i, j] > mat.max() * 0.5 else "black")

        plt.colorbar(im, ax=ax_heat, fraction=0.04, label="n nuclei")

    plt.tight_layout()
    out = QC_DIR / "qc_bg_call_changes.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — SNR improvement: before vs after (per channel per round)
# ─────────────────────────────────────────────────────────────────────────────

def fig_snr_improvement(df_int: pd.DataFrame, df_calls: pd.DataFrame):
    """
    SNR = winner_channel / mean_of_other_two, computed for both raw and corrected.
    Higher SNR after correction = cleaner channel separation.
    Expected: SNR should increase after background removal, especially for Blue
    (which had a disproportionately high background pedestal).
    """
    logger.info("Generating SNR improvement figure...")

    def compute_snr(vals: dict) -> float:
        sorted_v = sorted(vals.values(), reverse=True)
        second   = sorted_v[1] if sorted_v[1] > 0 else 1.0
        return sorted_v[0] / second

    rows = []
    for _, row in df_int.iterrows():
        nid = int(row["nucleus_id"]); rnd = row["round"]
        raw_vals  = {ch: float(row[ch])          for ch in CHANNELS}
        corr_vals = {ch: float(row[f"{ch}_corr"]) for ch in CHANNELS}
        rows.append({
            "nucleus_id": nid, "round": rnd,
            "snr_raw":  compute_snr(raw_vals),
            "snr_corr": compute_snr(corr_vals),
        })
    df_snr = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "SNR Improvement After Background Subtraction\n"
        "SNR = max_channel / 2nd_max_channel  |  Higher = cleaner channel separation\n"
        "Orange line = median SNR; shaded = IQR",
        fontsize=11
    )

    for ax, rnd in zip(axes, ROUNDS):
        df_r = df_snr[df_snr["round"] == rnd]
        raw_snr  = df_r["snr_raw"].clip(0, 20).values
        corr_snr = df_r["snr_corr"].clip(0, 20).values

        vp = ax.violinplot([raw_snr, corr_snr], positions=[1, 2], showmedians=True)
        colors_v = ["#AAAAAA", "#E67E22"]
        for pc, c in zip(vp["bodies"], colors_v):
            pc.set_facecolor(c); pc.set_alpha(0.75)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)

        # Annotate medians
        for pos, vals, col in [(1, raw_snr, "gray"), (2, corr_snr, "darkorange")]:
            med = float(np.median(vals))
            ax.text(pos, med + 0.3, f"median={med:.1f}×", ha="center",
                    fontsize=8, color=col, fontweight="bold")

        ax.axhline(1.5, color="red",   lw=1, linestyle="--", alpha=0.7, label="SNR=1.5 (ambiguous)")
        ax.axhline(3.0, color="green", lw=1, linestyle="--", alpha=0.7, label="SNR=3.0 (confident)")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["RAW\n(before)", "Corrected\n(after)"], fontsize=9)
        ax.set_ylabel("SNR (max / 2nd-max)", fontsize=8)
        ax.set_title(f"{rnd}", fontsize=10)
        ax.legend(fontsize=7)
        ax.set_ylim(0, 22)

        # Improvement annotation
        med_raw  = float(np.median(raw_snr))
        med_corr = float(np.median(corr_snr))
        pct_imp  = (med_corr - med_raw) / (med_raw + 1e-8) * 100
        ax.text(1.5, 19, f"Δmedian = {pct_imp:+.0f}%", ha="center",
                fontsize=9, color="darkorange" if pct_imp > 0 else "blue",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkorange", alpha=0.9))

    plt.tight_layout()
    out = QC_DIR / "qc_bg_snr_improvement.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — SNR per decoded color per round (3 rounds × 3 colors)
# ─────────────────────────────────────────────────────────────────────────────

def fig_snr_by_color(df_int: pd.DataFrame, df_calls: pd.DataFrame):
    """
    SNR broken down by decoded color (Purple/Blue/Yellow) per round.
    Layout: 3 rows (rounds: Hyb2/Hyb3/Hyb4) × 3 columns (colors: Purple/Blue/Yellow).

    SNR definition (per color):
      SNR = signal_channel / max(other two channels), clipped at 1 to avoid division by zero.
      Signal channel is the one encoding that color:
        Purple → Ch1_AF647,  Blue → Ch2_AF590,  Yellow → Ch3_AF488.

    Both RAW (before) and corrected (after) SNR are shown side-by-side for the
    same set of nuclei — those decoded as that color AFTER background subtraction.
    This isolates the per-color improvement: e.g., Blue cells specifically had
    their high background pedestal removed, which should show the most dramatic SNR gain.
    """
    logger.info("Generating per-color SNR figure...")

    # color → (signal channel, [other two channels])
    COLOR_CH = {
        "Purple": ("Ch1_AF647", ["Ch2_AF590", "Ch3_AF488"]),
        "Blue":   ("Ch2_AF590", ["Ch1_AF647", "Ch3_AF488"]),
        "Yellow": ("Ch3_AF488", ["Ch1_AF647", "Ch2_AF590"]),
    }
    COLOR_ORDER = ["Purple", "Blue", "Yellow"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=False)
    fig.suptitle(
        "SNR per Decoded Color per Round — Before vs After Background Subtraction\n"
        "SNR = signal_channel / max(other two channels)\n"
        "Only nuclei decoded as that color (after correction) | Gray = RAW · Colored = Corrected",
        fontsize=11
    )

    for row_idx, rnd in enumerate(ROUNDS):
        df_r_int   = df_int[df_int["round"] == rnd]
        df_r_calls = df_calls[df_calls["round"] == rnd][["nucleus_id", "color_after"]]
        df_merged  = df_r_int.merge(df_r_calls, on="nucleus_id", how="left")
        df_merged["color_after"] = df_merged["color_after"].fillna("None")

        for col_idx, color in enumerate(COLOR_ORDER):
            ax = axes[row_idx, col_idx]
            sig_ch, other_chs = COLOR_CH[color]

            # Only nuclei called as this color after correction
            df_color = df_merged[df_merged["color_after"] == color].copy()

            if len(df_color) < 5:
                ax.text(0.5, 0.5, f"n={len(df_color)}\n(too few)",
                        ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_title(f"{rnd} — {color}", fontsize=9)
                ax.set_xticks([1, 2])
                ax.set_xticklabels(["RAW\n(before)", "Corrected\n(after)"], fontsize=8)
                continue

            # SNR before (raw intensities)
            other_raw_max = df_color[other_chs].max(axis=1).clip(lower=1.0)
            snr_raw = (df_color[sig_ch] / other_raw_max).clip(0, 30).values

            # SNR after (corrected intensities)
            other_corr = [f"{c}_corr" for c in other_chs]
            other_corr_max = df_color[other_corr].max(axis=1).clip(lower=1.0)
            snr_corr = (df_color[f"{sig_ch}_corr"] / other_corr_max).clip(0, 30).values

            vp = ax.violinplot([snr_raw, snr_corr], positions=[1, 2], showmedians=True)
            vp["bodies"][0].set_facecolor("#BBBBBB");          vp["bodies"][0].set_alpha(0.75)
            vp["bodies"][1].set_facecolor(PALETTE[color]);     vp["bodies"][1].set_alpha(0.8)
            vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)

            med_raw  = float(np.median(snr_raw))
            med_corr = float(np.median(snr_corr))
            pct_imp  = (med_corr - med_raw) / (med_raw + 1e-8) * 100

            # Median value labels
            y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 5 else 30
            ax.text(1, med_raw  + 0.5, f"{med_raw:.1f}×",  ha="center",
                    fontsize=7, color="gray",         fontweight="bold")
            ax.text(2, med_corr + 0.5, f"{med_corr:.1f}×", ha="center",
                    fontsize=7, color=PALETTE[color], fontweight="bold")

            # Reference lines
            ax.axhline(1.5, color="red",   lw=0.8, linestyle="--", alpha=0.5, label="SNR=1.5 (ambiguous)")
            ax.axhline(3.0, color="green", lw=0.8, linestyle="--", alpha=0.5, label="SNR=3.0 (confident)")

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["RAW\n(before)", "Corrected\n(after)"], fontsize=8)
            ax.set_ylim(0, 32)
            ax.set_title(f"{rnd} — {color}  (n={len(df_color)})", fontsize=9,
                         color=PALETTE[color])
            if col_idx == 0:
                ax.set_ylabel("SNR (signal / max other)", fontsize=8)

            # Delta improvement box
            ec_col = "darkorange" if pct_imp > 0 else "steelblue"
            ax.text(1.5, 27.5, f"Δ {pct_imp:+.0f}%", ha="center", fontsize=9,
                    color=ec_col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ec_col, alpha=0.9))

            # n label bottom right
            ax.text(0.97, 0.03, f"n={len(df_color)}", ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=7, color="gray")

    plt.tight_layout()
    out = QC_DIR / "qc_bg_snr_by_color.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    for required in [
        PROJECT_ROOT / "python_results/module5/spot_intensities.csv",
        PROJECT_ROOT / "python_results/module6/barcodes.csv",
    ]:
        if not required.exists():
            logger.error(f"Missing: {required}"); sys.exit(1)

    logger.info("Loading data...")
    df_int = pd.read_csv(PROJECT_ROOT / "python_results/module5/spot_intensities.csv")
    df_bc  = pd.read_csv(PROJECT_ROOT / "python_results/module6/barcodes.csv")

    # Verify corrected columns exist
    if "Ch1_AF647_corr" not in df_int.columns:
        logger.error("Corrected columns not found — re-run run_module5.py first"); sys.exit(1)

    # Derive "before" calls (raw, threshold=2000) inline — no re-run needed
    logger.info(f"Deriving 'before' calls (raw intensities, threshold={RAW_THRESH} ADU)...")
    df_before = derive_calls_raw(df_int)

    # Derive "after" calls (corrected, threshold=500) inline
    logger.info(f"Deriving 'after' calls (corrected, threshold={CORR_THRESH} ADU)...")
    df_after  = derive_calls_corr(df_int)

    # Merge into comparison table
    df_calls = df_before.merge(df_after, on=["nucleus_id", "round"])

    # ── Summary statistics ─────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("COMPARISON SUMMARY (Before vs After Background Subtraction)")
    logger.info("─" * 50)

    n_before_ok = (
        df_calls.groupby("nucleus_id")
        .apply(lambda g: "None" not in g["color_before"].values)
        .sum()
    )
    n_after_ok  = (
        df_calls.groupby("nucleus_id")
        .apply(lambda g: "None" not in g["color_after"].values)
        .sum()
    )
    n_total = df_calls["nucleus_id"].nunique()

    logger.info(f"  Fully decoded BEFORE: {n_before_ok}/{n_total} "
                f"({100*n_before_ok/n_total:.1f}%)  [raw, thresh={RAW_THRESH} ADU]")
    logger.info(f"  Fully decoded AFTER:  {n_after_ok}/{n_total} "
                f"({100*n_after_ok/n_total:.1f}%)  [corrected, thresh={CORR_THRESH} ADU]")
    logger.info("")

    for rnd in ROUNDS:
        df_r    = df_calls[df_calls["round"] == rnd]
        n_chg   = (df_r["color_before"] != df_r["color_after"]).sum()
        n_none_b = (df_r["color_before"] == "None").sum()
        n_none_a = (df_r["color_after"]  == "None").sum()
        logger.info(f"  {rnd}: {n_chg}/{len(df_r)} nuclei changed call "
                    f"| None: {n_none_b} → {n_none_a}")
        # Breakdown of changes
        changes = df_r[df_r["color_before"] != df_r["color_after"]]
        for (cb, ca), grp in changes.groupby(["color_before", "color_after"]):
            logger.info(f"    {cb:8s} → {ca:8s}: n={len(grp)}")

    # Background/signal ratio summary
    logger.info("")
    logger.info("  Background / Raw-signal ratios (mean per channel per round):")
    for rnd in ROUNDS:
        df_r = df_int[df_int["round"] == rnd]
        for ch in CHANNELS:
            bg  = df_r[f"{ch}_bg"].mean()
            raw = df_r[ch].mean()
            logger.info(f"    {rnd} {ch}: bg={bg:.0f} ADU, raw={raw:.0f} ADU, "
                        f"ratio={bg/raw:.1%}")

    logger.info("─" * 50)

    # ── Generate figures ───────────────────────────────────────────────────
    logger.info("Generating comparison figures...")
    fig_scatter_comparison(df_int, df_calls)
    fig_background_magnitude(df_int)
    fig_call_changes(df_calls)
    fig_snr_improvement(df_int, df_calls)
    fig_snr_by_color(df_int, df_calls)

    print("\n" + "=" * 60)
    print("BG COMPARISON COMPLETE")
    print("=" * 60)
    print(f"  Before: {n_before_ok}/{n_total} fully decoded ({100*n_before_ok/n_total:.1f}%)")
    print(f"  After:  {n_after_ok}/{n_total} fully decoded ({100*n_after_ok/n_total:.1f}%)")
    print()
    print("  Figures saved to python_results/qc/:")
    print("    qc_bg_scatter_comparison.png  — raw vs corrected channel scatter")
    print("    qc_bg_background_magnitude.png — background pedestal analysis")
    print("    qc_bg_call_changes.png        — which cells changed + transition matrix")
    print("    qc_bg_snr_improvement.png     — SNR before vs after (aggregated per round)")
    print("    qc_bg_snr_by_color.png        — SNR per color per round (3×3 grid)")
    print("=" * 60)


if __name__ == "__main__":
    main()
