"""
Module 5.5 — mCherry Cross-Round Correction
============================================
Scientific rationale:
    mCherry fluorescent protein was expressed in the cells and could not be
    fully eliminated before FISH imaging. Its emission overlaps with Ch2_AF590
    (Blue channel). Because mCherry is a STABLE protein, its contribution to
    Ch2_AF590 is present in ALL hybridization rounds — not just Hyb3 where the
    genuine Blue FISH probe is expected.

    Imaging order: Hyb4 → Hyb3 → Hyb2
    Blue FISH probe: only present in Hyb3

    Therefore:
      - Ch2_AF590 in Hyb4 = mCherry signal (imaged first)
      - Ch2_AF590 in Hyb3 = Blue FISH punctum + mCherry signal (imaged second)
      - Ch2_AF590 in Hyb2 = mCherry signal (imaged last, may be slightly bleached)

Correction formula (per nucleus i):
    mCherry_baseline_i = mean(Ch2_AF590_corr_Hyb4_i, Ch2_AF590_corr_Hyb2_i)

    Because Hyb3 was imaged BETWEEN Hyb4 and Hyb2 with equal intervals,
    mean(Hyb4, Hyb2) is the exact linear interpolation of mCherry at Hyb3.
    This accounts for per-nucleus variability in mCherry expression level.

    Ch2_AF590_mc_rnd_i = max(Ch2_AF590_corr_rnd_i − mCherry_baseline_i, 0)

    For Hyb4 and Hyb2: Ch2_mc ≈ 0 → Blue calls automatically disappear.
    For Hyb3: only FISH signal above the mCherry baseline survives.

Inputs:
    python_results/module5/spot_intensities.csv  (with *_corr columns from M5)

Outputs:
    python_results/module5/spot_intensities_mc.csv  — adds Ch2_AF590_mc + mcherry_baseline
    python_results/module6/barcodes_mc.csv          — re-decoded barcodes
    python_results/qc/qc_mc_*.png                  — QC comparison figures

Usage:
    python run_mcherry_correction.py
"""

import sys
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

QC_DIR  = PROJECT_ROOT / "python_results" / "qc"
OUT_DIR = PROJECT_ROOT / "python_results" / "module5"
BC_DIR  = PROJECT_ROOT / "python_results" / "module6"
QC_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

# ── Constants ──────────────────────────────────────────────────────────────────
CHANNELS   = ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]
COLOR_MAP  = {"Ch1_AF647": "Purple", "Ch2_AF590": "Blue", "Ch3_AF488": "Yellow"}
ROUNDS     = ["Hyb2", "Hyb3", "Hyb4"]
CORR_THRESH = 500   # ADU — same threshold as Module 6 corrected decoding
PALETTE = {
    "Purple": "#9B59B6",
    "Blue":   "#3498DB",
    "Yellow": "#F4D03F",
    "None":   "#95A5A6",
}


# ── Decoder ────────────────────────────────────────────────────────────────────
def decode_row(ch1: float, ch2: float, ch3: float, thresh: float = CORR_THRESH) -> str:
    """Argmax decoder with minimum signal threshold."""
    vals = {"Ch1_AF647": ch1, "Ch2_AF590": ch2, "Ch3_AF488": ch3}
    mx = max(vals.values())
    if mx < thresh:
        return "None"
    return COLOR_MAP[max(vals, key=vals.get)]


# ── Figure 1: mCherry baseline characterization ────────────────────────────────
def fig_mcherry_baseline(df_wide: pd.DataFrame, df_long: pd.DataFrame):
    """
    Panel A: Violin of Ch2_AF590_corr per round (Hyb4 / Hyb3 / Hyb2) to visualize
             mCherry bleaching across the imaging sequence.
    Panel B: Distribution of per-nucleus mCherry baseline (mean of Hyb4 + Hyb2).
             Wide baseline spread = high cell-to-cell mCherry variability → confirms
             that a per-nucleus (not global) correction is essential.
    Panel C: Scatter of Hyb4 vs Hyb2 Ch2_corr per nucleus.
             Slope < 1 → bleaching occurred. Points near the diagonal → stable mCherry.
    """
    logger.info("Generating mCherry baseline characterization figure...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "mCherry Characterization — Ch2_AF590_corr Across Rounds\n"
        "Imaging order: Hyb4 → Hyb3 → Hyb2  |  Blue FISH probe expected only in Hyb3",
        fontsize=11
    )

    # Panel A: violin per round
    ax = axes[0]
    round_order = ["Hyb4", "Hyb3", "Hyb2"]   # imaging order
    round_colors = {"Hyb4": "#E74C3C", "Hyb3": "#3498DB", "Hyb2": "#E67E22"}
    data = [df_long[df_long["round"] == r]["Ch2_AF590_corr"].clip(0, 30000).values
            for r in round_order]
    vp = ax.violinplot(data, positions=[1, 2, 3], showmedians=True)
    for pc, r in zip(vp["bodies"], round_order):
        pc.set_facecolor(round_colors[r]); pc.set_alpha(0.7)
    vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
    for pos, r in zip([1, 2, 3], round_order):
        med = float(np.median(df_long[df_long["round"] == r]["Ch2_AF590_corr"]))
        ax.text(pos, med + 200, f"{med:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Hyb4\n(1st imaged)", "Hyb3\n(2nd imaged)", "Hyb2\n(3rd imaged)"], fontsize=9)
    ax.set_ylabel("Ch2_AF590_corr (ADU)", fontsize=9)
    ax.set_title("Ch2 corrected intensity per round\n(all nuclei)", fontsize=9)
    ax.set_ylim(0, None)

    # Panel B: mCherry baseline distribution
    ax = axes[1]
    baseline = df_wide["mcherry_baseline"].clip(0, 20000).values
    ax.hist(baseline, bins=50, color="#3498DB", alpha=0.75, edgecolor="white")
    ax.axvline(float(np.median(baseline)), color="black", lw=2, linestyle="--",
               label=f"median={np.median(baseline):.0f} ADU")
    ax.axvline(float(np.percentile(baseline, 25)), color="gray", lw=1, linestyle=":",
               label=f"p25={np.percentile(baseline,25):.0f} ADU")
    ax.axvline(float(np.percentile(baseline, 75)), color="gray", lw=1, linestyle=":",
               label=f"p75={np.percentile(baseline,75):.0f} ADU")
    ax.set_xlabel("Per-nucleus mCherry baseline (ADU)\n= mean(Ch2_corr_Hyb4, Ch2_corr_Hyb2)", fontsize=8)
    ax.set_ylabel("Number of nuclei", fontsize=9)
    ax.set_title("mCherry baseline distribution\n(wide spread → per-nucleus correction essential)", fontsize=9)
    ax.legend(fontsize=8)

    # Panel C: Hyb4 vs Hyb2 scatter (bleaching assessment)
    ax = axes[2]
    hyb4_ch2 = df_wide["Ch2_corr_Hyb4"].clip(0, 20000)
    hyb2_ch2 = df_wide["Ch2_corr_Hyb2"].clip(0, 20000)
    ax.scatter(hyb4_ch2, hyb2_ch2, s=4, alpha=0.4, color="#7F8C8D")
    # y = x line (no bleaching)
    lim = max(hyb4_ch2.max(), hyb2_ch2.max())
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.4, label="y = x (no bleach)")
    # Linear fit
    mask = (hyb4_ch2 > 100) & (hyb2_ch2 > 100)
    if mask.sum() > 10:
        slope = float(np.polyfit(hyb4_ch2[mask], hyb2_ch2[mask], 1)[0])
        ax.plot([0, lim], [0, slope * lim], "r-", lw=1.5, alpha=0.7,
                label=f"fit slope={slope:.2f} (1.0=no bleach)")
    ax.set_xlabel("Ch2_corr Hyb4 (1st imaged, ADU)", fontsize=8)
    ax.set_ylabel("Ch2_corr Hyb2 (3rd imaged, ADU)", fontsize=8)
    ax.set_title("Hyb4 vs Hyb2 Ch2 per nucleus\nSlope < 1 → mCherry bleaching", fontsize=9)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = QC_DIR / "qc_mc_baseline.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ── Figure 2: Ch2 before vs after mCherry correction per round ────────────────
def fig_ch2_correction(df_mc: pd.DataFrame):
    """
    For each round: violin of Ch2_AF590_corr (before) vs Ch2_AF590_mc (after mCherry removal).
    Expected:
      Hyb4 / Hyb2: Ch2_mc should collapse to ~0 (mCherry fully subtracted)
      Hyb3: Ch2_mc should show a bimodal distribution — zeros (no Blue FISH) + signal (Blue cells)
    """
    logger.info("Generating Ch2 before/after correction figure...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Ch2_AF590: Before vs After mCherry Cross-Round Subtraction\n"
        "Gray = corrected (spatial bg removed)  |  Blue = mCherry-corrected (mCherry also removed)\n"
        "Hyb2/Hyb4 should collapse to ~0; Hyb3 should show residual FISH signal",
        fontsize=10
    )

    for ax, rnd in zip(axes, ROUNDS):
        df_r = df_mc[df_mc["round"] == rnd]
        before = df_r["Ch2_AF590_corr"].clip(0, 20000).values
        after  = df_r["Ch2_AF590_mc"].clip(0, 20000).values

        vp = ax.violinplot([before, after], positions=[1, 2], showmedians=True)
        vp["bodies"][0].set_facecolor("#AAAAAA"); vp["bodies"][0].set_alpha(0.75)
        vp["bodies"][1].set_facecolor(PALETTE["Blue"]); vp["bodies"][1].set_alpha(0.8)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)

        for pos, vals, col in [(1, before, "gray"), (2, after, PALETTE["Blue"])]:
            med = float(np.median(vals))
            ax.text(pos, med + 150, f"{med:.0f}", ha="center", fontsize=8,
                    color=col, fontweight="bold")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Corr (bg only)", "MC-corrected\n(bg + mCherry)"], fontsize=8)
        ax.set_ylabel("Ch2_AF590 intensity (ADU)", fontsize=8)
        ax.set_title(f"{rnd}", fontsize=10)

        # Annotate % of nuclei with MC-corrected > 0
        n_pos = (after > 0).sum()
        ax.text(0.97, 0.97, f"{n_pos}/{len(after)} nuclei\nCh2_mc > 0",
                ha="right", va="top", transform=ax.transAxes, fontsize=7, color="gray")

    plt.tight_layout()
    out = QC_DIR / "qc_mc_ch2_distribution.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ── Figure 3: Decoding call comparison ────────────────────────────────────────
def fig_call_comparison(df_calls: pd.DataFrame):
    """
    For each round: stacked bar showing call distribution
    Side-by-side: BEFORE mCherry correction vs AFTER.
    Key expectation:
      Hyb2/Hyb4: Blue should drop to near-zero after correction.
      Hyb3: Blue should remain for genuine FISH signal.
    """
    logger.info("Generating call comparison figure...")

    color_order = ["Purple", "Yellow", "Blue", "None"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Decoding Call Distribution — Before vs After mCherry Correction\n"
        "Top: before (spatial bg subtracted only) | Bottom: after (mCherry also subtracted)\n"
        "Blue should vanish from Hyb2 and Hyb4; genuine Blue FISH signal preserved in Hyb3",
        fontsize=10
    )

    for col_idx, rnd in enumerate(ROUNDS):
        for row_idx, col_key in enumerate(["color_before", "color_after"]):
            ax = axes[row_idx, col_idx]
            df_r = df_calls[df_calls["round"] == rnd]
            counts = df_r[col_key].value_counts()

            left = 0
            for color in color_order:
                n = int(counts.get(color, 0))
                if n == 0:
                    continue
                ax.barh(0, n, left=left, color=PALETTE[color], edgecolor="white",
                        height=0.5, label=f"{color} (n={n})")
                if n > 20:
                    ax.text(left + n / 2, 0, str(n), ha="center", va="center",
                            fontsize=9, fontweight="bold", color="white")
                left += n

            ax.set_xlim(0, len(df_r))
            ax.set_yticks([])
            ax.set_xlabel("Number of nuclei", fontsize=8)
            title_suffix = "before MC correction" if row_idx == 0 else "after MC correction"
            ax.set_title(f"{rnd} — {title_suffix}", fontsize=9)
            if col_idx == 0 and row_idx == 0:
                handles = [mpatches.Patch(color=PALETTE[c], label=c) for c in color_order]
                ax.legend(handles=handles, fontsize=7, loc="upper right")

    plt.tight_layout()
    out = QC_DIR / "qc_mc_call_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ── Figure 4: Three-way SNR comparison ────────────────────────────────────────
def fig_snr_three_way(df_mc: pd.DataFrame, df_calls: pd.DataFrame):
    """
    For Blue calls (after mCherry correction) in Hyb3:
    Compare SNR at three stages:
      1. Raw (no correction)
      2. Spatially bg-subtracted only (_corr)
      3. mCherry-corrected (_mc)

    SNR = Ch2 / max(Ch1, Ch3) for Blue cells.
    Expected: SNR should increase at each stage, most dramatically at stage 3.
    """
    logger.info("Generating three-way SNR comparison figure...")

    # Blue nuclei in Hyb3 after mCherry correction
    hyb3_mc = df_mc[df_mc["round"] == "Hyb3"].copy()
    hyb3_after = df_calls[(df_calls["round"] == "Hyb3") & (df_calls["color_after"] == "Blue")]
    blue_nids = set(hyb3_after["nucleus_id"])
    blue_cells = hyb3_mc[hyb3_mc["nucleus_id"].isin(blue_nids)]

    def snr_blue(ch2_col):
        ch2 = blue_cells[ch2_col]
        other = blue_cells[["Ch1_AF647_corr", "Ch3_AF488_corr"]].max(axis=1).clip(lower=1.0)
        return (ch2 / other).clip(0, 30).values

    snr_raw   = snr_blue("Ch2_AF590")
    snr_corr  = snr_blue("Ch2_AF590_corr")
    snr_mc    = snr_blue("Ch2_AF590_mc")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"Hyb3 Blue Cells (n={len(blue_cells)}) — SNR at Three Correction Stages\n"
        "SNR = Ch2_AF590 / max(Ch1, Ch3)  |  Only nuclei decoded Blue after mCherry correction",
        fontsize=10
    )

    vp = ax.violinplot([snr_raw, snr_corr, snr_mc], positions=[1, 2, 3], showmedians=True)
    colors_v = ["#DDDDDD", "#85C1E9", PALETTE["Blue"]]
    labels_v = ["RAW\n(no correction)", "Spatial bg\nsubtracted", "mCherry\ncorrected"]
    for pc, c in zip(vp["bodies"], colors_v):
        pc.set_facecolor(c); pc.set_alpha(0.8)
    vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)

    for pos, vals, lbl in zip([1, 2, 3], [snr_raw, snr_corr, snr_mc], labels_v):
        med = float(np.median(vals))
        ax.text(pos, med + 0.5, f"{med:.1f}×", ha="center", fontsize=9,
                fontweight="bold", color="black")

    ax.axhline(1.5, color="red",   lw=1, linestyle="--", alpha=0.6, label="SNR=1.5 (ambiguous)")
    ax.axhline(3.0, color="green", lw=1, linestyle="--", alpha=0.6, label="SNR=3.0 (confident)")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels_v, fontsize=9)
    ax.set_ylabel("SNR (Ch2_Blue / max other channel)", fontsize=9)
    ax.set_ylim(0, 32)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = QC_DIR / "qc_mc_snr_three_way.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    int_path = PROJECT_ROOT / "python_results/module5/spot_intensities.csv"
    if not int_path.exists():
        logger.error(f"Missing: {int_path}"); sys.exit(1)

    logger.info("Loading spot intensities...")
    df_long = pd.read_csv(int_path)

    if "Ch2_AF590_corr" not in df_long.columns:
        logger.error("Ch2_AF590_corr not found — re-run run_module5.py first"); sys.exit(1)

    # ── Step 1: Pivot to wide (one row per nucleus, one col per round) ─────────
    logger.info("Step 1: Building per-nucleus wide table...")
    dfs = {}
    for rnd in ROUNDS:
        df_r = df_long[df_long["round"] == rnd][
            ["nucleus_id", "Ch2_AF590_corr"]
        ].rename(columns={"Ch2_AF590_corr": f"Ch2_corr_{rnd}"})
        dfs[rnd] = df_r.set_index("nucleus_id")

    df_wide = pd.concat(dfs.values(), axis=1).reset_index()
    df_wide.columns.name = None
    n_complete = df_wide[["Ch2_corr_Hyb4", "Ch2_corr_Hyb2"]].notna().all(axis=1).sum()
    logger.info(f"  {len(df_wide)} nuclei total, {n_complete} with complete Hyb4+Hyb2 data")

    # ── Step 2: Per-nucleus mCherry baseline ────────────────────────────────────
    # Imaging order: Hyb4 → Hyb3 → Hyb2
    # mean(Hyb4, Hyb2) = exact linear interpolation at Hyb3 timepoint
    logger.info("Step 2: Computing per-nucleus mCherry baseline = mean(Ch2_Hyb4, Ch2_Hyb2)...")
    df_wide["mcherry_baseline"] = (
        df_wide["Ch2_corr_Hyb4"].fillna(0) + df_wide["Ch2_corr_Hyb2"].fillna(0)
    ) / 2

    logger.info(f"  mCherry baseline — median: {df_wide['mcherry_baseline'].median():.0f} ADU  "
                f"| p25: {df_wide['mcherry_baseline'].quantile(0.25):.0f}  "
                f"| p75: {df_wide['mcherry_baseline'].quantile(0.75):.0f}  "
                f"| max: {df_wide['mcherry_baseline'].max():.0f}")

    # Bleaching check: Hyb4 vs Hyb2 median ratio
    med_hyb4 = df_wide["Ch2_corr_Hyb4"].median()
    med_hyb2 = df_wide["Ch2_corr_Hyb2"].median()
    bleach_pct = (1 - med_hyb2 / med_hyb4) * 100 if med_hyb4 > 0 else 0
    logger.info(f"  Bleaching check: Hyb4 median={med_hyb4:.0f} ADU → Hyb2 median={med_hyb2:.0f} ADU "
                f"→ {bleach_pct:.1f}% reduction")

    # ── Step 3: Apply mCherry subtraction to all rounds ────────────────────────
    logger.info("Step 3: Applying per-nucleus mCherry subtraction to all rounds...")
    df_mc = df_long.merge(
        df_wide[["nucleus_id", "mcherry_baseline"]], on="nucleus_id", how="left"
    )
    df_mc["Ch2_AF590_mc"] = (df_mc["Ch2_AF590_corr"] - df_mc["mcherry_baseline"]).clip(lower=0)

    # Save enriched intensity table
    mc_cols = list(df_long.columns) + ["mcherry_baseline", "Ch2_AF590_mc"]
    out_int = OUT_DIR / "spot_intensities_mc.csv"
    df_mc[mc_cols].to_csv(str(out_int), index=False)
    logger.info(f"  Intensities with mCherry correction saved → {out_int.name}")

    # ── Step 4: Re-decode with mCherry-corrected Blue ──────────────────────────
    logger.info("Step 4: Re-decoding with mCherry-corrected Blue channel...")
    rows_before, rows_after = [], []
    for _, row in df_mc.iterrows():
        nid = int(row["nucleus_id"]); rnd = row["round"]
        # Before: spatial bg only
        cb = decode_row(row["Ch1_AF647_corr"], row["Ch2_AF590_corr"], row["Ch3_AF488_corr"])
        # After: mCherry also subtracted from Blue
        ca = decode_row(row["Ch1_AF647_corr"], row["Ch2_AF590_mc"],   row["Ch3_AF488_corr"])
        rows_before.append({"nucleus_id": nid, "round": rnd, "color_before": cb})
        rows_after.append( {"nucleus_id": nid, "round": rnd, "color_after":  ca})

    df_calls = pd.merge(pd.DataFrame(rows_before), pd.DataFrame(rows_after),
                        on=["nucleus_id", "round"])

    # Save barcodes (one row per nucleus with 3-round color sequence)
    df_wide_bc = df_calls.pivot(index="nucleus_id", columns="round", values="color_after").reset_index()
    df_wide_bc.columns.name = None
    if all(r in df_wide_bc.columns for r in ROUNDS):
        df_wide_bc["barcode"] = (
            df_wide_bc["Hyb2"].str[0].fillna("?") +
            df_wide_bc["Hyb3"].str[0].fillna("?") +
            df_wide_bc["Hyb4"].str[0].fillna("?")
        )
    bc_path = BC_DIR / "barcodes_mc.csv"
    df_wide_bc.to_csv(str(bc_path), index=False)
    logger.info(f"  Barcodes saved → {bc_path.name}")

    # ── Step 5: Summary statistics ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("mCHERRY CORRECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Imaging order: Hyb4 (1st) → Hyb3 (2nd) → Hyb2 (3rd)")
    logger.info(f"  mCherry bleaching across imaging: {bleach_pct:.1f}% reduction Hyb4→Hyb2")
    logger.info(f"  Baseline formula: mean(Ch2_corr_Hyb4, Ch2_corr_Hyb2) per nucleus")
    logger.info("")

    for rnd in ROUNDS:
        df_r = df_calls[df_calls["round"] == rnd]
        cb_counts = df_r["color_before"].value_counts()
        ca_counts = df_r["color_after"].value_counts()
        n_changed = (df_r["color_before"] != df_r["color_after"]).sum()
        logger.info(f"  {rnd}: {n_changed}/{len(df_r)} calls changed")
        logger.info(f"    Blue BEFORE: {cb_counts.get('Blue', 0)} → AFTER: {ca_counts.get('Blue', 0)}")
        for color in ["Purple", "Yellow", "None"]:
            b = cb_counts.get(color, 0); a = ca_counts.get(color, 0)
            if b != a:
                logger.info(f"    {color}: {b} → {a}")

    n_before_ok = (
        df_calls.groupby("nucleus_id")
        .apply(lambda g: "None" not in g["color_before"].values, include_groups=False)
        .sum()
    )
    n_after_ok = (
        df_calls.groupby("nucleus_id")
        .apply(lambda g: "None" not in g["color_after"].values, include_groups=False)
        .sum()
    )
    n_total = df_calls["nucleus_id"].nunique()
    logger.info("")
    logger.info(f"  Fully decoded BEFORE MC correction: {n_before_ok}/{n_total} ({100*n_before_ok/n_total:.1f}%)")
    logger.info(f"  Fully decoded AFTER  MC correction: {n_after_ok}/{n_total}  ({100*n_after_ok/n_total:.1f}%)")

    if "barcode" in df_wide_bc.columns:
        bc_counts = df_wide_bc["barcode"].value_counts()
        n_unique = len(bc_counts)
        logger.info(f"  Unique barcodes after MC correction: {n_unique}")
        logger.info(f"  Top 10 barcodes:")
        for bc, cnt in bc_counts.head(10).items():
            logger.info(f"    {bc}: n={cnt}")

    logger.info("=" * 60)

    # ── Step 6: QC figures ─────────────────────────────────────────────────────
    logger.info("Generating QC figures...")
    fig_mcherry_baseline(df_wide, df_long)
    fig_ch2_correction(df_mc)
    fig_call_comparison(df_calls)
    fig_snr_three_way(df_mc, df_calls)

    print("\n" + "=" * 60)
    print("mCHERRY CORRECTION COMPLETE")
    print("=" * 60)
    print(f"  mCherry bleaching Hyb4→Hyb2: {bleach_pct:.1f}%")
    print(f"  Fully decoded BEFORE: {n_before_ok}/{n_total} ({100*n_before_ok/n_total:.1f}%)")
    print(f"  Fully decoded AFTER:  {n_after_ok}/{n_total} ({100*n_after_ok/n_total:.1f}%)")
    print()
    print("  Output files:")
    print("    python_results/module5/spot_intensities_mc.csv")
    print("    python_results/module6/barcodes_mc.csv")
    print()
    print("  QC figures saved to python_results/qc/:")
    print("    qc_mc_baseline.png          — mCherry bleaching + baseline distribution")
    print("    qc_mc_ch2_distribution.png  — Ch2 before/after mCherry subtraction per round")
    print("    qc_mc_call_comparison.png   — decoding call counts before vs after")
    print("    qc_mc_snr_three_way.png     — SNR at raw / spatial-bg / mCherry-corrected stages")
    print("=" * 60)


if __name__ == "__main__":
    main()
