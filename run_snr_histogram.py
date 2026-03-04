"""
Per-Channel Per-Round SNR Histogram Analysis
=============================================
Reads anchor_candidates.csv and plots 9 histograms (3 channels × 3 rounds)
of the peak-inside / mean-outside SNR at each Hyb4-detected punctum position.

Purpose:
  - Reveal whether the SNR distributions are bimodal (signal vs noise)
  - Compare Ch1/Ch2/Ch3 to decide: global threshold or per-channel?
  - Compare Hyb4 / Hyb3 / Hyb2 to see signal vs carry-over round behaviour

Outputs:
  python_results/puncta_anchor/snr_histograms.png   — 3×3 histogram grid
  python_results/puncta_anchor/snr_summary.csv       — median, p10, p90 per panel
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
OUT_DIR      = PROJECT_ROOT / "python_results/puncta_anchor"
CSV_PATH     = OUT_DIR / "anchor_candidates.csv"

# Column name mapping
CHANNELS = ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]
ROUNDS   = ["Hyb4", "Hyb3", "Hyb2"]
CH_SHORT = {"Ch1_AF647": "Ch1\n(Purple)", "Ch2_AF590": "Ch2\n(Blue)", "Ch3_AF488": "Ch3\n(Yellow)"}
CH_COLOR = {"Ch1_AF647": "#9B59B6",       "Ch2_AF590": "#3498DB",    "Ch3_AF488": "#F4D03F"}
COL_MAP  = {
    ("Hyb4", "Ch1_AF647"): "snr_ch1_h4",
    ("Hyb4", "Ch2_AF590"): "snr_ch2_h4",
    ("Hyb4", "Ch3_AF488"): "snr_ch3_h4",
    ("Hyb3", "Ch1_AF647"): "snr_ch1_h3",
    ("Hyb3", "Ch2_AF590"): "snr_ch2_h3",
    ("Hyb3", "Ch3_AF488"): "snr_ch3_h3",
    ("Hyb2", "Ch1_AF647"): "snr_ch1_h2",
    ("Hyb2", "Ch2_AF590"): "snr_ch2_h2",
    ("Hyb2", "Ch3_AF488"): "snr_ch3_h2",
}

def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found — run run_puncta_anchor.py first.")
        sys.exit(1)

    df = pd.read_csv(str(CSV_PATH))
    print(f"Loaded {len(df)} candidates from {CSV_PATH.name}")

    # Check required columns
    missing = [c for c in COL_MAP.values() if c not in df.columns]
    if missing:
        print(f"ERROR: missing SNR columns: {missing}")
        print("Re-run run_puncta_anchor.py (per-channel SNR was added in this version).")
        sys.exit(1)

    # ── 3×3 histogram grid ────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=False)
    summary_rows = []

    # Clip upper tail for display (SNR can be very large for very bright spots
    # against near-zero background — cap at 99th percentile for readability)
    global_p99 = np.nanpercentile(
        [df[c].values for c in COL_MAP.values()], 99
    )
    x_max = min(global_p99 * 1.1, 50)   # never exceed 50 for display

    for r_idx, rnd in enumerate(ROUNDS):
        for c_idx, ch in enumerate(CHANNELS):
            ax  = axes[r_idx, c_idx]
            col = COL_MAP[(rnd, ch)]
            vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            vals = vals.clip(upper=x_max)

            ax.hist(vals, bins=60, range=(0, x_max),
                    color=CH_COLOR[ch], alpha=0.75, edgecolor="none")

            # Annotate with key percentiles
            p10  = float(np.percentile(vals, 10))
            med  = float(np.percentile(vals, 50))
            p90  = float(np.percentile(vals, 90))
            ax.axvline(med, color="black",  lw=1.5, linestyle="--", label=f"median={med:.1f}")
            ax.axvline(p10, color="#888888", lw=1.0, linestyle=":",  label=f"p10={p10:.1f}")
            ax.axvline(p90, color="#444444", lw=1.0, linestyle=":",  label=f"p90={p90:.1f}")

            ax.set_title(f"{rnd} / {CH_SHORT[ch]}", fontsize=9)
            ax.set_xlabel("SNR (peak / local bg)", fontsize=7)
            ax.set_ylabel("# candidates", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="upper right")

            summary_rows.append(dict(
                round=rnd, channel=ch, col=col,
                n=len(vals), median=round(med, 2),
                p10=round(p10, 2), p90=round(p90, 2),
                pct_above_3=round(100 * (vals >= 3.0).mean(), 1),
                pct_above_5=round(100 * (vals >= 5.0).mean(), 1),
            ))

    fig.suptitle(
        "Per-Channel Per-Round SNR Distributions\n"
        "(peak-inside / mean-outside, at Hyb4-detected punctum positions)\n"
        f"n_candidates={len(df)}  |  log_threshold=0.20  |  SNR filter=disabled",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = OUT_DIR / "snr_histograms.png"
    plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Histogram saved → {out_png}")

    # ── Summary table ─────────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    out_csv = OUT_DIR / "snr_summary.csv"
    df_summary.to_csv(str(out_csv), index=False)

    print("\n=== SNR Summary (median | p10 | p90 | %≥3 | %≥5) ===")
    print(f"{'Round':<6} {'Channel':<12} {'median':>7} {'p10':>7} {'p90':>7} {'%≥3':>6} {'%≥5':>6}")
    for _, row in df_summary.iterrows():
        print(f"{row['round']:<6} {row['channel']:<12} "
              f"{row['median']:>7.2f} {row['p10']:>7.2f} {row['p90']:>7.2f} "
              f"{row['pct_above_3']:>5.1f}% {row['pct_above_5']:>5.1f}%")

    print(f"\nSaved → {out_csv}")
    print("\nInterpretation guide:")
    print("  - Bimodal histogram → clear signal/noise separation → good threshold exists")
    print("  - Unimodal → no clear separation → SNR alone insufficient, keep LoG threshold")
    print("  - If Ch2_AF590 median << Ch1/Ch3 → per-channel thresholds needed")
    print("  - 'Signal' peak typically at SNR > 3–5; 'noise' peak at SNR 1–2")


if __name__ == "__main__":
    main()
