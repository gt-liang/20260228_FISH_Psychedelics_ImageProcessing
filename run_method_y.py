"""
Run Method Y Spot Calling + Comparison vs Method X
===================================================
Usage:
    python run_method_y.py

Runs Ronan-style per-nucleus adaptive threshold + puncta area calling
(Method Y) and generates a side-by-side comparison figure against
Method X (max pixel intensity from Module 5).

Scientific Purpose:
    Method X and Method Y both aim to identify which channel carries the
    real smFISH spot per nucleus per round. Comparing them gives us:

    1. Argmax agreement rate — if X and Y both point to the same "winner"
       channel, the call is highly robust regardless of technique.
    2. Discordant nuclei — cases where X ≠ Y deserve manual inspection:
       they may have very low SNR, carry-over signal, or edge registration
       artefacts that push one method to call a different channel.
    3. Puncta area distribution — Method Y area violin plots let us check
       whether puncta sizes are biologically plausible (expected: 5–50 px
       for a single diffraction-limited spot at 20× magnification).

Prerequisites:
    - Module 5 (python_results/module5/spot_intensities.csv)
    - Module 4 (python_results/module4/nucleus_labels.npy)
    - Module 1 (python_results/module1/crop_coords.json)
    - Module 3 (python_results/module3/registration_hyb{2,3}_to_hyb4.json)

Outputs:
    python_results/module5/spot_intensities_methodY.csv
    python_results/qc/qc_methodX_vs_Y.png
    python_results/module5/method_y_calling.log
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.module5_spot_calling.method_y_caller import MethodYCaller

# Logging
log_dir  = PROJECT_ROOT / "python_results" / "module5"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "method_y_calling.log"

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(str(log_path), level="DEBUG",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
           rotation="10 MB")

CHANNELS = ["Ch1_AF647", "Ch2_AF590", "Ch3_AF488"]
ROUNDS   = ["Hyb2", "Hyb3", "Hyb4"]
PALETTE  = {
    "Purple": "#9B59B6",
    "Blue":   "#3498DB",
    "Yellow": "#F4D03F",
}
ROUND_COLORS = {"Hyb2": "#E74C3C", "Hyb3": "#2ECC71", "Hyb4": "#3498DB"}
CH_LABELS    = {ch: ch.split("_")[1] for ch in CHANNELS}


# ─────────────────────────────────────────────────────────────────────────────
# Comparison figure
# ─────────────────────────────────────────────────────────────────────────────

def fig_comparison(df_x: pd.DataFrame, df_y: pd.DataFrame, out_dir: Path):
    """
    3-row × 3-column figure comparing Method X and Method Y.

    Row 1 (per round): Scatter — Method X max intensity vs Method Y puncta area.
      - Each point = one nucleus. Color = argmax channel from Method X.
      - Title shows agreement rate for that round.

    Row 2 (per round): Bar chart — agreement rate broken down by X-argmax channel.
      - High agreement (>80%) = both methods are consistent → robust call.
      - Low agreement = channel borderline → manual inspection warranted.

    Row 3 (per round): Violin — Method Y puncta area distribution for positive nuclei.
      - Checks that puncta sizes are biologically reasonable (expected ~5–50 px²
        for a single diffraction-limited spot at 20× magnification).
      - Very large areas (>200 px²) suggest aggregates or debris.
    """
    logger.info("Generating Method X vs Y comparison figure...")

    # Merge X and Y on nucleus_id × round
    df_xr = df_x.rename(columns={ch: f"X_{ch}" for ch in CHANNELS})
    df_yr = df_y.rename(columns={ch: f"Y_{ch}" for ch in CHANNELS})
    df_m  = df_xr.merge(df_yr, on=["nucleus_id", "round"], how="inner")

    # Argmax per method
    x_cols = [f"X_{ch}" for ch in CHANNELS]
    y_cols = [f"Y_{ch}" for ch in CHANNELS]
    df_m["argmax_X"] = df_m[x_cols].idxmax(axis=1).str.replace("X_", "", regex=False)
    df_m["argmax_Y"] = df_m[y_cols].idxmax(axis=1).str.replace("Y_", "", regex=False)
    df_m["agree"]    = df_m["argmax_X"] == df_m["argmax_Y"]

    ch_colors = {
        "Ch1_AF647": PALETTE["Purple"],
        "Ch2_AF590": PALETTE["Blue"],
        "Ch3_AF488": PALETTE["Yellow"],
    }

    fig = plt.figure(figsize=(21, 15))
    fig.suptitle(
        "Method X (max intensity) vs Method Y (Ronan: adaptive threshold + puncta area)\n"
        "Cross-comparison of per-nucleus channel argmax calls  |  "
        "Agreement = both methods name the same winner channel",
        fontsize=12,
    )

    # ── Row 1: Scatter per round ──────────────────────────────────────────
    for col_idx, rnd in enumerate(ROUNDS):
        ax = fig.add_subplot(3, 3, col_idx + 1)
        df_r    = df_m[df_m["round"] == rnd]
        agree_r = df_r["agree"].mean()

        for ch in CHANNELS:
            grp = df_r[df_r["argmax_X"] == ch]
            ax.scatter(
                grp[f"X_{ch}"], grp[f"Y_{ch}"],
                c=ch_colors[ch], s=8, alpha=0.55,
                label=f"X-argmax={CH_LABELS[ch]} (n={len(grp)})",
            )

        ax.set_xlabel("Method X: max intensity (ADU)", fontsize=8)
        ax.set_ylabel("Method Y: puncta area (px²)", fontsize=8)
        ax.set_title(f"{rnd}  |  argmax agreement = {agree_r:.1%}", fontsize=9)
        ax.legend(fontsize=6, markerscale=2)

    # ── Row 2: Agreement by channel per round ────────────────────────────
    for col_idx, rnd in enumerate(ROUNDS):
        ax   = fig.add_subplot(3, 3, col_idx + 4)
        df_r = df_m[df_m["round"] == rnd]

        ch_order = [ch for ch in CHANNELS if ch in df_r["argmax_X"].values]
        agree_vals = [
            df_r.loc[df_r["argmax_X"] == ch, "agree"].mean()
            if (df_r["argmax_X"] == ch).any() else 0.0
            for ch in CHANNELS
        ]
        bar_colors = [ch_colors[ch] for ch in CHANNELS]

        bars = ax.bar(
            [CH_LABELS[ch] for ch in CHANNELS], agree_vals,
            color=bar_colors, alpha=0.85, edgecolor="white",
        )
        ax.set_ylim(0, 1.1)
        ax.axhline(0.80, color="green", lw=1, linestyle="--", alpha=0.7, label="80%")
        ax.set_ylabel("Agreement rate (X == Y argmax)", fontsize=7)
        ax.set_title(f"{rnd} — agreement by X-argmax channel", fontsize=9)
        ax.legend(fontsize=7)

        for bar, v in zip(bars, agree_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(v + 0.03, 1.05),
                f"{v:.0%}", ha="center", fontsize=8, fontweight="bold",
            )

    # ── Row 3: Method Y area distribution (positive nuclei) ──────────────
    for col_idx, rnd in enumerate(ROUNDS):
        ax   = fig.add_subplot(3, 3, col_idx + 7)
        df_r = df_m[df_m["round"] == rnd]

        data      = []
        tick_lbls = []
        for ch in CHANNELS:
            col  = f"Y_{ch}"
            vals = df_r.loc[df_r[col] > 0, col].values
            if len(vals) > 0:
                data.append(vals)
                tick_lbls.append(f"{CH_LABELS[ch]}\n(n={len(vals)})")

        if data:
            parts = ax.violinplot(data, positions=range(len(data)), showmedians=True)
            for pc, ch in zip(parts["bodies"], CHANNELS[:len(data)]):
                pc.set_facecolor(ch_colors[ch])
                pc.set_alpha(0.7)
            ax.set_xticks(range(len(tick_lbls)))
            ax.set_xticklabels(tick_lbls, fontsize=8)
            ax.set_ylabel("Method Y puncta area (px²) — positive only", fontsize=7)
            # Reference lines for biologically expected punctum sizes
            ax.axhline(5,   color="blue",  lw=0.8, linestyle=":", alpha=0.6, label="5 px²  (min)")
            ax.axhline(200, color="red",   lw=0.8, linestyle=":", alpha=0.6, label="200 px² (aggregate?)")
            ax.legend(fontsize=6)
        else:
            ax.text(0.5, 0.5, "No positive nuclei", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

        ax.set_title(f"{rnd} — Method Y puncta area distribution", fontsize=9)

    plt.tight_layout()
    out = out_dir / "qc_methodX_vs_Y.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Comparison figure saved → {out.name}")

    return df_m


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    for required in [
        PROJECT_ROOT / "python_results/module5/spot_intensities.csv",
        PROJECT_ROOT / "python_results/module4/nucleus_labels.npy",
        PROJECT_ROOT / "python_results/module1/crop_coords.json",
    ]:
        if not required.exists():
            logger.error(f"Prerequisite missing: {required}")
            sys.exit(1)

    config_path = PROJECT_ROOT / "config" / "module5_spot_calling.yaml"
    caller      = MethodYCaller(config_path=str(config_path), project_root=str(PROJECT_ROOT))
    df_y        = caller.run()

    # Load Method X for comparison
    df_x = pd.read_csv(PROJECT_ROOT / "python_results/module5/spot_intensities.csv")

    # Comparison figure + merged table
    qc_dir = PROJECT_ROOT / "python_results" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    df_merged = fig_comparison(df_x, df_y, qc_dir)

    # Summary report
    overall_agree = df_merged["agree"].mean()

    print("\n" + "=" * 58)
    print("METHOD Y COMPLETE")
    print("=" * 58)
    print(f"  Method Y CSV  : python_results/module5/spot_intensities_methodY.csv")
    print(f"  Comparison fig: python_results/qc/qc_methodX_vs_Y.png")
    print()
    print(f"  Overall X–Y argmax agreement  : {overall_agree:.1%}")
    print()
    for rnd in ROUNDS:
        r_agree = df_merged.loc[df_merged["round"] == rnd, "agree"].mean()
        n_rnd   = (df_merged["round"] == rnd).sum()
        print(f"    {rnd} (n={n_rnd}): agreement = {r_agree:.1%}")
    print()

    # Discordant nuclei summary
    df_discord = df_merged[~df_merged["agree"]]
    print(f"  Discordant nuclei (X ≠ Y): {len(df_discord)} rows")
    print(f"  These are candidates for manual inspection.")
    print(f"  Cross-reference with: python_results/qc/dual_high_nucleus_ids.csv")
    print("=" * 58)


if __name__ == "__main__":
    main()
