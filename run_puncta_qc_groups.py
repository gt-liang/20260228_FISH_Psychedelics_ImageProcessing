#!/usr/bin/env python3
"""
Puncta Anchor Group QC Analysis
================================
Reads existing anchor_summary.csv + anchor_candidates.csv (produced by
run_puncta_anchor.py) and splits nuclei into groups for targeted QC.

No image reloading required — pure CSV analysis.

Groups
------
  single_ok           n_candidates == 1 AND decoded_ok == True   (~79%)
  single_unconfirmed  n_candidates == 1 AND decoded_ok == False
  zero                n_candidates == 0   (LoG found no blobs above threshold)
  multi               n_candidates >= 2   (multiple blobs detected)

For the none_multi group (single_unconfirmed + zero + multi):
  Also shows puncta candidates that did NOT pass the normalized threshold
  (2.0× for Ch1/Ch3, 1.5× for Ch2) — so we can visualize what signals were
  present but not confirmed.

Outputs  (python_results/puncta_anchor/groups/)
-----------------------------------------------
  group_labels.csv              — nucleus_id + group classification
  single_ok_barcodes.csv        — barcode counts for single_ok group
  single_ok_nucleus_ids.csv     — per-barcode list of nucleus IDs (for manual auditing)
  none_multi_candidates.csv     — all candidates from none/multi nuclei (with norm values)
  fig_group_distribution.png    — group count bar chart
  fig_single_ok_barcodes.png    — barcode distribution for single_ok group
  fig_none_multi_signals.png    — Hyb3 vs Hyb2 normalized signal scatter + failed histogram
"""

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
ANCHOR_DIR   = PROJECT_ROOT / "python_results/puncta_anchor"
OUT_DIR      = ANCHOR_DIR / "groups"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV    = ANCHOR_DIR / "anchor_summary.csv"
CANDIDATES_CSV = ANCHOR_DIR / "anchor_candidates.csv"

# ── Thresholds (must match config/puncta_anchor.yaml) ─────────────────────
# Used only for drawing reference lines on plots — not re-filtering data.
THRESH_STRICT  = 2.0   # Ch1_AF647 and Ch3_AF488
THRESH_MCHERRY = 1.5   # Ch2_AF590 (mCherry diffuse nuclear background)

# ── Color constants ────────────────────────────────────────────────────────
COLOR_DISPLAY = {
    "Purple": "#9B59B6",
    "Blue":   "#3498DB",
    "Yellow": "#F4D03F",
    "None":   "#95A5A6",
}
GROUP_COLORS = {
    "single_ok":           "#2ECC71",   # green
    "single_unconfirmed":  "#F39C12",   # orange
    "zero":                "#E74C3C",   # red
    "multi":               "#9B59B6",   # purple
}


# ── Classification ─────────────────────────────────────────────────────────
def classify_nuclei(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'group' column to the anchor_summary DataFrame.

    Scientific logic
    ----------------
    Biological expectation: each HEK cell has exactly ONE mRNA punctum.
    The LoG detector can find 0, 1, or many local maxima per nucleus.

      single_ok          : 1 candidate, confirmed in Hyb3 + Hyb2 → ideal case
      single_unconfirmed : 1 candidate found, but it did not confirm in both rounds
                           → signal was below threshold; cell is "None" in decoding
      zero               : LoG found nothing → nucleus is dark or threshold too high
      multi              : LoG found ≥2 blobs → detection artifact (small/bright nuclei)
    """
    def _group(row):
        if row["n_candidates"] == 0:
            return "zero"
        elif row["n_candidates"] == 1 and row["decoded_ok"]:
            return "single_ok"
        elif row["n_candidates"] == 1 and not row["decoded_ok"]:
            return "single_unconfirmed"
        else:
            return "multi"

    df = df.copy()
    df["group"] = df.apply(_group, axis=1)
    return df


# ── Figure 1: Group distribution ───────────────────────────────────────────
def fig_group_distribution(df: pd.DataFrame, out_dir: Path):
    """
    Bar chart showing nucleus count for each of the four groups.
    Separates 'good' (single_ok) from 'needs investigation' (everything else).
    """
    order  = ["single_ok", "single_unconfirmed", "zero", "multi"]
    labels = [
        "Single\n(1 candidate,\nconfirmed)",
        "Single\n(1 candidate,\nnot confirmed)",
        "Zero\n(no detection)",
        "Multi\n(≥2 candidates)",
    ]
    colors = [GROUP_COLORS[g] for g in order]
    counts = [len(df[df["group"] == g]) for g in order]
    n_total = len(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(4), counts, color=colors, width=0.55,
                  edgecolor="white", linewidth=1.5)

    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            cnt + n_total * 0.012,
            f"n = {cnt}\n({cnt / n_total * 100:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Nucleus count", fontsize=12)
    ax.set_ylim(0, max(counts) * 1.28)
    ax.set_title(
        f"Puncta Anchor — Nucleus Group Distribution  (n = {n_total} total)\n"
        f"Group 1 (single_ok): for barcode analysis  |  "
        f"Groups 2–4: needs investigation",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = out_dir / "fig_group_distribution.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Figure 2: Single_ok barcode distribution ───────────────────────────────
def fig_single_ok_barcodes(df: pd.DataFrame, out_dir: Path):
    """
    Bar chart of barcode frequencies for the single_ok group.

    Scientific logic
    ----------------
    These 79% of nuclei each had exactly 1 LoG-detected candidate that
    confirmed in both Hyb3 and Hyb2.  Their barcodes should map onto the
    known drug/condition identity.  This chart lets us:
      1. Verify expected barcode diversity (not all one color → good mixing)
      2. Identify dominant barcodes for manual auditing
      3. Detect unexpected barcodes (barcode not in expected set → error)

    Bar color = first round's color (Hyb4 channel) for quick visual grouping.
    """
    single_ok = df[df["group"] == "single_ok"]
    if len(single_ok) == 0:
        print("  No single_ok nuclei — skipping barcode figure")
        return

    barcodes = single_ok["best_barcode"].value_counts().sort_values(ascending=False)

    def _first_color(bc: str) -> str:
        first = bc.split("-")[0] if "-" in bc else bc
        return COLOR_DISPLAY.get(first, "#95A5A6")

    bar_colors = [_first_color(bc) for bc in barcodes.index]

    fig, ax = plt.subplots(figsize=(max(12, len(barcodes) * 0.65 + 2), 5))
    bars = ax.bar(range(len(barcodes)), barcodes.values, color=bar_colors,
                  width=0.65, edgecolor="white", linewidth=0.8)

    for bar, cnt in zip(bars, barcodes.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            cnt + 0.5, str(cnt),
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(range(len(barcodes)))
    ax.set_xticklabels(barcodes.index, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Nucleus count", fontsize=12)
    ax.set_ylim(0, barcodes.values.max() * 1.18)
    ax.set_title(
        f"Barcode Distribution — single_ok group  "
        f"(n = {len(single_ok)},  {len(barcodes)} unique barcodes)\n"
        f"Nuclei with exactly 1 LoG candidate confirmed in both Hyb3 and Hyb2",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = out_dir / "fig_single_ok_barcodes.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}  ({len(barcodes)} unique barcodes)")


# ── Figure 3: None/Multi signal analysis ───────────────────────────────────
def fig_none_multi_signals(
    df_summary:  pd.DataFrame,
    df_cands:    pd.DataFrame,
    out_dir:     Path,
):
    """
    Two-panel figure for the none/multi group.

    Left panel  — Scatter: norm_max_h3 vs norm_max_h2 for ALL candidates
                  in none/multi nuclei, color-coded by confirmation status.
                  Threshold lines drawn at 2.0 (Ch1/Ch3) and 1.5 (Ch2).

    Right panel — Histogram: norm_max_h3 and norm_max_h2 distributions for
                  candidates that FAILED both rounds (neither confirmed).
                  Shows how far below the threshold these signals are —
                  helping us judge whether the threshold is too strict.

    Scientific logic
    ----------------
    A genuine smFISH punctum should appear at the same (y, x) in all rounds.
    If norm_max_h3 and norm_max_h2 are both below the threshold, either:
      (a) the cell is genuinely unlabeled (correct "None" call), or
      (b) the threshold is too strict for this cell (false negative).
    Inspecting how far below threshold the failed candidates fall helps
    distinguish case (a) from (b).
    """
    none_multi_ids = set(df_summary[df_summary["group"].isin(
        ["single_unconfirmed", "zero", "multi"])]["nucleus_id"])

    if df_cands is None or len(df_cands) == 0:
        print("  No candidates data — skipping none_multi signals figure")
        return

    df_nm = df_cands[df_cands["nucleus_id"].isin(none_multi_ids)].copy()
    n_nuclei = len(none_multi_ids)

    if len(df_nm) == 0:
        print(f"  none_multi group: {n_nuclei} nuclei, but 0 candidates in CSV")
        return

    # Label each candidate by its confirmation status
    def _status(row) -> str:
        if row["confirmed_h3"] and row["confirmed_h2"]:
            return "both_confirmed"
        elif row["confirmed_h3"]:
            return "h3_only"
        elif row["confirmed_h2"]:
            return "h2_only"
        else:
            return "neither"

    df_nm["cand_status"] = df_nm.apply(_status, axis=1)
    status_counts = df_nm["cand_status"].value_counts()
    print(f"  none_multi: {n_nuclei} nuclei, {len(df_nm)} candidates")
    for s, c in status_counts.items():
        print(f"    {s:<22}  {c:4d}")

    # ── Build figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    status_style = {
        "both_confirmed": ("#2ECC71", "Both H3+H2 confirmed",      60, "o", 0.75),
        "h3_only":        ("#F39C12", "Only H3 confirmed",          50, "^", 0.75),
        "h2_only":        ("#3498DB", "Only H2 confirmed",          50, "s", 0.75),
        "neither":        ("#E74C3C", "Neither confirmed (failed)", 30, "x", 0.60),
    }

    # ── Left: scatter norm_max_h3 vs norm_max_h2 ──────────────────────────
    ax0 = axes[0]
    for status, (color, label, sz, marker, alpha) in status_style.items():
        sub = df_nm[df_nm["cand_status"] == status]
        if len(sub) == 0:
            continue
        # "x" is an unfilled marker — edgecolors doesn't apply; use facecolors only
        scatter_kwargs = dict(
            c=color, label=f"{label}  (n={len(sub)})",
            s=sz, alpha=alpha, marker=marker, linewidths=0.8,
        )
        if marker != "x":
            scatter_kwargs["edgecolors"] = "none"
        ax0.scatter(sub["norm_max_h3"], sub["norm_max_h2"], **scatter_kwargs)

    # Threshold reference lines
    ax0.axvline(x=THRESH_STRICT, color="#444444", ls="--", lw=1.3,
                label=f"H3 threshold (Ch1/Ch3) = {THRESH_STRICT}")
    ax0.axhline(y=THRESH_STRICT, color="#444444", ls="--", lw=1.3,
                label=f"H2 threshold (Ch1/Ch3) = {THRESH_STRICT}")
    ax0.axvline(x=THRESH_MCHERRY, color="#AAAAAA", ls=":", lw=1.1,
                label=f"Threshold (Ch2) = {THRESH_MCHERRY}")
    ax0.axhline(y=THRESH_MCHERRY, color="#AAAAAA", ls=":", lw=1.1)

    # Shade the "both pass" quadrant (upper-right) lightly
    xlim_max = max(df_nm["norm_max_h3"].quantile(0.99), THRESH_STRICT + 1) * 1.1
    ylim_max = max(df_nm["norm_max_h2"].quantile(0.99), THRESH_STRICT + 1) * 1.1
    ax0.axhspan(THRESH_STRICT, ylim_max, xmin=THRESH_STRICT / xlim_max,
                alpha=0.06, color="#2ECC71")

    ax0.set_xlim(0, xlim_max)
    ax0.set_ylim(0, ylim_max)
    ax0.set_xlabel("norm_max_h3  (Hyb3 normalized signal)", fontsize=11)
    ax0.set_ylabel("norm_max_h2  (Hyb2 normalized signal)", fontsize=11)
    ax0.set_title(
        f"Hyb3 vs Hyb2 normalized signal\n"
        f"({n_nuclei} none/multi nuclei, {len(df_nm)} candidates total)",
        fontsize=11,
    )
    ax0.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax0.spines[["top", "right"]].set_visible(False)

    # ── Right: histogram of failed candidates ─────────────────────────────
    ax1 = axes[1]
    failed = df_nm[df_nm["cand_status"] == "neither"]

    if len(failed) > 0:
        max_val = max(
            failed["norm_max_h3"].max(),
            failed["norm_max_h2"].max(),
            THRESH_STRICT + 0.5,
        )
        bins = np.linspace(0, min(max_val * 1.1, 8.0), 35)

        ax1.hist(failed["norm_max_h3"], bins=bins, alpha=0.65, color="#F39C12",
                 label=f"Hyb3 norm_max  (n={len(failed)} candidates)",
                 edgecolor="white", linewidth=0.5)
        ax1.hist(failed["norm_max_h2"], bins=bins, alpha=0.65, color="#3498DB",
                 label=f"Hyb2 norm_max  (n={len(failed)} candidates)",
                 edgecolor="white", linewidth=0.5)

        ax1.axvline(x=THRESH_STRICT, color="#CC0000", ls="--", lw=1.8,
                    label=f"Threshold Ch1/Ch3 = {THRESH_STRICT}")
        ax1.axvline(x=THRESH_MCHERRY, color="#FF6666", ls=":", lw=1.5,
                    label=f"Threshold Ch2 = {THRESH_MCHERRY}")

        ax1.set_xlabel("Normalized max signal  (peak / nucleus_p25_background)", fontsize=11)
        ax1.set_ylabel("Candidate count", fontsize=11)
        ax1.set_title(
            f"Failed-filter candidates  (n = {len(failed)})\n"
            f"How far below threshold are the unconfirmed puncta?",
            fontsize=11,
        )
        ax1.legend(fontsize=9)
        ax1.spines[["top", "right"]].set_visible(False)

        # Annotate fraction close to threshold (within 0.5 of threshold)
        close_h3 = (failed["norm_max_h3"] >= THRESH_STRICT - 0.5).sum()
        close_h2 = (failed["norm_max_h2"] >= THRESH_STRICT - 0.5).sum()
        ax1.text(
            0.98, 0.97,
            f"Within 0.5 of threshold:\n"
            f"  Hyb3: {close_h3} ({close_h3/len(failed)*100:.0f}%)\n"
            f"  Hyb2: {close_h2} ({close_h2/len(failed)*100:.0f}%)",
            ha="right", va="top", transform=ax1.transAxes,
            fontsize=9, color="#555555",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F8F8",
                      edgecolor="#CCCCCC", alpha=0.9),
        )
    else:
        ax1.text(
            0.5, 0.5,
            "No fully-failed candidates\n(all candidates passed at least one round)",
            ha="center", va="center", transform=ax1.transAxes,
            fontsize=12, color="#888888",
        )
        ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax1.set_xticks([]); ax1.set_yticks([])

    fig.suptitle(
        f"None/Multi Group — Signal Analysis  |  {n_nuclei} nuclei, {len(df_nm)} candidates\n"
        f"Normalized threshold: Ch1/Ch3 ≥ {THRESH_STRICT},  Ch2 ≥ {THRESH_MCHERRY}"
        f"  (peak_in_window / nucleus_p25_background)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = out_dir / "fig_none_multi_signals.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Crop organization ──────────────────────────────────────────────────────
def organize_crops_into_groups(df: pd.DataFrame, crops_dir: Path, out_dir: Path):
    """
    Copy existing nucleus QC PNGs into two top-level group folders:

      single_ok_crops/          — 946 nuclei (1 candidate, confirmed)
      none_multi_crops/
        multi/                  — 259 nuclei (≥2 candidates)
        single_unconfirmed/     — 23  nuclei (1 candidate, not confirmed)
        zero/                   — 2   nuclei (no LoG detection)

    Uses shutil.copy2 (preserves timestamps).  The originals in nucleus_crops/
    are untouched — these are independent copies for browsing convenience.
    """
    dest_map = {
        "single_ok":          out_dir / "single_ok_crops",
        "multi":              out_dir / "none_multi_crops" / "multi",
        "single_unconfirmed": out_dir / "none_multi_crops" / "single_unconfirmed",
        "zero":               out_dir / "none_multi_crops" / "zero",
    }
    for d in dest_map.values():
        d.mkdir(parents=True, exist_ok=True)

    copied  = {g: 0 for g in dest_map}
    missing = []

    for _, row in df.iterrows():
        nid   = int(row["nucleus_id"])
        group = row["group"]
        src   = crops_dir / f"nucleus_{nid:04d}.png"
        dst_d = dest_map.get(group)
        if dst_d is None:
            continue
        if not src.exists():
            missing.append(nid)
            continue
        shutil.copy2(str(src), str(dst_d / src.name))
        copied[group] += 1

    print(f"  Organized nucleus crops into group folders:")
    for g, cnt in copied.items():
        print(f"    {g:<22}  {cnt:4d} files → {dest_map[g].relative_to(out_dir.parent.parent)}")
    if missing:
        print(f"  WARNING: {len(missing)} nucleus PNGs not found in {crops_dir}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Puncta Anchor Group QC Analysis")
    print("=" * 65)

    if not SUMMARY_CSV.exists():
        print(f"ERROR: {SUMMARY_CSV} not found.")
        print("Run run_puncta_anchor.py first to generate anchor outputs.")
        sys.exit(1)

    df_summary = pd.read_csv(str(SUMMARY_CSV))
    df_cands   = pd.read_csv(str(CANDIDATES_CSV)) if CANDIDATES_CSV.exists() else None

    if df_cands is None:
        print(f"WARNING: {CANDIDATES_CSV} not found — candidate-level analysis unavailable")

    # ── Classify nuclei ────────────────────────────────────────────────────
    df_summary = classify_nuclei(df_summary)
    n_total    = len(df_summary)
    group_counts = df_summary["group"].value_counts()

    print(f"\nNucleus classification  (n = {n_total}):")
    for g in ["single_ok", "single_unconfirmed", "zero", "multi"]:
        cnt = group_counts.get(g, 0)
        print(f"  {g:<25}  {cnt:4d}  ({cnt / n_total * 100:.1f}%)")

    none_multi_count = sum(
        group_counts.get(g, 0)
        for g in ["single_unconfirmed", "zero", "multi"]
    )
    print(f"\n  → none_multi total:  {none_multi_count}  ({none_multi_count / n_total * 100:.1f}%)")
    print(f"  → single_ok total:   {group_counts.get('single_ok', 0)}"
          f"  ({group_counts.get('single_ok', 0) / n_total * 100:.1f}%)")

    # ── CSV outputs ────────────────────────────────────────────────────────
    print("\nSaving CSV outputs...")

    # 1. Group labels for every nucleus
    df_summary[["nucleus_id", "n_candidates", "n_confirmed",
                "best_barcode", "decoded_ok", "group"]].to_csv(
        str(OUT_DIR / "group_labels.csv"), index=False)
    print(f"  Saved: group_labels.csv")

    # 2. Single_ok barcode counts
    single_ok = df_summary[df_summary["group"] == "single_ok"]
    bc_counts = (
        single_ok["best_barcode"]
        .value_counts()
        .rename_axis("barcode")
        .reset_index(name="count")
    )
    bc_counts.to_csv(str(OUT_DIR / "single_ok_barcodes.csv"), index=False)
    print(f"  Saved: single_ok_barcodes.csv  "
          f"({len(single_ok)} cells, {len(bc_counts)} unique barcodes)")

    # 3. Single_ok nucleus IDs per barcode (for manual auditing of crops)
    nid_by_barcode = (
        single_ok.groupby("best_barcode")["nucleus_id"]
        .apply(list)
        .reset_index()
        .rename(columns={"best_barcode": "barcode", "nucleus_id": "nucleus_ids"})
    )
    nid_by_barcode["nucleus_ids"] = nid_by_barcode["nucleus_ids"].apply(
        lambda ids: ",".join(str(x) for x in ids)
    )
    nid_by_barcode.to_csv(str(OUT_DIR / "single_ok_nucleus_ids.csv"), index=False)
    print(f"  Saved: single_ok_nucleus_ids.csv  "
          f"(nucleus IDs per barcode — for manual QC crop inspection)")

    # 4. None/multi candidates (including subthreshold)
    if df_cands is not None:
        none_multi_ids = set(df_summary[df_summary["group"].isin(
            ["single_unconfirmed", "zero", "multi"])]["nucleus_id"])
        df_nm = df_cands[df_cands["nucleus_id"].isin(none_multi_ids)]
        df_nm.to_csv(str(OUT_DIR / "none_multi_candidates.csv"), index=False)
        print(f"  Saved: none_multi_candidates.csv  "
              f"({len(none_multi_ids)} nuclei, {len(df_nm)} candidates)")
        # Report how many candidates are subthreshold (barcode == "unconfirmed")
        n_unconf = (df_nm["barcode"] == "unconfirmed").sum()
        print(f"    → of which {n_unconf} candidates are 'unconfirmed' "
              f"(failed Hyb3 or Hyb2 threshold)")

    # ── Figures ────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig_group_distribution(df_summary, OUT_DIR)
    fig_single_ok_barcodes(df_summary, OUT_DIR)
    fig_none_multi_signals(df_summary, df_cands, OUT_DIR)

    # ── Organize nucleus crops into group folders ──────────────────────────
    crops_dir = ANCHOR_DIR / "nucleus_crops"
    if crops_dir.exists() and any(crops_dir.glob("nucleus_*.png")):
        print("\nOrganizing nucleus QC crops into group folders...")
        organize_crops_into_groups(df_summary, crops_dir, OUT_DIR)
    else:
        print(f"\nWARNING: {crops_dir} not found or empty — run run_puncta_anchor.py first")

    print("\n" + "=" * 65)
    print("COMPLETE")
    print(f"  Outputs: {OUT_DIR}")
    print()
    print("  Nucleus crops organized into:")
    print(f"    groups/single_ok_crops/               ← browse confirmed single-punctum cells")
    print(f"    groups/none_multi_crops/multi/         ← ≥2 candidates (detection artifact?)")
    print(f"    groups/none_multi_crops/single_unconfirmed/  ← failed H3 or H2 threshold")
    print(f"    groups/none_multi_crops/zero/          ← no LoG detection at all")
    print()
    print("  Manual auditing workflow:")
    print("    1. Open single_ok_crops/ — sample a few per barcode, verify color calls")
    print("    2. Open none_multi_crops/multi/ — do these cells look like genuine multi-puncta?")
    print("    3. Open none_multi_crops/single_unconfirmed/ — why did they fail H3 or H2?")
    print("    4. Check fig_none_multi_signals.png — how close to threshold were failed cells?")
    print("=" * 65)


if __name__ == "__main__":
    main()
