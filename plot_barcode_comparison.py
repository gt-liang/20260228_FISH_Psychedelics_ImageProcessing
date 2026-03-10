"""
plot_barcode_comparison.py
==========================
Side-by-side barcode distribution comparison: v5 vs v6-bigfish.
Mirrors the style of the v5 barcode_distribution.png.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

BASE    = os.path.dirname(os.path.abspath(__file__))
V5_SUM  = os.path.join(BASE, "python_results/puncta_anchor/anchor_summary.csv")
V6_SUM  = os.path.join(BASE, "python_results/puncta_bigfish/anchor_summary.csv")
OUT_DIR = os.path.join(BASE, "python_results/comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colors ──────────────────────────────────────────────────────────────────
CH_COLOR = {
    "Yellow" : "#F5C518",
    "Purple" : "#8B5CF6",
    "Blue"   : "#3B9FD4",
    "None"   : "#AAAAAA",
}

def h4_color(barcode: str) -> str:
    """Extract H4 (first) color from 'H4-H3-H2' barcode string."""
    if not isinstance(barcode, str) or "-" not in barcode:
        return "None"
    return barcode.split("-")[0]

def bar_color(barcode: str) -> str:
    return CH_COLOR.get(h4_color(barcode), "#AAAAAA")

# ── Load ─────────────────────────────────────────────────────────────────────
v5 = pd.read_csv(V5_SUM)
v6 = pd.read_csv(V6_SUM)

v5_decoded = v5[v5["decoded_ok"] == True]
v6_decoded = v6[v6["decoded_ok"] == True]

n_v5 = len(v5_decoded)
n_v6 = len(v6_decoded)
n_total = len(v5)

# Barcode counts (top 20 each)
v5_counts = v5_decoded["best_barcode"].value_counts()
v6_counts = v6_decoded["best_barcode"].value_counts()

# All barcodes appearing in either pipeline (top 20 union, sorted by v5 count)
all_bc = sorted(
    set(v5_counts.index) | set(v6_counts.index),
    key=lambda b: -v5_counts.get(b, 0)
)[:22]  # top 22 for readability

v5_vals = [v5_counts.get(b, 0) for b in all_bc]
v6_vals = [v6_counts.get(b, 0) for b in all_bc]

# ── Hyb4 color breakdown ──────────────────────────────────────────────────────
def color_breakdown(decoded_df):
    colors = decoded_df["best_barcode"].apply(h4_color)
    total = len(colors)
    cnt = colors.value_counts()
    return {c: cnt.get(c, 0) / total * 100 for c in ["Yellow","Purple","Blue","None"]}

v5_h4 = color_breakdown(v5_decoded)
v6_h4 = color_breakdown(v6_decoded)

# ── Figure layout: 2 rows ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor("white")

# Title
fig.suptitle(
    f"Barcode Distribution Comparison  |  v5 (LoG + normalized): {n_v5}/{n_total} decoded  "
    f"vs  v6-bigfish (Big-FISH + spectral purity): {n_v6}/{n_total} decoded",
    fontsize=13, fontweight="bold", y=0.98
)

# Grid: top row = two bar charts; bottom row = two pie charts
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35,
                      top=0.92, bottom=0.06, left=0.06, right=0.97)

ax_bar_v5 = fig.add_subplot(gs[0, 0])
ax_bar_v6 = fig.add_subplot(gs[0, 1])
ax_pie_v5 = fig.add_subplot(gs[1, 0])
ax_pie_v6 = fig.add_subplot(gs[1, 1])

x = np.arange(len(all_bc))
bar_w = 0.62

# ── v5 bar chart ──────────────────────────────────────────────────────────────
colors_v5 = [bar_color(b) for b in all_bc]
bars_v5 = ax_bar_v5.bar(x, v5_vals, width=bar_w, color=colors_v5,
                         edgecolor="white", linewidth=0.5)
for bar, val in zip(bars_v5, v5_vals):
    if val > 0:
        ax_bar_v5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                       str(val), ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax_bar_v5.set_xticks(x)
ax_bar_v5.set_xticklabels(all_bc, rotation=45, ha="right", fontsize=7.5)
ax_bar_v5.set_ylabel("Count", fontsize=10)
ax_bar_v5.set_title(
    f"v5 — Barcode distribution\n(n={n_v5} decoded nuclei, {100*n_v5/n_total:.1f}%)",
    fontsize=10, pad=6
)
ax_bar_v5.set_xlim(-0.6, len(all_bc)-0.4)
ax_bar_v5.spines[["top","right"]].set_visible(False)
ax_bar_v5.yaxis.grid(True, alpha=0.3, linestyle="--")
ax_bar_v5.set_axisbelow(True)

# ── v6 bar chart ──────────────────────────────────────────────────────────────
colors_v6 = [bar_color(b) for b in all_bc]
bars_v6 = ax_bar_v6.bar(x, v6_vals, width=bar_w, color=colors_v6,
                         edgecolor="white", linewidth=0.5)
for bar, val in zip(bars_v6, v6_vals):
    if val > 0:
        ax_bar_v6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                       str(val), ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax_bar_v6.set_xticks(x)
ax_bar_v6.set_xticklabels(all_bc, rotation=45, ha="right", fontsize=7.5)
ax_bar_v6.set_ylabel("Count", fontsize=10)
ax_bar_v6.set_title(
    f"v6-bigfish — Barcode distribution\n(n={n_v6} decoded nuclei, {100*n_v6/n_total:.1f}%)",
    fontsize=10, pad=6
)
ax_bar_v6.set_xlim(-0.6, len(all_bc)-0.4)
ax_bar_v6.spines[["top","right"]].set_visible(False)
ax_bar_v6.yaxis.grid(True, alpha=0.3, linestyle="--")
ax_bar_v6.set_axisbelow(True)

# Match y-axis scale
y_max = max(max(v5_vals), max(v6_vals)) * 1.15
ax_bar_v5.set_ylim(0, y_max)
ax_bar_v6.set_ylim(0, y_max)

# ── Color legend for bars ─────────────────────────────────────────────────────
legend_patches = [mpatches.Patch(color=v, label=k) for k, v in CH_COLOR.items()]
ax_bar_v5.legend(handles=legend_patches, title="H4 color",
                 fontsize=8, title_fontsize=8, loc="upper right", framealpha=0.8)

# ── v5 pie ────────────────────────────────────────────────────────────────────
pie_colors = [CH_COLOR[c] for c in ["Yellow","Purple","Blue","None"]]
pie_labels_v5 = [
    f"{c}\n{v5_h4.get(c,0):.1f}%" for c in ["Yellow","Purple","Blue","None"]
    if v5_h4.get(c, 0) > 0
]
pie_vals_v5 = [v5_h4.get(c, 0) for c in ["Yellow","Purple","Blue","None"] if v5_h4.get(c,0)>0]
pie_cols_v5 = [CH_COLOR[c] for c in ["Yellow","Purple","Blue","None"] if v5_h4.get(c,0)>0]

ax_pie_v5.pie(pie_vals_v5, labels=pie_labels_v5, colors=pie_cols_v5,
              autopct=None, startangle=90,
              wedgeprops=dict(edgecolor="white", linewidth=1.5),
              textprops=dict(fontsize=9))
ax_pie_v5.set_title("v5 — Hyb4 color distribution", fontsize=10, pad=8)

# ── v6 pie ────────────────────────────────────────────────────────────────────
pie_labels_v6 = [
    f"{c}\n{v6_h4.get(c,0):.1f}%" for c in ["Yellow","Purple","Blue","None"]
    if v6_h4.get(c, 0) > 0
]
pie_vals_v6 = [v6_h4.get(c, 0) for c in ["Yellow","Purple","Blue","None"] if v6_h4.get(c,0)>0]
pie_cols_v6 = [CH_COLOR[c] for c in ["Yellow","Purple","Blue","None"] if v6_h4.get(c,0)>0]

ax_pie_v6.pie(pie_vals_v6, labels=pie_labels_v6, colors=pie_cols_v6,
              autopct=None, startangle=90,
              wedgeprops=dict(edgecolor="white", linewidth=1.5),
              textprops=dict(fontsize=9))
ax_pie_v6.set_title("v6-bigfish — Hyb4 color distribution", fontsize=10, pad=8)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "barcode_distribution_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_path}")
