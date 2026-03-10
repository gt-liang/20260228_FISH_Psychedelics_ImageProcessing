"""
compare_v5_v6.py
================
Compare v5 (LoG + normalized-ratio) vs v6-bigfish (Big-FISH + spectral-purity)
pipeline results.

Outputs
-------
- Console report
- python_results/comparison/comparison_report.txt
- python_results/comparison/barcode_comparison.csv
- python_results/comparison/wrong_winner_review.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
V5_SUMMARY   = os.path.join(BASE, "python_results/puncta_anchor/anchor_summary.csv")
V6_SUMMARY   = os.path.join(BASE, "python_results/puncta_bigfish/anchor_summary.csv")
QC_REVIEW    = os.path.join(BASE, "python_results/puncta_anchor/qc_review.csv")
OUT_DIR      = os.path.join(BASE, "python_results/comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
v5 = pd.read_csv(V5_SUMMARY)
v6 = pd.read_csv(V6_SUMMARY)
qc = pd.read_csv(QC_REVIEW)

print(f"v5 rows: {len(v5)}  v6 rows: {len(v6)}  qc_review rows: {len(qc)}")

# Merge on nucleus_id
merged = v5.merge(v6, on="nucleus_id", suffixes=("_v5", "_v6"), how="outer")
merged["decoded_ok_v5"] = merged["decoded_ok_v5"].fillna(False)
merged["decoded_ok_v6"] = merged["decoded_ok_v6"].fillna(False)

# ---------------------------------------------------------------------------
# Decoded rate overview
# ---------------------------------------------------------------------------
n_total  = len(merged)
n_v5_ok  = merged["decoded_ok_v5"].sum()
n_v6_ok  = merged["decoded_ok_v6"].sum()
n_both   = (merged["decoded_ok_v5"] & merged["decoded_ok_v6"]).sum()
n_v5only = (merged["decoded_ok_v5"] & ~merged["decoded_ok_v6"]).sum()
n_v6only = (~merged["decoded_ok_v5"] & merged["decoded_ok_v6"]).sum()
n_neither= (~merged["decoded_ok_v5"] & ~merged["decoded_ok_v6"]).sum()

# Among both decoded, same barcode?
both_decoded = merged[merged["decoded_ok_v5"] & merged["decoded_ok_v6"]].copy()
same_barcode = (both_decoded["best_barcode_v5"] == both_decoded["best_barcode_v6"]).sum()
diff_barcode = len(both_decoded) - same_barcode

lines = []
lines.append("=" * 65)
lines.append("  v5 vs v6-bigfish Comparison Report")
lines.append("=" * 65)
lines.append(f"  Total nuclei           : {n_total}")
lines.append(f"  v5  decoded            : {n_v5_ok}  ({100*n_v5_ok/n_total:.1f}%)")
lines.append(f"  v6  decoded            : {n_v6_ok}  ({100*n_v6_ok/n_total:.1f}%)")
lines.append(f"")
lines.append(f"  Both decoded           : {n_both}")
lines.append(f"    Same barcode         : {same_barcode}  ({100*same_barcode/max(n_both,1):.1f}%)")
lines.append(f"    Different barcode    : {diff_barcode}  ({100*diff_barcode/max(n_both,1):.1f}%)")
lines.append(f"  v5-only decoded        : {n_v5only}")
lines.append(f"  v6-only decoded        : {n_v6only}")
lines.append(f"  Neither decoded        : {n_neither}")

# ---------------------------------------------------------------------------
# Barcode distribution comparison
# ---------------------------------------------------------------------------
v5_barcodes = v5[v5["decoded_ok"]==True]["best_barcode"].value_counts()
v6_barcodes = v6[v6["decoded_ok"]==True]["best_barcode"].value_counts()

all_barcodes = sorted(set(v5_barcodes.index) | set(v6_barcodes.index))
bc_df = pd.DataFrame({
    "barcode"  : all_barcodes,
    "v5_count" : [v5_barcodes.get(b, 0) for b in all_barcodes],
    "v6_count" : [v6_barcodes.get(b, 0) for b in all_barcodes],
})
bc_df["v5_pct"] = (bc_df["v5_count"] / n_v5_ok * 100).round(1)
bc_df["v6_pct"] = (bc_df["v6_count"] / max(n_v6_ok, 1) * 100).round(1)
bc_df["delta"]  = bc_df["v6_count"] - bc_df["v5_count"]
bc_df = bc_df.sort_values("v5_count", ascending=False).reset_index(drop=True)

lines.append("")
lines.append("-" * 65)
lines.append("  Barcode Distribution Comparison (sorted by v5 count)")
lines.append("-" * 65)
lines.append(f"  {'Barcode':<28} {'v5':>6} {'v5%':>6} {'v6':>6} {'v6%':>6} {'Δ':>6}")
lines.append(f"  {'-'*28} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
for _, row in bc_df.iterrows():
    lines.append(f"  {row['barcode']:<28} {int(row['v5_count']):>6} {row['v5_pct']:>6.1f} "
                 f"{int(row['v6_count']):>6} {row['v6_pct']:>6.1f} {int(row['delta']):>+6}")

# ---------------------------------------------------------------------------
# Wrong-winner analysis (73 cases from QC review)
# ---------------------------------------------------------------------------
wrong_winner = qc[qc["verdict"] == "wrong_winner"].copy()
lines.append("")
lines.append("-" * 65)
lines.append(f"  Wrong-winner cases from QC review: {len(wrong_winner)}")
lines.append("-" * 65)

# Join with v6 results
ww = wrong_winner.merge(
    v6[["nucleus_id", "best_barcode", "decoded_ok"]].rename(
        columns={"best_barcode": "v6_barcode", "decoded_ok": "v6_decoded"}),
    on="nucleus_id", how="left"
).merge(
    v5[["nucleus_id", "best_barcode", "decoded_ok"]].rename(
        columns={"best_barcode": "v5_barcode", "decoded_ok": "v5_decoded"}),
    on="nucleus_id", how="left"
)

# How many are now decoded in v6?
ww_decoded = ww["v6_decoded"].sum()
ww_not_decoded = len(ww) - ww_decoded

# How many changed barcode?
ww_both = ww[ww["v6_decoded"] & ww["v5_decoded"]]
ww_same = (ww_both["v5_barcode"] == ww_both["v6_barcode"]).sum()
ww_changed = len(ww_both) - ww_same

lines.append(f"  v6 decoded (among wrong_winner): {ww_decoded}/{len(ww_winner := ww)}")
lines.append(f"  v6 not decoded                 : {ww_not_decoded}")
lines.append(f"  Both decoded, same barcode     : {ww_same}")
lines.append(f"  Both decoded, changed barcode  : {ww_changed}")

# Show sample of changed barcodes
if ww_changed > 0:
    changed_rows = ww_both[ww_both["v5_barcode"] != ww_both["v6_barcode"]]
    lines.append(f"")
    lines.append(f"  Changed barcode cases (first 20):")
    lines.append(f"  {'nuc':>5}  {'v5_barcode':<28}  {'v6_barcode':<28}")
    for _, row in changed_rows.head(20).iterrows():
        lines.append(f"  {int(row['nucleus_id']):>5}  {str(row['v5_barcode']):<28}  {str(row['v6_barcode']):<28}")

# ---------------------------------------------------------------------------
# Wrong_barcode cases
# ---------------------------------------------------------------------------
wrong_bc = qc[qc["verdict"] == "wrong_barcode"].copy()
wb = wrong_bc.merge(
    v6[["nucleus_id", "best_barcode", "decoded_ok"]].rename(
        columns={"best_barcode": "v6_barcode", "decoded_ok": "v6_decoded"}),
    on="nucleus_id", how="left"
).merge(
    v5[["nucleus_id", "best_barcode", "decoded_ok"]].rename(
        columns={"best_barcode": "v5_barcode", "decoded_ok": "v5_decoded"}),
    on="nucleus_id", how="left"
)
wb_decoded = wb["v6_decoded"].sum()
wb_both    = wb[wb["v6_decoded"] & wb["v5_decoded"]]
wb_changed = (wb_both["v5_barcode"] != wb_both["v6_barcode"]).sum()

lines.append("")
lines.append(f"  Wrong-barcode cases from QC review: {len(wrong_bc)}")
lines.append(f"  v6 decoded (among wrong_barcode): {wb_decoded}/{len(wb)}")
lines.append(f"  Both decoded, changed barcode   : {wb_changed}")

# ---------------------------------------------------------------------------
# v5-only decoded: are these false positives filtered by purity?
# ---------------------------------------------------------------------------
v5only_ids = merged[merged["decoded_ok_v5"] & ~merged["decoded_ok_v6"]]["nucleus_id"]
lines.append("")
lines.append("-" * 65)
lines.append(f"  Nuclei decoded by v5 but NOT v6: {len(v5only_ids)}")
lines.append(f"  (These may be low-purity / bleedthrough false positives)")
# Barcode distribution for v5-only
v5only_barcodes = v5[v5["nucleus_id"].isin(v5only_ids)]["best_barcode"].value_counts().head(10)
lines.append(f"  Top barcodes in v5-only group:")
for bc, cnt in v5only_barcodes.items():
    lines.append(f"    {bc:<28} {cnt:>5}")

# ---------------------------------------------------------------------------
# Specific nuclei check: 93 and 63
# ---------------------------------------------------------------------------
lines.append("")
lines.append("-" * 65)
lines.append("  Spot-check: nucleus 93 and 63 (reported wrong_winner in QC)")
lines.append("-" * 65)
for nid in [93, 63]:
    v5_row = v5[v5["nucleus_id"] == nid]
    v6_row = v6[v6["nucleus_id"] == nid]
    qc_row = qc[qc["nucleus_id"] == nid]
    v5_bc  = v5_row["best_barcode"].values[0] if len(v5_row) else "N/A"
    v6_bc  = v6_row["best_barcode"].values[0] if len(v6_row) else "N/A"
    v5_ok  = v5_row["decoded_ok"].values[0] if len(v5_row) else False
    v6_ok  = v6_row["decoded_ok"].values[0] if len(v6_row) else False
    verdict= qc_row["verdict"].values[0] if len(qc_row) else "not in QC"
    correct= qc_row["correct_candidate"].values[0] if len(qc_row) else "?"
    lines.append(f"  Nucleus {nid}:")
    lines.append(f"    QC verdict  : {verdict}  (correct={correct})")
    lines.append(f"    v5 barcode  : {v5_bc}  (decoded={v5_ok})")
    lines.append(f"    v6 barcode  : {v6_bc}  (decoded={v6_ok})")

lines.append("")
lines.append("=" * 65)

# Print to console
report_text = "\n".join(lines)
print(report_text)

# Save report
report_path = os.path.join(OUT_DIR, "comparison_report.txt")
with open(report_path, "w") as f:
    f.write(report_text + "\n")
print(f"\nReport saved → {report_path}")

# Save barcode comparison CSV
bc_path = os.path.join(OUT_DIR, "barcode_comparison.csv")
bc_df.to_csv(bc_path, index=False)
print(f"Barcode CSV  → {bc_path}")

# Save wrong_winner review CSV
ww_path = os.path.join(OUT_DIR, "wrong_winner_review.csv")
ww.to_csv(ww_path, index=False)
print(f"Wrong-winner → {ww_path}")
