"""
Puncta Detection Method Comparison — Entry Point
=================================================
Runs all 6 puncta detection methods on B7-FOVB smFISH data and
generates a comparative QC report.

Usage:
    python run_puncta_comparison.py

Outputs (python_results/puncta_comparison/):
    signal_table.csv          — per nucleus × round × channel × method signals
    comparison_table.csv      — per nucleus: barcode call from each method
    barcodes_{X,Y,Z,W,T,P}.csv — per-method barcode tables
    qc_decoded_rate.png       — decoded rate bar chart
    qc_snr_distribution.png   — SNR violin plot per method
    qc_pairwise_agreement.png — method agreement heatmap
    qc_barcode_distribution.png — barcode frequency per method
    qc_disagreement_map.png   — spatial FOV map of agreement / disagreement
"""

import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncta_comparison import PunctaComparator

# ── Logging ────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
log_path = PROJECT_ROOT / "python_results/puncta_comparison/puncta_comparison.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
logger.add(str(log_path), rotation="10 MB", level="DEBUG")

# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    comparator = PunctaComparator(
        config_path=str(PROJECT_ROOT / "config/puncta_comparison.yaml"),
        project_root=str(PROJECT_ROOT),
    )
    results, metrics = comparator.run()

    print()
    print("=" * 55)
    print("COMPARISON COMPLETE")
    print("=" * 55)
    m_keys = [m for m in results]
    print(f"  {'Method':<8} {'Decoded%':>9}  {'SNR median':>10}")
    print("  " + "-" * 35)
    for m in m_keys:
        if m in metrics:
            print(f"  {m:<8} {metrics[m]['decoded_rate']:>8.1f}%  "
                  f"{metrics[m]['snr_median']:>10.1f}")
    print()
    print("  Outputs: python_results/puncta_comparison/")
    print("=" * 55)
