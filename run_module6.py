"""
Run Module 6 — Barcode Decoding
=================================
Usage:
    python run_module6.py

Decodes 3-round × 3-channel fluorescence barcodes for each nucleus.
Uses argmax over channel intensities from Module 5, with a background
threshold to filter low-signal rounds.

Prerequisites:
    - Module 5 (python_results/module5/spot_intensities.csv)
    - Module 4 (python_results/module4/nucleus_properties.csv)

Outputs:
    python_results/module6/barcodes.csv
    python_results/module6/module6_decoding_QC.png
"""

import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.module6_decoding import Decoder

# Logging setup
log_dir = PROJECT_ROOT / "python_results" / "module6"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "module6_decoding.log"

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(str(log_path), level="DEBUG",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
           rotation="10 MB")

logger.info(f"Log file: {log_path}")


def main():
    for required in [
        PROJECT_ROOT / "python_results" / "module5" / "spot_intensities.csv",
        PROJECT_ROOT / "python_results" / "module4" / "nucleus_properties.csv",
    ]:
        if not required.exists():
            logger.error(f"Prerequisite missing: {required}")
            sys.exit(1)

    config_path = PROJECT_ROOT / "config" / "module6_decoding.yaml"
    decoder = Decoder(config_path=str(config_path), project_root=str(PROJECT_ROOT))
    df = decoder.run()

    print("\n" + "=" * 50)
    print("MODULE 6 COMPLETE")
    print("=" * 50)
    print(f"  Total nuclei     : {len(df)}")
    print(f"  Fully decoded    : {df['decoded_ok'].sum()}")
    print(f"  Unique barcodes  : {df['barcode'].nunique()}")
    print(f"\n  Top barcodes:")
    top = df[df["decoded_ok"]]["barcode"].value_counts().head(8)
    for bc, cnt in top.items():
        print(f"    {bc:35s}: {cnt} nuclei")
    print(f"\n  Barcodes CSV  : python_results/module6/barcodes.csv")
    print(f"  QC image      : python_results/module6/module6_decoding_QC.png")
    print("=" * 50)

    return df


if __name__ == "__main__":
    main()
