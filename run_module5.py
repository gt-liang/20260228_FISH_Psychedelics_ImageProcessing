"""
Run Module 5 — Spot Calling
============================
Usage:
    python run_module5.py

For each nucleus × each hybridization round × each fluorescence channel,
extracts the maximum pixel intensity within the nucleus mask (Method X).

Prerequisites:
    - Module 1 (hyb4_crop_*.npy, crop_coords.json)
    - Module 3 (registration_hyb{2,3}_to_hyb4.json)
    - Module 4 (nucleus_labels.npy, nucleus_properties.csv)

Outputs:
    python_results/module5/spot_intensities.csv
    python_results/module5/module5_spot_calling_QC.png
"""

import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.module5_spot_calling import SpotCaller

# Logging setup
log_dir = PROJECT_ROOT / "python_results" / "module5"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "module5_spot_calling.log"

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(str(log_path), level="DEBUG",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
           rotation="10 MB")

logger.info(f"Log file: {log_path}")


def main():
    for required in [
        PROJECT_ROOT / "python_results" / "module1" / "hyb4_crop_Ch1_AF647.npy",
        PROJECT_ROOT / "python_results" / "module3" / "registration_hyb2_to_hyb4.json",
        PROJECT_ROOT / "python_results" / "module3" / "registration_hyb3_to_hyb4.json",
        PROJECT_ROOT / "python_results" / "module4" / "nucleus_labels.npy",
    ]:
        if not required.exists():
            logger.error(f"Prerequisite missing: {required}")
            sys.exit(1)

    config_path = PROJECT_ROOT / "config" / "module5_spot_calling.yaml"
    caller = SpotCaller(config_path=str(config_path), project_root=str(PROJECT_ROOT))
    df = caller.run()

    print("\n" + "=" * 50)
    print("MODULE 5 COMPLETE")
    print("=" * 50)
    print(f"  Total rows        : {len(df)}")
    print(f"  Unique nuclei     : {df['nucleus_id'].nunique()}")
    print(f"  Rounds            : {sorted(df['round'].unique())}")
    print(f"  Intensities saved : python_results/module5/spot_intensities.csv")
    print(f"  QC image          : python_results/module5/module5_spot_calling_QC.png")
    print("=" * 50)

    return df


if __name__ == "__main__":
    main()
