"""
Run Module 4 — Nuclear Segmentation
=====================================
Usage:
    python run_module4.py

Segments nuclei from the Hyb4 DAPI crop using CellposeSAM.
Outputs a label image and per-nucleus properties CSV.

Prerequisites:
    - Module 1 must have been run (python_results/module1/hyb4_crop_DAPI.npy)

Outputs:
    python_results/module4/nucleus_labels.npy
    python_results/module4/nucleus_properties.csv
    python_results/module4/module4_segmentation_QC.png
"""

import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.module4_segmentation import NuclearSegmentor

# Logging setup
log_dir = PROJECT_ROOT / "python_results" / "module4"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "module4_segmentation.log"

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(str(log_path), level="DEBUG",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
           rotation="10 MB")

logger.info(f"Log file: {log_path}")


def main():
    # Guard: M1 prerequisite
    m1_dapi = PROJECT_ROOT / "python_results" / "module1" / "hyb4_crop_DAPI.npy"
    if not m1_dapi.exists():
        logger.error(f"Prerequisite missing: {m1_dapi}")
        logger.error("Run Module 1 first (python run_module1.py)")
        sys.exit(1)

    config_path = PROJECT_ROOT / "config" / "module4_segmentation.yaml"
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    segmentor = NuclearSegmentor(
        config_path=str(config_path),
        project_root=str(PROJECT_ROOT),
    )

    result = segmentor.run()

    print("\n" + "=" * 50)
    print("MODULE 4 COMPLETE")
    print("=" * 50)
    print(f"  Nuclei detected : {result['n_nuclei']}")
    print(f"  Labels saved    : python_results/module4/nucleus_labels.npy")
    print(f"  Properties CSV  : python_results/module4/nucleus_properties.csv")
    print(f"  QC image        : python_results/module4/module4_segmentation_QC.png")
    print("=" * 50)

    return result


if __name__ == "__main__":
    main()
