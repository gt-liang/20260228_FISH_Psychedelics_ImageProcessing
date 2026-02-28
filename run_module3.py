"""
Run Module 3 — Hyb-to-Hyb BF Registration
==========================================
Usage:
    python run_module3.py

Registers Hyb2 and Hyb3 BF channels to the Hyb4 BF crop (reference frame)
using phase cross-correlation.

Prerequisites:
    - Module 1 must have been run (python_results/module1/crop_coords.json
      and hyb4_crop_BF.npy must exist)

Outputs:
    python_results/module3/registration_hyb2_to_hyb4.json
    python_results/module3/registration_hyb3_to_hyb4.json
    python_results/module3/module3_registration_QC_hyb2.png
    python_results/module3/module3_registration_QC_hyb3.png
"""

import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.module3_hyb_registration import HybRegistrar

# Logging setup
log_dir = PROJECT_ROOT / "python_results" / "module3"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "module3_hyb_registration.log"

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(str(log_path), level="DEBUG",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
           rotation="10 MB")

logger.info(f"Log file: {log_path}")


def main():
    # Guard: M1 prerequisites must exist
    m1_crop_coords = PROJECT_ROOT / "python_results" / "module1" / "crop_coords.json"
    m1_bf_crop = PROJECT_ROOT / "python_results" / "module1" / "hyb4_crop_BF.npy"
    for required in [m1_crop_coords, m1_bf_crop]:
        if not required.exists():
            logger.error(f"Prerequisite missing: {required}")
            logger.error("Run Module 1 first (python run_module1.py)")
            sys.exit(1)

    config_path = PROJECT_ROOT / "config" / "module3_hyb_registration.yaml"
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    registrar = HybRegistrar(
        config_path=str(config_path),
        project_root=str(PROJECT_ROOT),
    )

    results = registrar.run()

    print("\n" + "=" * 50)
    print("MODULE 3 COMPLETE")
    print("=" * 50)
    for label, reg in results.items():
        print(f"\n  [{label.upper()} → Hyb4]")
        print(f"    dy           : {reg['dy']:+.2f} px")
        print(f"    dx           : {reg['dx']:+.2f} px")
        print(f"    Magnitude    : {reg['shift_magnitude_px']:.2f} px")
        print(f"    Pearson r    : {reg['pearson_r']:.4f}  (higher=better)")
    print(f"\n  Results saved to: python_results/module3/")
    print("=" * 50)

    return results


if __name__ == "__main__":
    main()
