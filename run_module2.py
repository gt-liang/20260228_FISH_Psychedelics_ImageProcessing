"""
Run Module 2 — Live → Hyb4 Registration
=========================================
Usage:
    python run_module2.py

Prerequisite: Module 1 must have been run (python_results/module1/hyb4_crop_DAPI.npy)

Computes the translation (dy, dx) between 10x Live DAPI and 20x Hyb4 DAPI crop
using phase correlation. Output used in Module 6 to map Live cell IDs → Hyb4 space.
"""

import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.module2_live_hyb4_registration import LiveHyb4Registrar

# Configure logging
log_dir = PROJECT_ROOT / "python_results" / "module2"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "module2_registration.log"

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(str(log_path), level="DEBUG",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
           rotation="10 MB")

logger.info(f"Log file: {log_path}")

# Check M1 output exists
m1_output = PROJECT_ROOT / "python_results" / "module1" / "hyb4_crop_DAPI.npy"
if not m1_output.exists():
    logger.error(f"Module 1 output not found: {m1_output}")
    logger.error("Please run Module 1 first: python run_module1.py")
    sys.exit(1)


def main():
    config_path = PROJECT_ROOT / "config" / "module2_live_hyb4_registration.yaml"

    registrar = LiveHyb4Registrar(
        config_path=str(config_path),
        project_root=str(PROJECT_ROOT),
    )

    reg = registrar.run()

    print("\n" + "=" * 50)
    print("MODULE 2 COMPLETE")
    print("=" * 50)
    print(f"  dy (row shift) : {reg['dy']:+.2f} px")
    print(f"  dx (col shift) : {reg['dx']:+.2f} px")
    print(f"  Magnitude      : {reg['shift_magnitude_px']:.2f} px")
    print(f"  Pearson r      : {reg['pearson_r']:.4f}  (higher=better)")
    print(f"\n  Interpretation:")
    print(f"    Live DAPI must be shifted by (dy={reg['dy']:+.2f}, dx={reg['dx']:+.2f}) px")
    print(f"    to align with Hyb4 DAPI crop.")
    print(f"\n  Results saved to: python_results/module2/")
    print("=" * 50)

    return reg


if __name__ == "__main__":
    main()
