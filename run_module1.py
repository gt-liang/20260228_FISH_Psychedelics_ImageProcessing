"""
Run Module 1 — FOV Mapping
===========================
Usage:
    python run_module1.py

Finds the 10x Live FOV in the 20x Hyb4 tiled image via BF template matching.
Outputs: python_results/module1/crop_coords.json + hyb4_crop_*.npy
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.module1_fov_mapping import FOVMapper

# Configure loguru: log to console + file
log_dir = PROJECT_ROOT / "python_results" / "module1"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "module1_fov_mapping.log"

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(str(log_path), level="DEBUG",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
           rotation="10 MB")

logger.info(f"Log file: {log_path}")


def main():
    config_path = PROJECT_ROOT / "config" / "module1_fov_mapping.yaml"

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    mapper = FOVMapper(
        config_path=str(config_path),
        project_root=str(PROJECT_ROOT),
    )

    crop_info = mapper.run()

    # Final summary print
    print("\n" + "=" * 50)
    print("MODULE 1 COMPLETE")
    print("=" * 50)
    print(f"  Crop y0      : {crop_info['y0']}")
    print(f"  Crop x0      : {crop_info['x0']}")
    print(f"  Crop H × W   : {crop_info['crop_h']} × {crop_info['crop_w']}")
    print(f"  Match score  : {crop_info['match_score']:.4f}  (1.0 = perfect)")
    print(f"  dy offset    : {crop_info['offset_from_center_dy']:+d} px from center")
    print(f"  dx offset    : {crop_info['offset_from_center_dx']:+d} px from center")
    print(f"\n  Results saved to: python_results/module1/")
    print("=" * 50)

    return crop_info


if __name__ == "__main__":
    main()
