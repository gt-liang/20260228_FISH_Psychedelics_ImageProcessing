# 20260228 FISH Psychedelics Image Processing

**Lab**: Boeynaems Lab, Baylor College of Medicine
**Status**: ✅ Full pipeline operational — mCherry correction implemented
**Last Updated**: 2026-03-01

---

## Project Overview

Multi-round smFISH barcode decoding pipeline for the Psychedelics project.
Each cell was labeled with a unique combination of fluorescent oligos across
3 hybridization rounds (Hyb2, Hyb3, Hyb4). The pipeline assigns a 3-color
barcode to every nucleus — e.g., `Purple-Yellow-Blue` — which identifies the
drug or condition that cell was exposed to.

### Color Mapping
| Channel | Fluorophore | Color Label |
|---------|-------------|-------------|
| Ch1 | AF647 | Purple |
| Ch2 | AF590 | Blue |
| Ch3 | AF488 | Yellow |
| DAPI | — | Segmentation only |

### Imaging Data
| Round | Imaging Order | Notes |
|-------|--------------|-------|
| Hyb4 | 1st imaged | DAPI present; reference round for registration |
| Hyb3 | 2nd imaged | DNase-treated → DAPI ≈ 0; use BF for registration |
| Hyb2 | 3rd imaged | DNase-treated → DAPI ≈ 0; use BF for registration |

> **Note**: Imaging order is Hyb4 → Hyb3 → Hyb2. mCherry (Ch2/AF590) bleaches ~19.8%
> over this sequence — the pipeline accounts for this with a cross-round correction.

---

## Pipeline Architecture

```
Raw .tif Images (Hyb2/3/4 + Live)
          │
          ▼
┌─────────────────────────────────┐
│ Module 1 — FOV Mapping          │  run_module1.py
│ DAPI template matching          │  config/module1_fov_mapping.yaml
│ → crop_coords.json              │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Module 2 — Live → Hyb4 Reg.    │  run_module2.py
│ Phase correlation (DAPI)        │  config/module2_live_hyb4_registration.yaml
│ → registration_live_hyb4.json   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Module 3 — Hyb-to-Hyb Reg.     │  run_module3.py
│ Phase correlation (BF)          │  config/module3_hyb_registration.yaml
│ Hyb2→Hyb4, Hyb3→Hyb4           │
│ → registration_hybN_to_hyb4.json│
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Module 4 — Nuclear Segmentation │  run_module4.py
│ CellposeSAM v4 on Hyb4 DAPI    │  config/module4_segmentation.yaml
│ 1230 nuclei detected            │
│ → nucleus_labels.npy            │
│ → nucleus_properties.csv        │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Module 5 — Spot Calling         │  run_module5.py
│ Method X: max pixel per nucleus │  config/module5_spot_calling.yaml
│ per channel per round           │
│ 3-level correction cascade:     │
│   raw → _corr → _xr (Ch2 only) │
│ → spot_intensities.csv          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Module 6 — Barcode Decoding     │  run_module6.py
│ Argmax over 3 channels/round    │  config/module6_decoding.yaml
│ 22 unique barcodes decoded      │
│ 1216/1230 (98.9%) decoded       │
│ → barcodes.csv                  │
└─────────────────────────────────┘
```

---

## Background Correction Methodology

Module 5 computes intensities at three correction levels for full traceability:

| Column Suffix | Level | Description |
|---|---|---|
| (none) | Raw | Max pixel intensity, no correction |
| `_bg` | Spatial | Per-nucleus local background estimate |
| `_corr` | Spatial subtracted | Raw − spatial background |
| `_xr_bg` | Cross-round min | min(`_corr` across Hyb2/Hyb3/Hyb4) per nucleus |
| `_xr` | Cross-round subtracted | `_corr` − `_xr_bg`, clipped to 0 |

**Module 6 uses channel-specific correction (scientifically justified):**

| Channel | Column used | Reason |
|---|---|---|
| Ch1_AF647 (Purple) | `_corr` | No persistent fluorescent protein |
| Ch2_AF590 (Blue) | `_xr` | mCherry persists across all rounds → requires cross-round correction |
| Ch3_AF488 (Yellow) | `_corr` | No persistent fluorescent protein |

> **Why not `_xr` for all channels?** Ch1/Ch3 cross-round carry-over (~800 ADU median)
> is the same order of magnitude as borderline FISH signals. Applying `_xr` to Ch1/Ch3
> incorrectly removes genuine Purple/Yellow signals → false None calls.

---

## Current Results (B7-FOVB)

| Metric | Value |
|---|---|
| Nuclei detected | 1,230 |
| Fully decoded | **1,216 (98.9%)** |
| None population | 14 (1.1%) — biologically expected |
| Unique barcodes | 22 |
| Top barcode | Purple-Yellow-Yellow (n=384) |

The 14 None cells are retained in `barcodes.csv` with `decoded_ok=False`.
Downstream analysis should filter on `decoded_ok=True`.

---

## Repo Structure

```
├── src/
│   ├── module1_fov_mapping/
│   ├── module2_live_hyb4_registration/
│   ├── module3_hyb_registration/
│   ├── module4_segmentation/
│   ├── module5_spot_calling/
│   │   ├── spot_caller.py          # Method X (primary) + cross-round correction
│   │   └── method_y_caller.py      # Method Y: Ronan-style adaptive threshold
│   └── module6_decoding/
│       └── decoder.py              # Argmax barcode decoder
│
├── config/                         # YAML config files (one per module)
│
├── run_module{1-6}.py              # Module entry points
├── run_qc_enhanced.py              # M5 + M6 QC figures (spot overlays, heatmaps)
├── run_qc_bg_comparison.py         # Before/after background subtraction QC
├── run_mcherry_correction.py       # Standalone mCherry cross-round exploration
├── run_method_y.py                 # Method Y puncta caller + comparison vs Method X
│
├── tasks/
│   ├── todo.md                     # Task list + module status
│   └── lessons.md                  # Lab knowledge base
│
├── Registration_src/               # Reference: Ronan O'Connell's original pipeline
├── docs/
└── .gitignore                      # Excludes *.tif, *.npy, *.csv, *.png, IMAGES/
```

---

## QC Outputs

All QC figures are written to `python_results/qc/` (excluded from git).

| Figure | Script | Description |
|---|---|---|
| `qc_m5_channel_scatter.png` | `run_qc_enhanced.py` | Ch1 vs Ch3 per round; dual-high nuclei highlighted |
| `qc_m5_dual_high_spatial.png` | `run_qc_enhanced.py` | Spatial map of dual-high population |
| `qc_m5_snr_distribution.png` | `run_qc_enhanced.py` | SNR (max/2nd-max) distribution per round |
| `qc_m5_intensity_heatmap.png` | `run_qc_enhanced.py` | Nucleus × channel intensity heatmap per barcode |
| `qc_m6_spot_overlay_Hyb{2,3,4}.png` | `run_qc_enhanced.py` | Fluorescence + decoded barcode color overlay |
| `qc_m6_barcode_counts.png` | `run_qc_enhanced.py` | Decoded barcode frequency chart |
| `qc_m6_confidence_map.png` | `run_qc_enhanced.py` | SNR spatial map across FOV |
| `qc_bg_scatter_comparison.png` | `run_qc_bg_comparison.py` | Raw vs corrected channel scatter |
| `qc_bg_background_magnitude.png` | `run_qc_bg_comparison.py` | Background pedestal by channel and round |
| `qc_bg_call_changes.png` | `run_qc_bg_comparison.py` | Cell color changes after correction |
| `qc_bg_snr_improvement.png` | `run_qc_bg_comparison.py` | SNR before vs after (per round) |
| `qc_bg_snr_by_color.png` | `run_qc_bg_comparison.py` | SNR per color × per round (3×3 grid) |

---

## How to Run

```bash
# Activate environment
conda activate idr-pipeline
# or use directly:
/Users/guo-tengliang/miniconda3/envs/idr-pipeline/bin/python

# Run full pipeline (in order)
python run_module1.py   # FOV mapping
python run_module2.py   # Live → Hyb4 registration
python run_module3.py   # Hyb-to-Hyb registration
python run_module4.py   # Nuclear segmentation
python run_module5.py   # Spot calling + background correction
python run_module6.py   # Barcode decoding

# QC figures
python run_qc_enhanced.py        # M5 + M6 visualizations
python run_qc_bg_comparison.py   # Background correction QC
```

> All configs live in `config/`. Adjust paths and thresholds there — do not edit module source directly.

---

## Next Steps

- [ ] Puncta detection method cross-comparison (Method X vs Y vs LoG vs TrackPy)
- [ ] Map barcodes to drug/condition identity (requires barcode lookup table)
- [ ] Cell ID mapping: Live → Hyb4 (Module 7 candidate)
- [ ] Per-condition statistics and spatial clustering analysis

---

## Environment

```bash
conda activate idr-pipeline
pip install -r requirements.txt
```

Key dependencies: `cellpose>=4`, `aicsimageio`, `scikit-image`, `pandas`, `loguru`, `pyyaml`, `matplotlib`

> Raw `.tif` images, `.npy` arrays, `.csv` results, and `.png` figures are excluded from git.
> Data lives on OneDrive/local drives. Only code and configs are tracked here.

---

## Reference Pipeline

`Registration_src/` contains Ronan O'Connell's original FISH pipeline (2025):
- `1_initial_cleanup.py` — CZI → MIP → npy
- `2_cellpose.py` — Nuclear segmentation
- `4_puncta_detection_multi-channel.py` — Puncta detection + registration (Method Y basis)
