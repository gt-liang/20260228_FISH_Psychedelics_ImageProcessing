# FISH Psychedelics Image Processing - Task List
**Last Updated**: 2026-02-28
**Status**: Full pipeline implemented and running ✅

---

## Project Goal
Multi-round smFISH barcode decoding for Psychedelics project.
Final output: Per-nucleus 3-round barcode — e.g., `Cell_42: Purple-Yellow-Blue`

---

## Data Inventory

| File | Dtype | Shape | Notes |
|------|-------|-------|-------|
| `B7-FOVB--t1118--C00.tif` | uint16 (12-bit) | (2032, 2432) | Live DAPI |
| `B7-FOVB--t1118--C03.tif` | uint16 (12-bit) | (2032, 2432) | Live BF |
| `hyb4_ICC_Processed001.tif` | uint16 | (1,1,5,5714,6852) | Z0=DAPI, Z1=Ch2/AF590, Z2=Ch3/AF488, Z3=Ch1/AF647, Z4=BF |
| `hyb3_ICC_Processed001.tif` | uint16 | (1,1,5,5722,6847) | same |
| `hyb2_ICC_Processed001.tif` | uint16 | (1,1,5,5688,6845) | same |

**Color mapping** (confirmed):
- Ch1 / AF647 → Purple
- Ch2 / AF590 → Blue
- Ch3 / AF488 → Yellow
- DAPI → Gray (segmentation only)

**Key biological constraint**: Each nucleus has exactly ONE punctum + ONE color per round.
Other-channel signals = carry-over from incomplete washing → suppressed by argmax.

**Key imaging constraint**: Hyb2 & Hyb3 DNase-treated → DAPI ≈ 0 in those rounds.

---

## Module Status — B7-FOVB Results

### ✅ Module 1 — FOV Mapping (`run_module1.py`)
- DAPI template matching (not BF — BF contrast <9%, insufficient for matching)
- Result: `y0=864, x0=928`, crop `4064×4864` px, match score=0.557
- Offset from center: dy=+39, dx=−66 px (small, expected)
- Outputs: `python_results/module1/` → `crop_coords.json` + `hyb4_crop_*.npy` (5 channels)

### ✅ Module 2 — Live → Hyb4 Registration (`run_module2.py`)
- Phase correlation (`normalization=None`) on upscaled Live DAPI vs Hyb4 DAPI crop
- Result: `dy=−0.20 px, dx=−0.20 px` (sub-pixel! M1 crop was already near-perfect)
- Quality: Pearson r=0.557 (GOOD)
- Output: `python_results/module2/registration_live_hyb4.json`

### ✅ Module 3 — Hyb-to-Hyb BF Registration (`run_module3.py`)
- BF channel (Z=4) used because DAPI=0 in Hyb2/Hyb3 (DNase-treated)
- Each Hyb registered independently to Hyb4 (no chaining → no error accumulation)
- Results:
  - Hyb3 → Hyb4: `dy=+22.70, dx=+34.40 px` | Pearson r=0.228 (moderate — expected for BF)
  - Hyb2 → Hyb4: `dy=+15.70, dx=+31.60 px` | Pearson r=0.294 (moderate)
- Output: `python_results/module3/registration_hyb{2,3}_to_hyb4.json`

### ✅ Module 4 — Nuclear Segmentation (`run_module4.py`)
- CellposeSAM v4 on Hyb4 DAPI crop (MPS/GPU), `diameter=None` (auto-detect)
- Filters: area ∈ [500, 50000] px²; sequential relabeling (NOT sk_label — avoids merging)
- Result: **1230 nuclei**, mean area=2113 px², max=7434 px²
- Output: `python_results/module4/nucleus_labels.npy` + `nucleus_properties.csv`

### ✅ Module 5 — Spot Calling (`run_module5.py`)
- Method X: max pixel intensity per nucleus × channel × round
- Nucleus mask shifted by M3 (dy, dx) into HybN frame for Hyb2/Hyb3 sampling
- Result: 3687 rows (1230 nuclei × 3 rounds)
- Output: `python_results/module5/spot_intensities.csv`

### ✅ Module 6 — Barcode Decoding (`run_module6.py`)
- Argmax over 3 channels per round; `background_threshold=2000 ADU`
- Result: **1059/1230 (86.1%) fully decoded**, 50 unique barcodes
- Top barcodes: Purple-Yellow-Yellow (355), Yellow-Blue-Purple (186), Yellow-Yellow-Purple (180)
- Output: `python_results/module6/barcodes.csv`

### ✅ Enhanced QC (`run_qc_enhanced.py`)
- 8 figures in `python_results/qc/`:
  - M5: channel scatter, SNR distribution, intensity heatmap
  - M6: spot overlay (Hyb2/3/4), barcode count chart, confidence spatial map

---

## Pending — QC Review & Parameter Tuning

- [ ] Review `qc_m6_spot_overlay_Hyb4.png` — verify Purple nucleus = bright in AF647 (Red channel)
- [ ] Review `qc_m6_barcode_counts.png` — do top barcodes match expected cell pool ratios?
- [ ] Review `qc_m5_snr_distribution.png` — is `background_threshold=2000 ADU` appropriate?
- [ ] Decide: adjust `background_threshold` in `config/module6_decoding.yaml` and re-run M6 + QC
- [ ] (Future) Add Module 5 Method Y (Ronan-style adaptive threshold) for comparison

---

## Pending — Analysis & Downstream

- [ ] Map barcodes to drug/condition identity (requires barcode lookup table from experiment design)
- [ ] Merge with Live nucleus IDs (use M2 shift to map Live cell coordinates → Hyb4 frame)
- [ ] Per-condition statistics: cell count per barcode, spatial clustering analysis
- [ ] Export figures for paper / presentation

---

## Git / GitHub

- Repo: `gt-liang/20260228_FISH_Psychedelics_ImageProcessing` (private)
- Working branch: `feat/module2-live-hyb4-registration` (all 6 modules + QC committed here)
- Latest commit: `e3d02e4` — enhanced QC visualizations
- Never commit: `*.tif`, `*.npy`, `*.czi`, `IMAGES/`, `*.csv`, `*.png`
