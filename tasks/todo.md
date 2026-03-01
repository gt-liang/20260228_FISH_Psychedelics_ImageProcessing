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

---

## Next Implementation Tasks (prioritised)

### 1. QC — Add nucleus boundaries to spot overlay (quick, high value)
- In `run_qc_enhanced.py → fig_spot_overlay()`, draw nucleus boundary contours on top of the colored fill
- Use `skimage.segmentation.find_boundaries(labels)` to get 1-px boundary mask
- Draw in white/light grey so boundaries are visible even when adjacent nuclei share a color
- This makes it possible to see individual nucleus outlines when colors are the same

### 2. Spot Calling — Add Method Y (Ronan-style) for cross-comparison
- Implement in `src/module5_spot_calling/spot_caller.py` or as a separate `method_y_caller.py`
- Ronan's logic per nucleus per channel: threshold = `cell_mean + 6 × cell_std` (capped at 65000)
- Morphology: `measure.label(binary)` → `remove_small_objects(min_size=10)`
- Signal = total puncta area (not max intensity)
- Compare Method X vs Method Y: do they agree on which channel is brightest?
- Output: `spot_intensities_methodY.csv` + side-by-side comparison figure

### 3. Investigate the top-right population in M5 channel scatter
- In Hyb4 scatter (Ch1_AF647 vs Ch3_AF488): a small cluster appears in the top-right
  (both channels simultaneously high) — see discussion below
- Action: identify which nucleus_ids fall in this cluster (define as: Ch1 > X AND Ch3 > X)
- Check their spot overlay appearance, SNR, and barcode calls
- Determine cause: incomplete washing vs debris vs registration artifact

### 4. Cell ID mapping — Live → Hyb4 (Module 7 candidate)
- Use M2 shift (dy=−0.20, dx=−0.20 px) to map Live cell centroids into Hyb4 frame
- Input: Live nucleus segmentation (needs to be run on Live DAPI C00 image)
- Output: mapping table `live_nucleus_id ↔ hyb4_nucleus_id` (nearest-centroid matching)
- Allows linking Live cell morphology / timelapse data to the barcode assignment

---

## Pending — Analysis & Downstream

- [ ] Map barcodes to drug/condition identity (requires barcode lookup table from experiment design)
- [ ] Merge with Live nucleus IDs (task 4 above)
- [ ] Per-condition statistics: cell count per barcode, spatial clustering analysis
- [ ] Export figures for paper / presentation

---

## Science Discussion — Top-Right Population in M5 Channel Scatter

### Observation
In `qc_m5_channel_scatter.png` (Hyb4 panel), there is a small population of nuclei
in the top-right corner: **high Ch1_AF647 AND high Ch3_AF488 simultaneously**.
These nuclei get an argmax call (whichever is slightly higher wins) but their SNR is low.

### Possible causes (in order of likelihood)

1. **Incomplete washing / carry-over** ← most likely
   - If washing between rounds was insufficient, residual AF647 or AF488 oligos
     from a previous hybridization remain bound
   - These nuclei have genuine signal in one channel but elevated background in another
   - Typical indicator: the "extra bright" channel is consistent with carry-over direction
   - Action: check if these nuclei have specific barcodes that suggest carry-over

2. **Autofluorescence hotspots**
   - Some cells (especially stressed or dying cells) have elevated autofluorescence
     across multiple channels simultaneously
   - Autofluorescence is typically broad-spectrum → lights up all channels equally
   - Indicator: all 3 channels bright, not just 2

3. **Registration artifact (less likely)**
   - If the nucleus mask slightly overlaps a neighboring nucleus that has a different color
   - Would cause both max intensities to be high (sampling from two cells)
   - More likely at cell boundaries; check spatial position of these nuclei

4. **True biology (unlikely but possible)**
   - In theory, a cell could legitimately have two spots if barcoding failed
   - Would indicate a cell that received two oligo pools — possible but unlikely in well-controlled experiments

### Recommended handling
- Flag these nuclei: define as dual-high if both Ch1 and Ch3 > (e.g.) 15000 ADU
- Tag them as `dual_high=True` in barcodes.csv
- Do NOT automatically exclude — report them separately for biological interpretation
- Cross-check with Method Y: does Ronan-style also find signal in both channels?

---

## Git / GitHub

- Repo: `gt-liang/20260228_FISH_Psychedelics_ImageProcessing` (private)
- Working branch: `feat/module2-live-hyb4-registration` (all 6 modules + QC committed here)
- Latest commit: `e3d02e4` — enhanced QC visualizations
- Never commit: `*.tif`, `*.npy`, `*.czi`, `IMAGES/`, `*.csv`, `*.png`
