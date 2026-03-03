# FISH Psychedelics Image Processing - Task List
**Last Updated**: 2026-03-02
**Status**: Full pipeline + mCherry correction + Puncta Anchor Pipeline (trial run complete) ✅

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

**Imaging order**: Hyb4 → Hyb3 → Hyb2 (Hyb4 imaged first; mCherry bleaches ~19.8% by Hyb2).

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
- **2026-03-01**: Added `_apply_cross_round_correction()`:
  - `_xr_bg = min(Ch_corr_Hyb2, Ch_corr_Hyb3, Ch_corr_Hyb4)` per nucleus per channel
  - `_xr = max(Ch_corr − xr_bg, 0)` — removes cross-round carry-over and persistent fluorescence
  - All 3 channels have `_xr_bg` and `_xr` columns in output for traceability
- Result: 3687 rows (1230 nuclei × 3 rounds), columns include raw / `_bg` / `_corr` / `_xr_bg` / `_xr`
- Output: `python_results/module5/spot_intensities.csv`

### ✅ Module 6 — Barcode Decoding (`run_module6.py`)
- **Final correction mode (2026-03-01)**:
  - Ch1_AF647: `_corr` (spatial bg subtraction only — no persistent protein)
  - Ch2_AF590: `_xr` (cross-round mCherry correction)
  - Ch3_AF488: `_corr` (spatial bg subtraction only — no persistent protein)
  - Threshold: 500 ADU
- Result: **1216/1230 (98.9%) fully decoded**, 14 None cells (1.1%), 22 unique barcodes
- None population: biologically acknowledged — unlabeled cells, failed hybridization, or cells outside barcode library
- Top barcodes: Purple-Yellow-Yellow (384), Yellow-Yellow-Purple (208), Yellow-Blue-Purple (165)
- Output: `python_results/module6/barcodes.csv` (1230 rows, includes `decoded_ok` flag)

### ✅ Enhanced QC (`run_qc_enhanced.py`)
- 8 figures in `python_results/qc/`:
  - M5: channel scatter (dual-high highlighted), SNR distribution, intensity heatmap, dual-high spatial map
  - M6: spot overlay (Hyb2/3/4), barcode count chart, confidence spatial map

### ✅ Background Subtraction QC (`run_qc_bg_comparison.py`) — NEW 2026-03-01
- 5 figures in `python_results/qc/`:
  - `qc_bg_scatter_comparison.png` — raw vs corrected channel scatter
  - `qc_bg_background_magnitude.png` — background pedestal analysis per channel per round
  - `qc_bg_call_changes.png` — which cells changed call + transition matrix
  - `qc_bg_snr_improvement.png` — SNR before vs after (aggregated per round)
  - `qc_bg_snr_by_color.png` — SNR per color × per round 3×3 grid (NEW)

---

## ✅ Completed Today (2026-03-01)

### mCherry Cross-Round Correction Pipeline
1. **Diagnosed Blue contamination** — Ch2_AF590 (Blue) had false calls in Hyb2/Hyb4 because mCherry
   fluorescent protein was not fully eliminated and persists across all imaging rounds
2. **Confirmed imaging order**: Hyb4 → Hyb3 → Hyb2; mCherry bleaches 19.8% over this sequence
3. **Created `run_mcherry_correction.py`** — standalone exploration script using mean(Ch2_Hyb4, Ch2_Hyb2)
   as per-nucleus mCherry baseline for Hyb3 subtraction
4. **Integrated into Module 5** — `_apply_cross_round_correction()` computes `_xr_bg` and `_xr` columns
   for ALL 3 channels (for traceability and future flexibility)
5. **Tested all-channel _xr in Module 6** — caused 22.8% None rate (281 cells) due to over-correction:
   Ch1/Ch3 cross-round carry-over (~800 ADU) overlaps with borderline Purple/Yellow signals
6. **Diagnosed None cell subtypes**:
   - Type A (all rounds low signal): genuinely unlabeled cells → correct to call None
   - Type B (2 strong rounds + 1 zeroed by _xr): over-correction artifact → should decode
7. **Final decision**: Ch2_AF590 → `_xr`, Ch1/Ch3 → `_corr` (reverted to channel-specific approach)
8. **Re-exported `barcodes.csv`**: 1216 decoded (98.9%), 14 None (1.1%) — biologically accepted
9. **Regenerated all QC figures** (both `run_qc_enhanced.py` and `run_qc_bg_comparison.py`)

---

## ✅ Completed Today (2026-03-02)

### Puncta Detection Cross-Comparison (6 Methods)
1. **Implemented `src/puncta_comparison/`** — `methods.py` + `comparator.py` + `__init__.py`
   - 6 methods: X (max pixel), Y (adaptive thresh), Z (LoG), W (DoG), T (TrackPy), P (peak_local_max)
   - Each method: `(crop, mask, **params) → float signal` for one nucleus × channel × round
2. **Config**: `config/puncta_comparison.yaml` (all parameters, image paths, output dir)
3. **Entry point**: `run_puncta_comparison.py` → QC figures + comparison_table.csv
4. **Results**:
   | Method | Decoded% | SNR median |
   |--------|----------|-----------|
   | X (max pixel)   | 98.9% | 10.7 |
   | Y (adaptive thr)| 95.7% | 30.6 |
   | Z (LoG)         | 99.7% |  8.4 |
   | W (DoG)         | 99.7% |  8.6 |
   | T (TrackPy)     | 97.0% | 11.6 |
   | P (peak_max)    | 99.4% |  9.2 |
   - Z↔W highest pairwise agreement (93.3%); X↔Y lowest (74.1%)
5. **Disagreement QC** — `run_puncta_qc_disagreement.py`: 415 PNGs for nuclei where ≥2 methods disagree
   - v2: color-tinted channel panels + merged RGB composite + centroid-based crop
   - Layout: 3 rounds × 4 columns (Ch1/Ch2/Ch3/Merged) + method call table

### Puncta Anchor Validation Pipeline (NEW)
6. **Biological rationale**: mRNA position fixed between rounds → use Hyb4 positions to anchor cross-round validation
7. **Implemented `run_puncta_anchor.py`** — standalone pipeline:
   - Step 1: LoG on max(Ch1, Ch2, Ch3) per nucleus in Hyb4 → candidate positions
   - Step 2: Cross-reference each position in Hyb3/Hyb2 (search_radius=3px window)
   - Step 3: Color = argmax per round; confirmed = max ≥ 300 ADU
   - Step 4: QC figure per nucleus (3×4 grid + puncta circles + candidate table)
8. **Config**: `config/puncta_anchor.yaml`
9. **Trial run result** (log_threshold=0.02): avg 24.5 candidates/nucleus — LoG too sensitive
   - Caused 100% decoded rate (false positive: every nucleus has at least one noise blob)
   - Root cause: log_threshold=0.02 detects noise as blobs; needs tuning (try 0.10–0.15)
10. **Layout v2 fixes** — fixed canvas + table cap:
    - `_crop_canvas()`: fixed 2r×2r zero-padded canvas → consistent crop size across rounds
    - Table capped at 10 rows; figure height fixed at 11 inches (table can't cover image panels)
    - Circle positioning updated to canvas coordinate system

---

## 🔜 Next Session — Anchor Pipeline Tuning

### Priority review (tomorrow)
- [ ] Open `python_results/puncta_anchor/nucleus_crops/` — inspect 5–10 single-candidate nuclei (n_candidates=1)
  - Confirm white circle lands on the real smFISH spot in Hyb4
  - Confirm green circles appear in correct Hyb3/Hyb2 positions
- [ ] Open nuclei with n_candidates ≥ 3 — count false detections vs real spots
- [ ] Decide threshold adjustment: try `log_threshold: 0.10` then 0.15 if still too many

### Tuning approach
```yaml
# config/puncta_anchor.yaml — adjust these:
detection:
  log_threshold: 0.10   # raise from 0.02 → expect ~1–3 candidates/nucleus
validation:
  min_signal: 300       # may need to tune after seeing confirmed rates
```
- Goal: avg candidates/nucleus ≈ 1–3; confirmed_h3 AND confirmed_h2 rate ≈ 90%+

### After threshold tuning
- [ ] Compare `anchor_summary.csv` decoded barcode vs `module6/barcodes.csv` — how many match?
- [ ] Nuclei where anchor disagrees with Module 6: inspect QC crops for ground truth
- [ ] Decide if anchor method should REPLACE or SUPPLEMENT Module 6 argmax approach
- [ ] Cell ID mapping: Live → Hyb4 (Module 7 candidate)
- [ ] Map barcodes to drug/condition identity (requires barcode lookup table)
- [ ] Per-condition statistics: cell count per barcode, spatial clustering

---

## Git / GitHub

- Repo: `gt-liang/20260228_FISH_Psychedelics_ImageProcessing` (private)
- Branch: **`main`** (direct push — solo research workflow)
- Never commit: `*.tif`, `*.npy`, `*.czi`, `IMAGES/`, `*.csv`, `*.png`
