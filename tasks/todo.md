# FISH Psychedelics Image Processing - Task List
**Last Updated**: 2026-03-04 (Session 2)
**Status**: Full pipeline ✅ | Puncta Anchor v5 (per-ch LoG + SNR + multi-cand, 99.9%) ✅ | Gold winner highlight ✅ | Barcode plot ✅

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

## ✅ Completed Today (2026-03-03) — Session 1

### Anchor Pipeline — SNR filter + Variable circle sizes + Threshold calibration
1. **Added `_compute_snr()`** — peak-inside / mean-outside fluorescence ratio
   - inside: max pixel within blob radius (√2 × sigma) within nucleus mask
   - outside: mean of annulus (blob_r → 2×blob_r) within nucleus mask
   - Fallback: all nucleus pixels outside blob if annulus is empty
2. **Variable circle sizes** — circles now scale with blob sigma (radius = √2 × sigma, clamped 4–20 px)
3. **SNR threshold tuning attempts**:
   - `min_snr_ratio=10.0` (mean/mean formula) → 0 candidates (too strict, dilution problem)
   - Fixed to peak/mean → more appropriate but still set to `0.0` (disabled) for now
   - Lesson: calibrate log_threshold FIRST, then add SNR filter on top
4. **LoG threshold calibration**:
   | Threshold | Avg candidates | 0-candidate nuclei |
   |-----------|---------------|-------------------|
   | 0.02 | 24.54 | 0 |
   | 0.10 | 5.02 | 0 |
   | **0.20** | **1.54** | **2** ← best |
   - **Current setting: `log_threshold: 0.20`** (avg 1.54, 79% of cells have exactly 1 candidate)
5. **Per-channel per-round SNR** (9 values) computed and stored in `anchor_candidates.csv`
6. **`run_snr_histogram.py`** — standalone 3×3 SNR histogram analysis script
   - Key finding: **Hyb4/Ch1 is the ONLY bimodal channel** — confirms it as dominant Hyb4 color
   - Ch2 (mCherry) has systematically lower SNR (p90=1.88 vs Ch1 p90=3.99)

---

## ✅ Completed Today (2026-03-03) — Session 2

### Per-Nucleus Normalized Threshold (replaces absolute 300 ADU)
7. **Scientific rationale**: fixed ADU threshold ignores cell-to-cell background variation
   - A ratio threshold (peak / nucleus_bg) is cell-independent: 2.0× means "2× brighter than this cell's own baseline"
   - This is the correct framework after per-nucleus normalization
8. **`compute_nucleus_background()`** — 25th percentile of nucleus pixels, per channel × round (9 values)
   - p25 is robust to puncta bias (mean/median pulled high by bright spots)
9. **`call_color_normalized()`** — color call using `peak_in_window / nucleus_p25_background`
   - Per-channel thresholds: Ch1/Ch3 ≥ 2.0, Ch2 ≥ 1.5 (mCherry lower SNR)
10. **Results with normalized threshold** (log_threshold=0.20, n=1230):
    - Decoded: **1198/1230 (97.4%)** (up from 100% with false absolute threshold)
    - 969 nuclei have 1 candidate (79%), 125 have 2, 134 have ≥3
    - `anchor_candidates.csv` gains 9 bg columns + 3 norm_max columns

### Multi-Candidate Nuclei Analysis
11. **Key finding**: multi-candidate nuclei are NOT larger — they're smaller on average
    - n_cand=1: median area=2019px²;  n_cand≥6: median area=1404px²
    - Small nuclei (<1000px²) with many candidates → likely edge artifacts, not merged cells
    - 15-candidate nucleus has area=1719px² (normal size!) → detection algorithm issue, not segmentation
12. **Biological constraint confirmed**: each HEK cell should have exactly 1 punctum
    - Multi-candidate nuclei are suspect regardless of cause
    - Two types: small-nucleus artifacts + merged cells (n_cand=2 with normal area)

---

## ✅ Completed Today (2026-03-04)

### Group QC Analysis (`run_puncta_qc_groups.py`)
1. **Implemented group classification** — split 1230 nuclei from anchor results into:
   - single_ok: 946 (76.9%) — 1 candidate, confirmed in H3+H2 → barcode analysis ready
   - single_unconfirmed: 23 (1.9%) — 1 candidate but failed H3 or H2 threshold
   - zero: 2 (0.2%) — LoG found no blobs
   - multi: 259 (21.1%) — ≥2 blobs detected
2. **Outputs (python_results/puncta_anchor/groups/)**:
   - `group_labels.csv` — all 1230 nuclei with group label
   - `single_ok_barcodes.csv` — 25 unique barcodes, 946 cells
   - `single_ok_nucleus_ids.csv` — nucleus IDs per barcode (for manual auditing of QC crops)
   - `none_multi_candidates.csv` — 949 candidates from 284 none/multi nuclei (including subthreshold)
   - `fig_group_distribution.png` — group counts bar chart
   - `fig_single_ok_barcodes.png` — barcode distribution for single_ok
   - `fig_none_multi_signals.png` — H3 vs H2 signal scatter + failed-filter histogram
3. **Key scientific finding**: 77% (735/949) of candidates in none/multi nuclei confirm in
   BOTH rounds — LoG is detecting the same real spot at multiple scales, not noise.
   Only 60 candidates (6.3%) fully failed both rounds.
4. **25 unique barcodes** in single_ok vs 22 in Module 6 — 3 extra barcodes to investigate.

---

## ✅ Completed Today (2026-03-04) — Session 2 (Detection Algorithm Redesign)

### v3 → v4 → v5 Algorithm Evolution

**v3 (per-channel argmax)** — 94.4% decoded
- Problem: nucleus 306 — Purple selected over Yellow (true signal confirmed all 3 rounds)
- Root cause: argmax picks bright isolated pixels (noise, hot pixels) over genuine puncta

**v4 (per-channel LoG, size-first selection)** — 78.9% decoded — FAILED
- Idea: larger sigma = more photons = more likely real signal
- Root cause: Blue mCherry background blobs sigma=5.0 win the size competition
- Nucleus 306: Purple and Yellow BOTH sigma=1.9 — size doesn't differentiate them

**v5 (per-channel LoG + local SNR filter + multi-candidate winner selection)** — 99.9% ✅
- Key insight: cross-round biological consistency is primary; SNR + size are secondary
- SNR calibrated on nucleus 306: mCherry blobs SNR 1.3–2.0; true signals SNR >3.5
- min_blob_snr=2.0, log_threshold=0.20, top 3/channel, winner = most confirmed rounds → highest total signal

### QC Figure Winner Highlighting
- Gold solid thick circle (lw=2.5) = winner in ALL rows (Hyb4, Hyb3, Hyb2)
- Dashed thin (lw=1.2) = non-winner candidates (white in H4, green/red in HybN)
- Suptitle: "★ Winner: #N Barcode" clearly shown
- Table: gold background on winner row's "#" cell
- failed_qc/ folder: nucleus_0434.png (only 1 failed cell)
- barcode_distribution.png: top barcode Yellow-Yellow-Purple (390/1229, 31.7%)

### Results Summary (v5 final)
| Metric | Value |
|--------|-------|
| Total nuclei | 1230 |
| Decoded | 1229 (99.9%) |
| Avg candidates/nucleus | 7.28 |
| Unique barcodes | 20 |
| Failed QC | 1 (nucleus 434) |

---

## 🔜 NEXT SESSION — Downstream Analysis

- [ ] Compare `anchor_summary.csv` barcodes vs `module6/barcodes.csv` — agreement rate?
- [ ] Map barcodes to drug/condition identity (requires barcode lookup table)
- [ ] Per-condition statistics: cell count per barcode, spatial clustering
- [ ] Decide if anchor method should REPLACE or SUPPLEMENT Module 6 argmax approach
- [ ] Cell ID mapping: Live → Hyb4 (Module 7 candidate)

---

## Git / GitHub

- Repo: `gt-liang/20260228_FISH_Psychedelics_ImageProcessing` (private)
- Branch: **`main`** (direct push — solo research workflow)
- Never commit: `*.tif`, `*.npy`, `*.czi`, `IMAGES/`, `*.csv`, `*.png`
