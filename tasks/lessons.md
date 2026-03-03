# Lab Knowledge Base - Lessons Learned
**Project**: FISH Psychedelics Image Processing
**Updated**: 2026-03-01

---

## Data Format Lessons

### ICC_Processed TIF files — Z is NOT z-planes
- Files: `hyb2/3/4_ICC_Processed001.tif` (372–374 MB each)
- Shape: `(T=1, C=1, Z=5, Y, X)` — **Z=5 encodes 5 channels**, NOT focal planes
- This is a FIJI export quirk: Thunder + MIP → exported as RAW ImageJ TIFF
- Channel order (Z-index): Z0=DAPI, Z1=Ch2/AF590, Z2=Ch3/AF488, Z3=Ch1/AF647, Z4=BF
- Access with: `img.get_image_data("TCZYX")[0, 0, z_idx, :, :]`

### Per-channel TIF files (111–112 MB) are uint8 — DO NOT USE for analysis
- `Hyb4_BF.tif`, `Hyb4_Ch1.tif`, etc. — all uint8 (LUT-scaled display exports)
- True 16-bit data ONLY in ICC_Processed files

### Live images: use C00/C03 files (12-bit uint16), NOT the old BF/DAPI files
- Old: `B7-FOVB--t1118_DAPI.tif` → uint8 (display only)
- New: `B7-FOVB--t1118--C00.tif` → uint16, max≈1796 (12-bit ADC) ← USE THIS
- New: `B7-FOVB--t1118--C03.tif` → uint16, max≈1629 (12-bit ADC) ← USE THIS
- C00 = DAPI, C03 = Brightfield (final frame t118)

### Python environment
- Always use: `/Users/guo-tengliang/miniconda3/envs/idr-pipeline/bin/python`
- Do NOT use `conda run -n idr-pipeline` — routes to wrong env on this machine

---

## Biology Lessons

### DNase treatment: Hyb2 and Hyb3 have no usable DAPI
- Hyb2 and Hyb3 are DNase-treated → DAPI signal ≈ 0
- Only Hyb4 has reliable DAPI for nuclear segmentation
- Consequence: Hyb-to-Hyb registration MUST use BF channel (Z=4), not DAPI

### One punctum per nucleus per round — critical for decoding
- Each cell pool has exactly ONE spot in ONE channel per hybridization round
- Other-channel signals = carry-over from incomplete washing
- Decoding strategy: argmax(channel intensity within nucleus) per round
- If max < background_threshold → label as None (no signal)

### Color mapping (Psychedelics project)
- AF647 (Ch1) → Purple
- AF590 (Ch2) → Blue
- AF488 (Ch3) → Yellow
- DAPI → Gray (segmentation channel, not decoded)

### Imaging order and mCherry bleaching
- Imaging order: **Hyb4 → Hyb3 → Hyb2** (Hyb4 is imaged FIRST)
- mCherry fluorescent protein was not fully eliminated before imaging
- Observed bleaching: ~19.8% from Hyb4 to Hyb2 (mCherry signal decreases over time)
- mCherry appears in Ch2_AF590 (Blue channel) in all 3 rounds → false Blue calls
- Cross-round correction uses mean(Ch2_Hyb4, Ch2_Hyb2) as per-nucleus mCherry baseline
  because Hyb4 and Hyb2 are the imaging endpoints and their mean linearly interpolates
  the mCherry signal at the Hyb3 timepoint

### None population is biologically expected
- Not all cells in the FOV will have been successfully labeled with a barcode
- Cells outside the barcode library (contaminating cells, unlabeled cells) → legitimately no signal
- Accept None calls as a valid biological outcome; do NOT try to force 100% decoding
- In barcodes.csv: retain all cells with `decoded_ok=False`; downstream analysis filters on `decoded_ok=True`

---

## Registration Lessons

### BF template matching fails — use DAPI for M1
- BF images at 10x and 20x have <9% contrast (nearly uniform) → template matching score ~0.44 at corner of search space (suspicious)
- DAPI nuclear staining has sparse high-contrast point patterns → score ~0.56, correct position
- Rule: always use DAPI for M1 template matching, not BF

### Nuclear morphology changes between Live and Hyb4 — expected, not a bug
- Live DAPI (10x): lower NA optics → larger PSF → nuclei appear blurry/larger
- Hyb4 DAPI (20x): higher NA → sharper boundaries; fixed cells have condensed chromatin
- Template matching score ~0.55 (not 1.0) is expected and acceptable
- Phase correlation (M2) is more robust to morphology changes than template matching

### Scale ratio: Live 10x → Hyb 20x
- Same camera used for both modalities → pixel ratio = exactly 2:1
- Live FOV at 20x resolution: `2432 × 2 = 4864 px` (width), `2032 × 2 = 4064 px` (height)
- 10x FOV is approximately centered in the 3×3 tiled 20x image (small offset expected)

### Phase correlation: use normalization=None for sparse images
- `normalization="phase"` divides Fourier amplitudes by their magnitude
- For sparse DAPI images (mostly black background), many bins have amplitude ≈ 0 → division by zero → error metric returns 1.0 (meaningless)
- Fix: always use `normalization=None` (amplitude-weighted cross-correlation, numerically stable)
- Quality metric: compute Pearson r AFTER applying the estimated shift (independent, interpretable)
  - r > 0.3 = GOOD; r < 0.1 = WARNING

### Pearson r interpretation per modality
- DAPI (M2): r ~ 0.55 is GOOD (morphology difference between Live/Hyb4 limits r)
- BF (M3): r ~ 0.22–0.29 is MODERATE but expected (BF has inherently low contrast)
  - The shift values themselves are more reliable than Pearson r for BF

### Hyb-to-Hyb BF registration: register each round independently to Hyb4
- Do NOT chain (Hyb2→Hyb3→Hyb4) — error accumulates
- Register Hyb2→Hyb4 and Hyb3→Hyb4 separately (two independent JSON outputs)
- Observed shifts B7-FOVB: Hyb3=41 px, Hyb2=35 px — within expected range for buffer exchange

### M3 coordinate convention (critical for M5)
- M3 output (dy, dx) = "shift HybN image by (dy, dx) to align to Hyb4"
- In M5 to sample HybN from Hyb4 mask: shift mask by (-dy, -dx)
- To display HybN fluorescence in Hyb4 frame: ndimage_shift(HybN_img, shift=(dy, dx))

---

## Segmentation Lessons

### CellposeSAM v4 API changes
- `model_type` argument is ignored in v4 — only one universal model
- `diameter=0` causes ZeroDivisionError (image_scaling = 30/diameter)
- Use `diameter=None` for auto-detection in v4
- `channels` argument deprecated in v4 — omit it entirely

### sk_label after filtering merges adjacent regions — use sequential relabeling instead
- After zeroing out filtered nuclei, calling `sk_label(filtered > 0)` can merge
  previously-separated regions that now share a border → wrong area measurements
- Correct approach: compact the existing label set without touching boundaries:
  ```python
  unique_labels = np.unique(filtered); unique_labels = unique_labels[unique_labels > 0]
  for new_id, old_id in enumerate(unique_labels, start=1):
      filtered_relabeled[filtered == old_id] = new_id
  ```

---

## Spot Calling & Decoding Lessons

### Method X — max pixel intensity (primary approach)
- Per nucleus, per channel, per round: `signal = max_pixel_intensity within mask`
- Argmax over 3 channels → color label
- If max < background_threshold → None
- Background threshold default: 2000 ADU (adjust based on M5 QC violin plot)
- Advantages: simple, robust, no spot morphology assumptions

### Coordinate shift for HybN mask sampling (M5)
- Nucleus labels are in Hyb4 space
- To sample from HybN image: apply ndimage_shift(labels, shift=(-dy, -dx), order=0)
- order=0 (nearest-neighbour) preserves integer label values — critical!
- order=1 or higher would interpolate between labels → wrong nucleus IDs

### np.maximum.at for vectorised max-per-label
- Faster than looping over nuclei for large images:
  ```python
  max_vals = np.zeros(n_labels + 1)
  np.maximum.at(max_vals, flat_labels[mask], flat_arr[mask])
  ```

---

## Cross-Round Correction Lessons (2026-03-01)

### Apply _xr correction only to Ch2_AF590 — NOT to Ch1 or Ch3
- **Problem**: applying cross-round minimum subtraction (_xr) to all 3 channels caused 22.8% None rate (281 cells)
- **Root cause**: Ch1_AF647 and Ch3_AF488 have cross-round carry-over at ~800 ADU median,
  which is the SAME order of magnitude as borderline Purple/Yellow FISH signals (~500–2000 ADU).
  Subtracting this carry-over zeroes out borderline cells → false None calls (Type B over-correction)
- **mCherry is different**: Ch2_AF590 carry-over is driven by mCherry persistent fluorescence,
  not by FISH carry-over. mCherry is present uniformly across all rounds → min subtraction
  correctly removes it without touching genuine Blue FISH signal peaks
- **Rule**: `_xr` for Ch2 only; `_corr` (spatial bg) for Ch1 and Ch3

### Two types of None cells — must distinguish before accepting
- **Type A** (all rounds low signal, all _xr < 500 ADU): genuinely unlabeled cells → correct None
- **Type B** (2 strong rounds + 1 collapsed to 0 by _xr): over-correction artifact → wrong None
- Type B cells can be identified by: max_xr in "None" round = 0 while other rounds >> threshold
- If Type B cells are numerous → reduce aggressiveness of correction (apply to fewer channels)

### Cross-round _xr correction formula
```python
# Per nucleus, per channel:
xr_bg = min(Ch_corr_Hyb2, Ch_corr_Hyb3, Ch_corr_Hyb4)  # minimum across rounds = carry-over baseline
xr    = max(Ch_corr - xr_bg, 0)                          # clip to 0 (no negative intensities)
```
- The minimum across rounds is the best per-nucleus estimate of cross-round carry-over
- Assumes each cell has genuine FISH signal in at most ONE round per channel (true for this experiment)

### mCherry baseline for Hyb3 correction
- Imaging order Hyb4 → Hyb3 → Hyb2 with 19.8% bleaching
- mean(Ch2_Hyb4, Ch2_Hyb2) linearly interpolates mCherry at the Hyb3 timepoint
- This is equivalent to the _xr approach applied only to Ch2

### Threshold selection after _xr correction
- After _xr correction, threshold = 500 ADU is appropriate:
  - Ch2_xr in non-signal rounds (mCherry baseline subtracted) collapses to ~0-200 ADU
  - Genuine Blue FISH peaks remain >> 500 ADU after subtraction
  - Ch1/Ch3 use _corr with same 500 ADU threshold — spatial bg subtraction brings noise to ~200 ADU

---

## Method Y (Ronan-style) Lessons

### Use regionprops bounding boxes for per-nucleus crops — avoids full-image scan
- `skimage.measure.regionprops(labels)` returns bounding boxes for each nucleus
- Crop to `labels[r0:r1, c0:c1]` before threshold/binary/label → ~50×50 px instead of 5714×6852
- Makes the per-nucleus loop fast: 1230 nuclei × 3 channels × 3 rounds = 3.4 s total

### Zero-inflation inflates X–Y disagreement — report carefully
- When Method Y area=0 in ALL channels, `idxmax()` returns the first column (Ch1_AF647 by default)
- This is not a real call — the nucleus simply has no detectable punctum
- Disagreement between X and Y is often driven by these "no punctum" cases, not true mis-calls
- Recommendation: compute agreement only on nuclei where Method Y has area > 0 in ≥1 channel

### Puncta area ~14 px² is biologically expected for smFISH at 20× (0.65 µm/px)
- Single diffraction-limited spot ≈ 300 nm diameter → ~0.5 µm × 0.5 µm → ~0.25 µm²
- At 0.65 µm/px: 0.25 µm² / 0.4225 µm²/px ≈ 0.6 px — BUT oversampled by PSF → ~14 px²
- Areas > 200 px² likely indicate aggregates, debris, or two overlapping cells

### Dual-high population (both Ch1 AND Ch3 > p80) — expected interpretation
- Hyb4 has the largest dual-high population (n=121) because it is the last round with the
  most cumulative carry-over from Hyb2 and Hyb3 residual oligos
- Cross-check with Method Y: if both channels show area > 0 → real carry-over signal;
  if both areas = 0 → elevated background/autofluorescence, not true puncta

## QC & Visualization Lessons

### Spot overlay: vectorise mask coloring with LUT
- Looping over 1230 nuclei to fill RGBA mask: ~5 min per round
- Vectorised LUT approach (H×W fancy indexing): ~2 s per round
  ```python
  lut = np.zeros((max_id + 1, 4), dtype=np.float32)
  for nid, hex_color in nid_to_color.items():
      lut[nid] = [r, g, b, 0.55]
  color_img = lut[labels]   # (H, W) int → (H, W, 4) float
  ```

### Spot overlay composite channel mapping
- For the fluorescence background composite in QC figures:
  - R = Ch1_AF647 (Purple channel)
  - G = Ch3_AF488 (Yellow channel)
  - B = Ch2_AF590 (Blue channel)
- Validation check: Purple nucleus fill should correspond to bright Red in composite

### Pandas groupby with include_groups=False drops grouping column
- In pandas >= 2.2, `groupby(...).apply(..., include_groups=False)` excludes the
  grouping column from the result → KeyError if you try to access it later
- Fix: use `groupby(..., group_keys=False).apply(...)` without include_groups

---

## Puncta Comparison Lessons (2026-03-02)

### 6-method comparison summary
- All 6 methods agree on easy nuclei (strong, well-isolated signal)
- Z (LoG) and W (DoG) have highest decoded rate (99.7%) but lower SNR than Y
- Y (adaptive threshold) has highest SNR (30.6) but lowest decoded rate (95.7%) — often returns 0 on dim cells
- Method X (current primary) balanced: 98.9% decoded, SNR 10.7
- Z↔W pairwise agreement 93.3% (most similar); X↔Y only 74.1% (most different)
- Disagreement QC: 415/1230 nuclei show ≥2 conflicting calls → manual review needed

### matplotlib table: cellColours must NOT include header row
- `ax.table(colLabels=..., cellColours=cell_colors)`: `cell_colors` must have ONLY data rows
- Adding a header row to cell_colors causes ValueError (shape mismatch)
- Always: `cell_colors = []` then `cell_colors.append(row)` for data rows only

### Disagreement QC visualization
- Use centroid-based crop (radius = sqrt(area_px/π) + 25px) — NOT bbox-based
  - Bbox from shifted labels can be partially out-of-frame → nucleus cut off
  - Centroid is stable and always centers the nucleus
- Color-tint channel panels by fluorophore color (not grayscale) for biological intuition
- Merged RGB composite: R=Ch1(Purple), G=Ch3(Yellow), B=Ch2(Blue)
- Output to `python_results/puncta_comparison/disagreement_crops/`, sort descending by ndiff

---

## Puncta Anchor Pipeline Lessons (2026-03-02)

### LoG threshold 0.02 is far too sensitive for smFISH
- `blob_log(norm, threshold=0.02)` on nucleus max-projection: avg 24.5 blobs/nucleus
- For smFISH, expect 1–3 puncta per nucleus (one RNA per cell)
- **Start at log_threshold=0.10**; increase to 0.15–0.20 if still too many detections
- Rule of thumb: threshold ~ 5–10% of normalized max intensity

### Fixed-size canvas is essential for multi-round visualization
- Problem: cropping `image[max(0, cy-r):min(H, cy+r)]` gives different sizes when nucleus
  is near image edge → imshow stretches to fill panel → inconsistent apparent zoom
- Solution: always create a `(2r × 2r)` canvas, zero-pad where out of bounds, paste actual crop
  ```python
  canvas = np.zeros((2*r, 2*r))
  r0_req = cy - r  # may be negative
  r0_act = max(0, r0_req)
  canvas[r0_act - r0_req: ...] = image[r0_act: ...]
  ```
- Canvas top-left corner in global coords = (r0_req, c0_req), potentially negative
- Circle positions: `x_canvas = x_global - c0_req` (works correctly even when c0_req < 0)

### Table height must be capped in matplotlib gridspec
- `height_ratios=[1,1,1, 0.3 + 0.28*n]` with n=24 → table takes 70% of figure height
- Fix: cap height_ratios table value at ~1.1; cap displayed rows at `MAX_TABLE_ROWS=10`
- Add "… N more not shown" text note below table when candidates are truncated

### Puncta anchor coordinate formula (confirmed)
- Registration (dy, dx) from M3 = "shift HybN by (dy, dx) to align to Hyb4"
- `shift_labels` applies `ndimage_shift(labels, shift=(-dy, -dx))`
- A pixel at (y, x) in Hyb4 → at (y − dy, x − dx) in HybN image
- Same formula for both display crops AND puncta position cross-referencing

### Search radius for cross-round position matching
- SEARCH_RADIUS=3px accounts for residual sub-pixel registration error after Phase Correlation
- If registration quality is good (Pearson r > 0.3), 3px is sufficient
- If many false negatives (confirmed_h3/h2 rate < 50%), increase to 5px

---

## Repo / GitHub

- Repo: `gt-liang/20260228_FISH_Psychedelics_ImageProcessing` (private)
- gh CLI: user `gt-liang`, authenticated via browser
- Branch: **`main`** (direct push — solo research, no feature branches needed)
- Never commit: `*.tif`, `*.npy`, `*.czi`, `IMAGES/`, `*.csv`, `*.png`
