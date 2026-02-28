# Lab Knowledge Base - Lessons Learned
**Project**: FISH Psychedelics Image Processing
**Updated**: 2026-02-28

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

## Repo / GitHub

- Repo: `gt-liang/20260228_FISH_Psychedelics_ImageProcessing` (private)
- gh CLI: user `gt-liang`, authenticated via browser
- Branch naming: `wip/<topic>`, `feat/<topic>`, `fix/<topic>`
- Working branch: `feat/module2-live-hyb4-registration` (contains M2–M6 + QC)
- Never commit: `*.tif`, `*.npy`, `*.czi`, `IMAGES/`, `*.csv`, `*.png`
