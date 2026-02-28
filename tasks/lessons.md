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

### Nuclear morphology changes between Live and Hyb4 — expected, not a bug
- Live DAPI (10x): lower NA optics → larger PSF → nuclei appear blurry/larger
- Hyb4 DAPI (20x): higher NA → sharper boundaries; also fixed cells have condensed chromatin
- Template matching score ~0.55 (not 1.0) is expected and acceptable due to this difference
- Phase correlation (M2) is more robust to morphology changes than template matching
- Impact on pipeline: minimal — nuclear centroids are still in the same positions

### Scale ratio: Live 10x → Hyb 20x
- Same camera used for both modalities → pixel ratio = exactly 2:1
- Live FOV at 20x resolution: `2432 × 2 = 4864 px` (width), `2032 × 2 = 4064 px` (height)
- 10x FOV is always centered in the 3×3 tiled 20x image → center crop is correct approach
- No template matching needed unless fine-tuning is required

### Phase correlation is robust to sparse debris
- Occasional bright debris artifacts do NOT significantly affect phase correlation
- Phase correlation is a global frequency-domain operation
- Add SNR guard: if correlation peak < 0.5 → log WARNING (suggests major mismatch)
- Add shift magnitude guard: if |shift| > 50 px → log WARNING for Hyb-to-Hyb

### M2 registration direction
- Live DAPI → Hyb4 DAPI (after crop)
- Output (dy, dx) maps Live nucleus mask INTO Hyb4 coordinate space

---

## Spot Calling Reference

### Ronan's method (from 4_puncta_detection_multi-channel.py)
- Threshold: `cell_mean + 6*std` (per cell, per channel), capped at 65000
- Morphology: `measure.label(binary)` → `remove_small_objects(min_size=10)`
- Decoding: argmax by **puncta area** (not intensity)
- No-signal: `area <= 10` → Gray

### Method X (our primary approach — adapted for 1 spot/cell biology)
- Per nucleus, per channel, per round: `signal = max_pixel_intensity within mask`
- Argmax over 3 channels → color
- If max < background_threshold → None

### Method Y (Ronan-style, for validation comparison)
- Adaptive threshold → label → compare area
- Better for multi-spot scenarios, but more parameters

---

## Repo / GitHub

- Repo: `gt-liang/20260228_FISH_Psychedelics_ImageProcessing` (private)
- gh CLI: user `gt-liang`, authenticated via browser
- Branch naming: `wip/<topic>`, `feat/<topic>`, `fix/<topic>`
- Never commit: `*.tif`, `*.npy`, `*.czi`, `IMAGES/`, `*.csv`, `*.png`
