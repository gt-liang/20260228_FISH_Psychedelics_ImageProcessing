# FISH Psychedelics Image Processing - Task List
**Last Updated**: 2026-02-28
**Status**: Architecture confirmed ✅ — Ready to implement

---

## Project Goal
Multi-round smFISH barcode decoding for Psychedelics project.
Final output: Per-nucleus barcode — e.g., `Cell_42: [Purple(Hyb4), Blue(Hyb3), Yellow(Hyb2)]`

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
- Ch1 / AF647 = Purple
- Ch2 / AF590 = Blue
- Ch3 / AF488 = Yellow
- DAPI = Gray (segmentation only)

**Key biological constraint**: Each nucleus has exactly ONE punctum + ONE color per round.
Other-channel signals = carry-over from incomplete washing → suppressed by argmax.

**Key imaging constraint**: Hyb2 & Hyb3 DNase-treated → DAPI ≈ 0 in those rounds.

---

## Finalized Module Architecture

### Module 1 — FOV Mapping
- [ ] Load Live BF (uint16, 12-bit, 2432×2032)
- [ ] Center crop Hyb4 BF channel (Z=4) using 2:1 pixel ratio
- [ ] `crop_w = 2432 × 2 = 4864`, `crop_h = 2032 × 2 = 4064`
- [ ] `y0 = (5714 - 4064)//2`, `x0 = (6852 - 4864)//2`
- [ ] Apply same crop to all 5 Hyb4 channels
- [ ] Output: `crop_coords.json` (y0, x0, h, w)

### Module 2 — Live → Hyb4 Global Registration
- [ ] Load Live DAPI (C00, uint16) + Hyb4 DAPI crop (uint16)
- [ ] Normalize both to [0,1]
- [ ] Phase correlation → `shift_live_hyb4 = (dy, dx)`
- [ ] Guard: if correlation peak < 0.5 → log WARNING (debris/artifact)
- [ ] Output: `registration_live_hyb4.json`

### Module 3 — Hyb-to-Hyb BF Registration
- [ ] Load BF channel (Z=4, uint16) from each ICC_Processed file
- [ ] Phase correlate: Hyb4 BF crop → Hyb3 BF → Hyb2 BF (chain)
- [ ] Guard: if |shift| > 50 px → log WARNING
- [ ] Output: `registration_hyb_chain.json` → `{shift_43: [dy,dx], shift_32: [dy,dx]}`

### Module 4 — Nuclear Segmentation
- [ ] Load Hyb4 DAPI crop (Z=0, uint16)
- [ ] Run Cellpose-SAM (model: `cellpose_sam`, diameter=TBD)
- [ ] Post-process: remove border nuclei, filter area < min_size
- [ ] Output: `nucleus_mask.npy` (label array, 0=bg, 1..N=cell IDs)

### Module 5 — Spot Calling
- [ ] Load registered channels: 3 channels × 3 rounds = 9 arrays
- [ ] Per nucleus × per channel × per round: compute `max_intensity` within mask (Method X)
- [ ] Also compute Ronan-style adaptive threshold (Method Y) for comparison
- [ ] Output: `per_cell_signals.csv` (Cell_ID | round | channel | max_intensity | spot_area)

### Module 6 — Decoding
- [ ] Per cell, per round: argmax over 3 channels → color label
- [ ] If max < background_threshold → color = None
- [ ] Output: `barcode_table.csv` (Cell_ID | Hyb4_color | Hyb3_color | Hyb2_color)

---

## Completed
- [x] Explored folder structure and all existing scripts
- [x] Created GitHub repo (private): gt-liang/20260228_FISH_Psychedelics_ImageProcessing
- [x] Created .gitignore, README.md, docs/SLACK_SOP.md
- [x] Initial commit + pushed to origin/main
- [x] Confirmed ICC_Processed TIFs: uint16, Z=5 = 5 channels (not z-planes)
- [x] Confirmed channel order: Z0=DAPI, Z1=AF590, Z2=AF488, Z3=AF647, Z4=BF
- [x] Confirmed 12-bit Live data: C00=DAPI (max=1796), C03=BF (max=1629)
- [x] Confirmed color mapping: AF647=Purple, AF590=Blue, AF488=Yellow
- [x] Confirmed biology: 1 punctum/nucleus/round; argmax is correct decoding strategy
- [x] Architecture discussion complete — 6 modules confirmed

---

## Next Step
Begin implementation with **Module 1** (FOV Mapping) — simplest entry point, no external dependencies.
