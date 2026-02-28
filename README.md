# 20260228 FISH Psychedelics Image Processing

**Lab**: Boeynaems Lab, Baylor College of Medicine
**Status**: 🔄 In Development
**Last Updated**: 2026-02-28

---

## Project Overview

Multi-round smFISH image analysis pipeline for the Psychedelics project.
Detects and quantifies RNA puncta across 3 hybridization rounds (Hyb2/3/4) in fixed cells.

### Imaging Data
| Round | Channels | Description |
|-------|----------|-------------|
| Hyb2 | BF, DAPI, Ch1, Ch2, Ch3 | Round 2 hybridization |
| Hyb3 | BF, DAPI, Ch1, Ch2, Ch3 | Round 3 hybridization |
| Hyb4 | BF, DAPI, Ch1, Ch2, Ch3 | Round 4 hybridization |

---

## Pipeline Architecture

```
Raw .tif Images (Hyb2/3/4)
         │
         ▼
  Module 1: Preprocessing
  (TIF loading, MIP, QC)
         │
         ▼
  Module 2: Multi-round Registration
  (Phase correlation, BF-guided, Hyb-to-Hyb)
         │
         ▼
  Module 3: Nuclear Segmentation
  (Cellpose-SAM on DAPI channel)
         │
         ▼
  Module 4: Puncta Detection
  (Per-cell, per-channel, per-round)
         │
         ▼
  Module 5: Visualization & Export
  (Cell maps, per-cell counts, summary CSV)
```

---

## Repo Structure

```
├── src/                          # Core pipeline code
│   ├── module1_preprocessing/
│   ├── module2_registration/
│   ├── module3_segmentation/
│   └── module4_puncta_detection/ # (in development)
├── config/                       # YAML config files per module
├── Registration_src/             # Reference: Ronan's original pipeline
├── tasks/
│   ├── todo.md                   # Current session tasks
│   └── lessons.md                # Lab knowledge base
├── docs/                         # Protocol notes, scientific rationale
├── .gitignore
├── requirements.txt
└── README.md
```

> **Note**: Raw `.tif` images and processed `.npy` arrays are excluded from git (see `.gitignore`).
> Data lives on OneDrive/local drives. Only code and configs are tracked here.

---

## Reference Pipeline

The `Registration_src/` folder contains the original FISH pipeline (Ronan O'Connell, 2025):
- `1_initial_cleanup.py` — CZI → MIP → npy
- `2_cellpose.py` — Nuclear segmentation
- `3_napari.py` — Manual QC
- `4_puncta_detection_multi-channel.py` — Puncta detection + registration

The new pipeline integrates this logic into a modular, config-driven architecture
(adapted from `20250718_VIS-Seq/`).

---

## Progress

See [tasks/todo.md](tasks/todo.md) for current task list.
See open PRs for active work branches.

---

## Environment Setup

```bash
conda activate idr-pipeline
pip install -r requirements.txt
```

Key dependencies: `cellpose`, `aicsimageio`, `scikit-image`, `napari`, `loguru`, `pyyaml`
