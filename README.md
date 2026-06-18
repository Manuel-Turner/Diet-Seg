# Diet-Seg

Diet-Seg is a compact reference implementation for the key components used in the Diet-Seg manuscript:

- RWKV-UNet segmentation backbone
- EdgeNet boundary-enhancement branch
- BraTS-style four-modality preprocessing demo

This repository intentionally keeps only the core model architecture and a minimal preprocessing example. It does not include full training pipelines, fold-management code, experiment logs, checkpoints, or reported metrics.

## Repository Layout

```text
models/
  edge_detection.py       # EdgeNet boundary enhancement module
  rwkv_unet.py            # RWKV-UNet backbone
  rwkv_unet_edge.py       # RWKV-UNet with EdgeNet branch
datasets/
  dataset.py              # Minimal BraTS preprocessing helpers
  preprocess_demo.py      # Runnable preprocessing demo
requirements.txt
```

## Core Model

The main model entry point is `models/rwkv_unet_edge.py`.

```python
from models.rwkv_unet_edge import RWKV_UNet

model = RWKV_UNet(
    in_chans=4,    # T1, T1ce, T2, FLAIR
    out_chans=3,   # ET, WT, TC
    img_size=128,
)
```

The implementation combines:

- a 2D RWKV-UNet style encoder-decoder backbone
- an EdgeNet branch from `models/edge_detection.py`
- edge-feature concatenation before the main encoder

## Preprocessing Demo

The preprocessing demo follows the BraTS-style inputs used by Diet-Seg:

- modality order: `t1`, `t1ce`, `t2`, `flair`
- non-zero brain-region z-score normalization
- center crop / zero pad to `128 x 128 x 128`
- region remapping:
  - ET: label `4`
  - WT: labels `1, 2, 4`
  - TC: labels `1, 4`

Run the demo with synthetic data:

```bash
python datasets/preprocess_demo.py
```

Run the same preprocessing on one local BraTS-style case:

```bash
python datasets/preprocess_demo.py --case_dir /path/to/case --case_id BraTS19_2013_2_1
```

Expected output shapes:

```text
image: [4, 128, 128, 128]
label: [3, 128, 128, 128]  # ET, WT, TC
```

## Install

```bash
pip install -r requirements.txt
```

`nibabel` is only needed when reading real NIfTI files. The synthetic preprocessing demo only requires NumPy.

## Scope

This repository is a lightweight code reference. Full training, teacher-student hardness generation, sliding-window inference, and experiment reproduction scripts are deliberately omitted to keep the repository focused on the core architecture and preprocessing logic.
