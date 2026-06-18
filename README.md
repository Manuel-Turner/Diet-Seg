# Diet-Seg

This repository provides the revised Diet-Seg implementation for BraTS-style 3D brain tumor segmentation. It is organized as a reproducible code/configuration release for the Diet-Seg manuscript revision.

The code includes model definitions, dataset loader, fold-wise training scripts, teacher-derived entropy-hardness generation, hardness-weighted loss implementation, sliding-window inference, and evaluation-oriented utilities. It does not include fabricated training logs, checkpoints, or experimental metrics.

## Reproducibility Release

All controlled experiments use the same fold split, preprocessing pipeline, patch size, augmentation policy, optimizer configuration, validation rule, checkpoint selection criterion, and inference protocol. No architecture-specific patch-size tuning, test-time augmentation, or model-specific post-processing is used.

The fold-wise teacher-student protocol is leakage-controlled:

- The teacher is trained only on the training cases in each fold.
- Entropy hardness maps are generated only for training cases.
- Validation and test cases are never used for teacher training, hardness-map generation, hyperparameter selection, or model-specific tuning.

## Public Defaults

- Python: 3.10.13
- PyTorch: 2.1.2
- TorchVision: 0.16.2
- CUDA runtime: 11.8 through `pytorch-cuda=11.8`
- MONAI: 1.3.0
- NumPy: 1.24.4
- SciPy: 1.10.1
- nibabel: 5.1.0
- SimpleITK: 2.3.1
- scikit-image: 0.21.0
- einops: 0.7.0
- PyYAML: 6.0.1
- Optimizer: AdamW
- Learning rate: 1e-4
- AdamW betas: `(0.9, 0.999)`
- AdamW epsilon: `1e-8`
- Weight decay: `1e-5`
- Precision: AMP mixed precision enabled by default
- Random seed: 1234
- Patch / ROI size: `128 x 128 x 128`
- Input modalities: T1, T1ce, T2, FLAIR
- Output regions: ET, WT, TC
- Inference: sliding-window inference with ROI `128 x 128 x 128` and overlap `0.5`
- Test-time augmentation: disabled
- Model-specific post-processing: disabled

## Repository Layout

```text
configs/
  dietseg_brats2019_repro.yaml
  dietseg_brats2020_repro.yaml
datasets/
  dataset.py
legacy/
  legacy_calc_error_hardness.py
models/
  rwkv_unet.py
  rwkv_unet_edge.py
scripts/
  check_environment.py
  generate_entropy_hardness.py
  infer_sliding_window.py
  smoke_test_repro.py
  train_student.py
  train_teacher.py
utils/
  entropy_hardness.py
  repro_config.py
environment.yml
requirements.txt
```

## Environment

```bash
conda env create -f environment.yml
conda activate dietseg
python scripts/check_environment.py
```

## Data And Splits

Set `data.root` and `data.split_file` in the YAML config before running training. Split files are expected to contain fold-wise train/val/test case IDs, for example:

```json
{
  "folds": {
    "0": {
      "train": ["BraTS19_case_001"],
      "val": ["BraTS19_case_101"],
      "test": ["BraTS19_case_201"]
    }
  }
}
```

The dataset loader uses fixed modality order `t1, t1ce, t2, flair`, non-zero brain-region z-score normalization, foreground crop/sampling, and BraTS region remapping:

- ET: label `4`
- WT: labels `1, 2, 4`
- TC: labels `1, 4`

## Commands

Train the fold-wise teacher only on training cases:

```bash
python scripts/train_teacher.py --config configs/dietseg_brats2019_repro.yaml --fold 0
```

Generate teacher-derived entropy hardness maps only for training cases:

```bash
python scripts/generate_entropy_hardness.py --config configs/dietseg_brats2019_repro.yaml --fold 0
```

Train the RWKV-UNet student with EdgeNet and hardness-weighted region CE using `1+h`:

```bash
python scripts/train_student.py --config configs/dietseg_brats2019_repro.yaml --fold 0
```

Run sliding-window inference:

```bash
python scripts/infer_sliding_window.py --config configs/dietseg_brats2019_repro.yaml --fold 0 --checkpoint checkpoints/student/brats2019/fold0/best.pt
```

Run lightweight reproducibility smoke tests:

```bash
python scripts/smoke_test_repro.py
```

## Entropy Hardness

The revised manuscript uses teacher-derived entropy hardness. The script reads teacher probability maps, computes voxel-wise entropy,

```text
H(x) = -sum_c p_c(x) log(p_c(x) + eps)
```

and normalizes by `log(C)` so the hardness map lies in `[0, 1]`. The script reads the fold split file and refuses to generate maps for validation/test cases.

Legacy prediction-vs-ground-truth Dice-error hardness code has been moved to `legacy/legacy_calc_error_hardness.py` and is kept only for old comparisons. It is not used for the revised main experiments.

## Notes

The repository supplies the reproducible framework and public defaults. It does not claim full dataset-level reproducibility until local BraTS paths, fold split files, teacher probability maps, and checkpoints are provided.
