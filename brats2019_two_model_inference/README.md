# BraTS2019 Two-Model Inference Demo

This project runs two public pretrained brain tumor segmentation models on a small labeled subset of BraTS2019 training cases, with no training or fine-tuning:

- MONAI `brats_mri_segmentation` bundle
- KAIST BraTS21 nnU-Net Docker solution, `rixez/brats21nnunet`

The goal is an inference-only comparison that can produce original MRI/ground-truth/prediction visualizations, Dice metrics for WT/TC/ET, and a short report identifying one strong case and one poor case. The scripts never fabricate metrics or figures; outputs are generated only when data and model inference actually run.

## Hardware And Software Assumptions

- Windows workstation with NVIDIA RTX 4090 or similar CUDA GPU
- Python 3.10 recommended
- CUDA-capable PyTorch for MONAI inference
- Docker Desktop plus NVIDIA Container Toolkit for KAIST Docker inference
- BraTS2019 training labels, either from official CBICA/IPP access or the Kaggle mirror

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Access

Use BraTS2019 training data only, because validation/test labels are not available. Official BraTS2019 data access may require CBICA IPP registration:

- Official page: https://www.med.upenn.edu/cbica/brats2019/data.html
- Kaggle mirror: https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019

Expected training layout usually contains `HGG` and `LGG` folders. Each selected case must include:

```text
*_flair.nii.gz
*_t1.nii.gz
*_t1ce.nii.gz
*_t2.nii.gz
*_seg.nii.gz
```

The default subset target is 6 cases, preferably 3 HGG and 3 LGG. If fewer complete cases are available, the data script selects as many as possible and logs the reason.

## One-Command Workflow

PowerShell:

```powershell
.\scripts\run_all.ps1
```

Bash:

```bash
bash scripts/run_all.sh
```

## Step-By-Step Commands

Check environment:

```bash
python scripts/00_check_env.py
```

Prepare data from a local BraTS2019 path:

```bash
python scripts/01_prepare_data.py --brats_root "D:/datasets/BraTS2019/MICCAI_BraTS_2019_Data_Training" --num_cases 6
```

Prepare data automatically from `data/raw/`, falling back to Kaggle:

```bash
python scripts/01_prepare_data.py --auto --num_cases 6
```

Run MONAI inference:

```bash
python scripts/02_run_monai_inference.py
```

Run KAIST Docker inference:

```bash
python scripts/03_run_kaist_inference.py
```

Normalize predictions, evaluate Dice, visualize, and report:

```bash
python scripts/04_normalize_predictions.py
python scripts/05_evaluate.py
python scripts/06_visualize_success_failure.py
python scripts/07_make_report.py
```

## Outputs

```text
outputs/metrics/selected_cases.csv
outputs/metrics/dice_by_case.csv
outputs/metrics/summary_by_model.csv
outputs/metrics/success_failure_cases.json
outputs/figures/success_case_<case_id>.png
outputs/figures/failure_case_<case_id>.png
outputs/figures/all_cases/<case_id>.png
outputs/report.md
```

These files are created only after the corresponding inference/evaluation steps have enough real inputs.

## Model Notes

MONAI bundle:

- Source: https://huggingface.co/MONAI/brats_mri_segmentation
- Input order: T1c, T1, T2, FLAIR
- Output regions: TC, WT, ET
- The scripts convert region outputs to BraTS label convention.

KAIST:

- Source: https://github.com/rixez/Brats21_KAIST_MRI_Lab
- Docker image: `rixez/brats21nnunet`
- Required input naming: `BraTS2021_ID_<contrast>.nii.gz`, where contrast is `flair`, `t1`, `t1ce`, `t2`
- The scripts copy BraTS2019 cases into `data/kaist_input/` using compatible safe names.

## Label Convention

```text
0 = background
1 = necrotic and non-enhancing tumor core, NCR/NET
2 = peritumoral edema, ED
4 = enhancing tumor, ET
```

Evaluation regions:

- WT: labels `{1, 2, 4}`
- TC: labels `{1, 4}`
- ET: labels `{4}`

## Troubleshooting

Missing Kaggle credentials:

- Create a Kaggle account.
- Go to Account settings.
- Create API token.
- Put `kaggle.json` in `~/.kaggle/kaggle.json` on Linux/WSL or `C:/Users/<USER>/.kaggle/kaggle.json` on Windows.
- Run `python scripts/01_prepare_data.py --auto --num_cases 6` again.

Docker unavailable:

- KAIST inference requires Docker Desktop and NVIDIA Container Toolkit.
- `scripts/00_check_env.py` prints Docker availability and does not crash if Docker is missing.

CUDA unavailable:

- MONAI may run slowly or fail depending on the installed PyTorch build.
- Install a CUDA-enabled PyTorch build matching your NVIDIA driver.

MONAI bundle download failure:

- The script first tries `python -m monai.bundle download --name brats_mri_segmentation --bundle_dir models/monai_bundle`.
- If that fails, it tries `huggingface_hub.snapshot_download`.
- See `logs/02_monai_bundle_download*.log` for the exact error.

KAIST output label 3 vs BraTS label 4:

- `scripts/03_run_kaist_inference.py` and `scripts/04_normalize_predictions.py` convert label `3` to label `4`.

Prediction shape mismatch:

- `scripts/04_normalize_predictions.py` resamples predictions to the ground-truth grid using nearest-neighbor interpolation only and preserves the ground-truth affine/header.

## Reproducibility Boundary

This is an inference-only demo. It does not train, fine-tune, tune hyperparameters, or claim benchmark-level reproducibility. Generated reports state the exact selected cases and models actually run.
