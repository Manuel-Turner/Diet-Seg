from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from common import (
    MONAI_BUNDLE_DIR,
    MONAI_MODALITY_ORDER,
    MONAI_PRED_DIR,
    NORM_DIR,
    ensure_project_dirs,
    load_selected_cases,
    log_text,
    run_command,
)


def download_bundle() -> bool:
    if (MONAI_BUNDLE_DIR / "brats_mri_segmentation").exists():
        return True

    command = [
        sys.executable,
        "-m",
        "monai.bundle",
        "download",
        "--name",
        "brats_mri_segmentation",
        "--bundle_dir",
        str(MONAI_BUNDLE_DIR),
    ]
    result = run_command(command, "02_monai_bundle_download.log")
    if result.returncode == 0:
        return True

    fallback = (
        "MONAI bundle download command failed. Fallback attempted with huggingface_hub snapshot_download.\n"
    )
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="MONAI/brats_mri_segmentation",
            local_dir=str(MONAI_BUNDLE_DIR / "brats_mri_segmentation"),
            local_dir_use_symlinks=False,
        )
        log_text("02_monai_bundle_download_fallback.log", fallback + "fallback succeeded\n")
        return True
    except Exception as exc:
        log_text(
            "02_monai_bundle_download_fallback.log",
            fallback + f"fallback failed: {type(exc).__name__}: {exc}\n",
        )
        return False


def bundle_root() -> Path:
    direct = MONAI_BUNDLE_DIR / "brats_mri_segmentation"
    nested = direct / "brats_mri_segmentation"
    if (nested / "configs" / "inference.json").exists():
        return nested
    return direct


def write_case_datalist(bundle: Path, case) -> Path:
    datalist = {
        "testing": [
            {
                "image": [case[modality] for modality in MONAI_MODALITY_ORDER],
                "label": case["seg"],
            }
        ]
    }
    path = bundle / "configs" / f"datalist_{case['case_id']}.json"
    path.write_text(json.dumps(datalist, indent=2), encoding="utf-8")
    return path


def find_monai_output(case_output: Path):
    candidates = sorted(case_output.rglob("*_seg.nii.gz")) + sorted(case_output.rglob("*.nii.gz")) + sorted(case_output.rglob("*.nii"))
    return candidates[0] if candidates else None


def normalize_monai_labelmap(src: Path, dst: Path, gt_path: Path) -> None:
    import nibabel as nib
    import numpy as np

    pred_img = nib.load(str(src))
    pred = pred_img.get_fdata()
    if pred.ndim == 4 and min(pred.shape) <= 4:
        channel_axis = int(np.argmin(pred.shape))
        pred = np.moveaxis(pred, channel_axis, 0)
        tc = pred[0] > 0.5
        wt = pred[1] > 0.5
        et = pred[2] > 0.5
        label = np.zeros(tc.shape, dtype=np.uint8)
        label[wt & ~tc] = 2
        label[tc & ~et] = 1
        label[et] = 4
    else:
        label = pred.astype(np.uint8)
        label[label == 3] = 4

    gt_img = nib.load(str(gt_path))
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(label, gt_img.affine, gt_img.header), str(dst))


def main() -> int:
    ensure_project_dirs()
    cases = load_selected_cases()
    if not download_bundle():
        print("MONAI bundle download failed. See logs/02_monai_bundle_download*.log")
        return 2

    bundle = bundle_root()
    if not (bundle / "configs" / "inference.json").exists():
        text = f"MONAI bundle inference config not found under {bundle}"
        print(text)
        log_text("02_run_monai_inference.log", text + "\n")
        return 2

    completed_cases = 0
    for case in cases:
        case_id = case["case_id"]
        case_output = MONAI_PRED_DIR / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        datalist = write_case_datalist(bundle, case)
        command = [
            sys.executable,
            "-m",
            "monai.bundle",
            "run",
            "--config_file",
            str(bundle / "configs" / "inference.json"),
            "--bundle_root",
            str(bundle),
            "--data_list_file_path",
            str(datalist),
            "--dataset_dir",
            ".",
            "--output_dir",
            str(case_output),
        ]
        result = run_command(command, f"02_monai_inference_{case_id}.log", cwd=bundle)
        if result.returncode != 0:
            print(f"MONAI inference failed for {case_id}; see logs/02_monai_inference_{case_id}.log")
            continue

        output = find_monai_output(case_output)
        if output is None:
            log_text(f"02_monai_normalize_{case_id}.log", f"No MONAI NIfTI output found in {case_output}\n")
            continue
        normalize_monai_labelmap(output, NORM_DIR / "monai" / f"{case_id}_seg.nii.gz", Path(case["seg"]))
        completed_cases += 1

    print(f"MONAI completed cases: {completed_cases}/{len(cases)}")
    return 0 if completed_cases else 2


if __name__ == "__main__":
    raise SystemExit(main())
