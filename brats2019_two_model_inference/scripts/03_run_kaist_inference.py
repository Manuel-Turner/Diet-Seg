from __future__ import annotations

import shutil
from pathlib import Path

from common import (
    KAIST_INPUT_DIR,
    KAIST_PRED_DIR,
    NORM_DIR,
    ensure_project_dirs,
    executable_available,
    load_selected_cases,
    log_text,
    run_command,
)


IMAGE = "rixez/brats21nnunet"


def prepare_kaist_input(cases):
    if KAIST_INPUT_DIR.exists():
        for path in KAIST_INPUT_DIR.glob("*.nii*"):
            path.unlink()
    KAIST_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for case in cases:
        case_id = case["case_id"]
        for contrast in ("flair", "t1", "t1ce", "t2"):
            dst = KAIST_INPUT_DIR / f"BraTS2021_{case_id}_{contrast}.nii.gz"
            shutil.copy2(case[contrast], dst)


def find_kaist_output(case_id: str):
    candidates = []
    for path in KAIST_PRED_DIR.rglob("*.nii*"):
        if case_id in path.name:
            candidates.append(path)
    return sorted(candidates)[0] if candidates else None


def normalize_labelmap(src: Path, dst: Path, gt_path: Path):
    import nibabel as nib
    import numpy as np

    pred = nib.load(str(src)).get_fdata().astype(np.uint8)
    pred[pred == 3] = 4
    pred[~np.isin(pred, [0, 1, 2, 4])] = 0
    gt_img = nib.load(str(gt_path))
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred, gt_img.affine, gt_img.header), str(dst))


def main() -> int:
    ensure_project_dirs()
    cases = load_selected_cases()
    prepare_kaist_input(cases)

    if not executable_available("docker"):
        text = (
            "Docker is not available. KAIST inference is blocked until Docker and NVIDIA Container Toolkit are installed.\n"
            "Required image: rixez/brats21nnunet\n"
        )
        print(text)
        log_text("03_run_kaist_inference.log", text)
        return 2

    pull = run_command(["docker", "pull", IMAGE], "03_kaist_docker_pull.log")
    if pull.returncode != 0:
        print("Docker pull failed. See logs/03_kaist_docker_pull.log")
        return 2

    KAIST_PRED_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "device=0",
        "-v",
        f"{KAIST_INPUT_DIR.resolve()}:/input",
        "-v",
        f"{KAIST_PRED_DIR.resolve()}:/output",
        IMAGE,
    ]
    result = run_command(command, "03_kaist_docker_run.log")
    if result.returncode != 0:
        print("KAIST Docker inference failed. See logs/03_kaist_docker_run.log")
        return 2

    completed = 0
    for case in cases:
        output = find_kaist_output(case["case_id"])
        if output is None:
            log_text(f"03_kaist_normalize_{case['case_id']}.log", "No KAIST output found for case\n")
            continue
        normalize_labelmap(output, NORM_DIR / "kaist" / f"{case['case_id']}_seg.nii.gz", Path(case["seg"]))
        completed += 1

    print(f"KAIST completed cases: {completed}/{len(cases)}")
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
