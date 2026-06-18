from __future__ import annotations

import argparse
from pathlib import Path

from common import NORM_DIR, ensure_project_dirs, load_selected_cases, log_text, parse_models_arg


def nearest_resample_to_gt(pred, gt_shape):
    import numpy as np
    from scipy.ndimage import zoom

    factors = [g / p for g, p in zip(gt_shape, pred.shape)]
    return zoom(pred, factors, order=0).astype(np.uint8)


def normalize_one(model: str, case) -> bool:
    import nibabel as nib
    import numpy as np

    case_id = case["case_id"]
    pred_path = NORM_DIR / model / f"{case_id}_seg.nii.gz"
    if not pred_path.exists():
        log_text(f"04_normalize_{model}_{case_id}.log", f"Missing prediction: {pred_path}\n")
        return False

    gt_img = nib.load(case["seg"])
    gt_shape = gt_img.shape
    pred = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
    pred[pred == 3] = 4
    pred[~np.isin(pred, [0, 1, 2, 4])] = 0
    if pred.shape != gt_shape:
        pred = nearest_resample_to_gt(pred, gt_shape)
        log_text(
            f"04_normalize_{model}_{case_id}.log",
            f"Shape mismatch fixed with nearest-neighbor resampling to {gt_shape}\n",
        )
    nib.save(nib.Nifti1Image(pred, gt_img.affine, gt_img.header), str(pred_path))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize model predictions to BraTS labelmaps")
    parser.add_argument("--models", default="monai,kaist", help="Comma-separated model list: monai, kaist")
    args = parser.parse_args()
    models = parse_models_arg(args.models)

    ensure_project_dirs()
    cases = load_selected_cases()
    ok = 0
    total = len(cases) * len(models)
    for case in cases:
        for model in models:
            ok += int(normalize_one(model, case))
    print(f"Normalized predictions available: {ok}/{total}")
    return 0 if ok == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
