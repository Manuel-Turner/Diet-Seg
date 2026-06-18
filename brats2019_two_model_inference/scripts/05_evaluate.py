from __future__ import annotations

from collections import defaultdict

from common import METRICS_DIR, NORM_DIR, REGIONS, ensure_project_dirs, load_selected_cases, write_csv, write_json


def dice_score(pred_mask, gt_mask) -> float:
    import numpy as np

    pred_sum = int(np.count_nonzero(pred_mask))
    gt_sum = int(np.count_nonzero(gt_mask))
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    intersection = int(np.count_nonzero(pred_mask & gt_mask))
    return float(2.0 * intersection / (pred_sum + gt_sum))


def region_mask(label, labels):
    import numpy as np

    return np.isin(label, labels)


def evaluate_case_model(case, model):
    import nibabel as nib

    case_id = case["case_id"]
    pred_path = NORM_DIR / model / f"{case_id}_seg.nii.gz"
    if not pred_path.exists():
        return None
    gt = nib.load(case["seg"]).get_fdata().astype("uint8")
    pred = nib.load(str(pred_path)).get_fdata().astype("uint8")
    row = {"case_id": case_id, "grade_group": case["grade_group"], "model": model}
    dice_values = []
    for region, labels in REGIONS.items():
        gt_mask = region_mask(gt, labels)
        pred_mask = region_mask(pred, labels)
        dice = dice_score(pred_mask, gt_mask)
        row[f"dice_{region}"] = f"{dice:.6f}"
        row[f"gt_{region}_voxels"] = int(gt_mask.sum())
        row[f"pred_{region}_voxels"] = int(pred_mask.sum())
        dice_values.append(dice)
    row["mean_dice"] = f"{sum(dice_values) / len(dice_values):.6f}"
    return row


def main() -> int:
    ensure_project_dirs()
    cases = load_selected_cases()
    rows = []
    for case in cases:
        for model in ("monai", "kaist"):
            row = evaluate_case_model(case, model)
            if row is not None:
                rows.append(row)

    fieldnames = [
        "case_id",
        "grade_group",
        "model",
        "dice_WT",
        "dice_TC",
        "dice_ET",
        "mean_dice",
        "gt_WT_voxels",
        "gt_TC_voxels",
        "gt_ET_voxels",
        "pred_WT_voxels",
        "pred_TC_voxels",
        "pred_ET_voxels",
    ]
    write_csv(METRICS_DIR / "dice_by_case.csv", rows, fieldnames)

    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    summary = []
    for model, model_rows in sorted(by_model.items()):
        summary.append(
            {
                "model": model,
                "mean_dice_WT": f"{sum(float(r['dice_WT']) for r in model_rows) / len(model_rows):.6f}",
                "mean_dice_TC": f"{sum(float(r['dice_TC']) for r in model_rows) / len(model_rows):.6f}",
                "mean_dice_ET": f"{sum(float(r['dice_ET']) for r in model_rows) / len(model_rows):.6f}",
                "mean_mean_dice": f"{sum(float(r['mean_dice']) for r in model_rows) / len(model_rows):.6f}",
                "num_cases": len(model_rows),
            }
        )
    write_csv(
        METRICS_DIR / "summary_by_model.csv",
        summary,
        ["model", "mean_dice_WT", "mean_dice_TC", "mean_dice_ET", "mean_mean_dice", "num_cases"],
    )

    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(float(row["mean_dice"]))
    complete_cases = {case_id: sum(values) / len(values) for case_id, values in by_case.items() if len(values) == 2}
    if complete_cases:
        success = max(complete_cases, key=complete_cases.get)
        failure = min(complete_cases, key=complete_cases.get)
        write_json(
            METRICS_DIR / "success_failure_cases.json",
            {
                "success_case": success,
                "failure_case": failure,
                "selection_rule": "highest/lowest average of MONAI and KAIST mean_dice",
                "case_scores": complete_cases,
            },
        )
        print(f"Success case: {success}; failure case: {failure}")
    else:
        write_json(
            METRICS_DIR / "success_failure_cases.json",
            {"success_case": None, "failure_case": None, "reason": "No case has both MONAI and KAIST predictions."},
        )
        print("No complete MONAI+KAIST case pairs available for success/failure selection.")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
