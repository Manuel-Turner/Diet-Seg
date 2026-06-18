from __future__ import annotations

import argparse

from common import METRICS_DIR, OUTPUTS_DIR, ensure_project_dirs, load_selected_cases, parse_models_arg, read_csv, read_json


def rows_for_case(rows, case_id):
    return [row for row in rows if row["case_id"] == case_id]


def describe_failure(rows):
    if not rows:
        return "No paired prediction metrics are available."
    observations = []
    for row in rows:
        weakest = min(("WT", "TC", "ET"), key=lambda r: float(row[f"dice_{r}"]))
        observations.append(f"{row['model']} weakest region: {weakest} Dice={float(row[f'dice_{weakest}']):.3f}")
    return "; ".join(observations)


def metric_line(row):
    return (
        f"{row['model']}: WT={float(row['dice_WT']):.3f}, "
        f"TC={float(row['dice_TC']):.3f}, ET={float(row['dice_ET']):.3f}, "
        f"mean={float(row['mean_dice']):.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create markdown report")
    parser.add_argument("--models", default="monai,kaist", help="Comma-separated model list: monai, kaist")
    args = parser.parse_args()
    models = parse_models_arg(args.models)

    ensure_project_dirs()
    selected = load_selected_cases()
    source_path = METRICS_DIR / "dataset_source.json"
    selection_path = METRICS_DIR / "success_failure_cases.json"
    summary_path = METRICS_DIR / "summary_by_model.csv"
    dice_path = METRICS_DIR / "dice_by_case.csv"
    if not (selection_path.exists() and summary_path.exists() and dice_path.exists()):
        print("Metrics are incomplete; run scripts/05_evaluate.py first.")
        return 2

    source = read_json(source_path) if source_path.exists() else {}
    selection = read_json(selection_path)
    summary_rows = read_csv(summary_path)
    dice_rows = read_csv(dice_path)
    success_id = selection.get("success_case")
    failure_id = selection.get("failure_case")

    lines = [
        "# BraTS2019 Two-Model Inference Report",
        "",
        "This is an inference-only demonstration and not a retraining experiment.",
        "",
        "## Dataset",
        "",
        f"Dataset source used: `{source.get('source_type', 'unknown')}` from `{source.get('source_root', 'unknown')}`.",
        f"Number of selected cases: {len(selected)}.",
        "",
        "| case_id | grade_group |",
        "| --- | --- |",
    ]
    for case in selected:
        lines.append(f"| {case['case_id']} | {case['grade_group']} |")

    lines.extend(
        [
            "",
            "## Models",
            "",
            "- Model A: MONAI `brats_mri_segmentation`, pretrained, inference only.",
            "- Model B: KAIST BraTS21 nnU-Net, pretrained Docker image `rixez/brats21nnunet`, inference only"
            + ("." if "kaist" in models else " (not run in this stage)."),
            "",
            "## Summary Metrics",
            "",
            "| model | mean_dice_WT | mean_dice_TC | mean_dice_ET | mean_mean_dice | num_cases |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {float(row['mean_dice_WT']):.3f} | {float(row['mean_dice_TC']):.3f} | "
            f"{float(row['mean_dice_ET']):.3f} | {float(row['mean_mean_dice']):.3f} | {row['num_cases']} |"
        )

    lines.extend(["", "## Success Case", ""])
    if success_id:
        success_rows = rows_for_case(dice_rows, success_id)
        lines.append(f"Success case: `{success_id}`.")
        lines.extend([f"- {metric_line(row)}" for row in success_rows])
        lines.append("This case was selected because it had the highest average mean Dice across the requested model outputs.")
        lines.append(f"![Success case](figures/success_case_{success_id}.png)")
    else:
        lines.append("No success case could be selected because paired model outputs were unavailable.")

    lines.extend(["", "## Failure Case", ""])
    if failure_id:
        failure_rows = rows_for_case(dice_rows, failure_id)
        lines.append(f"Failure or poor case: `{failure_id}`.")
        lines.extend([f"- {metric_line(row)}" for row in failure_rows])
        lines.append(
            "Likely failure pattern based on metrics and visualization: "
            + describe_failure(failure_rows)
            + ". Inspect the overlays for missed ET, edema over-segmentation, tumor-core under-segmentation, or boundary mismatch."
        )
        lines.append(f"![Failure case](figures/failure_case_{failure_id}.png)")
    else:
        lines.append("No failure case could be selected because paired model outputs were unavailable.")

    if "kaist" not in models:
        lines.extend(
            [
                "",
                "## Deferred KAIST Stage",
                "",
                "KAIST Docker inference was not requested for this report. Run Stage 2 after Docker Desktop, WSL2 GPU support, and NVIDIA Container Toolkit are available.",
            ]
        )

    report = OUTPUTS_DIR / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
