from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from common import (
    RAW_DIR,
    MANUAL_CASES_DIR,
    append_log,
    ensure_project_dirs,
    discover_brats_cases,
    copy_case_to_subset,
    kaggle_credentials_path,
    kaggle_help_text,
    log_text,
    run_command,
    selected_cases_csv,
    write_csv,
    write_json,
    METRICS_DIR,
)


DATASET_SLUG = "aryashah2k/brain-tumor-segmentation-brats-2019"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a small BraTS2019 training subset")
    parser.add_argument("--brats_root", default=None, help="Path to MICCAI_BraTS_2019_Data_Training")
    parser.add_argument("--download_kaggle", action="store_true", help="Download Kaggle BraTS2019 mirror")
    parser.add_argument("--auto", action="store_true", help="Use data/raw if present, otherwise try Kaggle")
    parser.add_argument("--num_cases", type=int, default=6)
    return parser.parse_args()


def find_existing_raw_root() -> Path | None:
    candidates = [RAW_DIR]
    candidates.extend([p for p in RAW_DIR.rglob("*") if p.is_dir()])
    for candidate in candidates:
        if discover_brats_cases(candidate):
            return candidate
    return None


def find_manual_cases_root() -> Path | None:
    if not MANUAL_CASES_DIR.exists():
        return None
    return MANUAL_CASES_DIR if discover_brats_cases(MANUAL_CASES_DIR) else None


def expected_manual_format_text() -> str:
    return (
        "Expected manual mini dataset format:\n"
        "data/manual_cases/\n"
        "  <case_id>/\n"
        "    <case_id>_flair.nii.gz\n"
        "    <case_id>_t1.nii.gz\n"
        "    <case_id>_t1ce.nii.gz\n"
        "    <case_id>_t2.nii.gz\n"
        "    <case_id>_seg.nii.gz\n"
    )


def download_kaggle() -> bool:
    if shutil.which("kaggle") is None or not kaggle_credentials_path().exists():
        text = kaggle_help_text()
        print(text)
        log_text("01_prepare_data.log", text)
        return False

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(RAW_DIR),
        "--unzip",
    ]
    result = run_command(command, "01_prepare_data_kaggle_download.log")
    if result.returncode != 0:
        print("Kaggle download failed. See logs/01_prepare_data_kaggle_download.log")
        return False
    return True


def select_balanced(cases, num_cases):
    hgg = [case for case in cases if case["grade_group"] == "HGG"]
    lgg = [case for case in cases if case["grade_group"] == "LGG"]
    target_each = num_cases // 2
    selected = hgg[:target_each] + lgg[:target_each]
    remaining = [case for case in cases if case not in selected]
    selected.extend(remaining[: max(0, num_cases - len(selected))])
    notes = []
    if len(hgg) < target_each:
        notes.append(f"Only {len(hgg)} HGG cases found; selected fewer than requested.")
    if len(lgg) < target_each:
        notes.append(f"Only {len(lgg)} LGG cases found; selected fewer than requested.")
    if len(selected) < num_cases:
        notes.append(f"Only {len(selected)} complete cases found total.")
    return selected, notes


def main() -> int:
    args = parse_args()
    ensure_project_dirs()
    invoked = "python scripts/01_prepare_data.py " + " ".join(sys.argv[1:])

    source_root = None
    source_type = None
    if args.brats_root:
        source_root = Path(args.brats_root)
        source_type = "user_provided"
    elif args.download_kaggle:
        source_type = "kaggle"
        if not download_kaggle():
            return 2
        source_root = find_existing_raw_root()
    elif args.auto:
        source_root = find_manual_cases_root()
        source_type = "data/manual_cases"
        if source_root is None:
            source_root = find_existing_raw_root()
            source_type = "data/raw"
        if source_root is None:
            print("No complete cases found in data/manual_cases or data/raw; trying Kaggle download.")
            source_type = "kaggle"
            if not download_kaggle():
                text = (
                    f"\ninvoked command: {invoked}\n"
                    "error: no manual cases, no local BraTS2019 data, and Kaggle download unavailable\n\n"
                    + expected_manual_format_text()
                )
                print(expected_manual_format_text())
                append_log("01_prepare_data.log", text)
                return 2
            source_root = find_existing_raw_root()
    else:
        print("Specify --brats_root, --download_kaggle, or --auto.")
        return 2

    if source_root is None or not source_root.exists():
        text = f"BraTS2019 source root not found: {source_root}"
        print(text)
        log_text("01_prepare_data.log", f"invoked command: {invoked}\n{text}\n")
        return 2

    cases = discover_brats_cases(source_root)
    if not cases:
        text = f"No complete BraTS2019 training cases found under {source_root}"
        print(text)
        log_text("01_prepare_data.log", f"invoked command: {invoked}\n{text}\n")
        return 2

    selected, notes = select_balanced(cases, args.num_cases)
    copied = [copy_case_to_subset(case) for case in selected]
    fieldnames = ["case_id", "grade_group", "flair", "t1", "t1ce", "t2", "seg"]
    write_csv(selected_cases_csv(), copied, fieldnames)
    write_json(
        METRICS_DIR / "dataset_source.json",
        {
            "source_type": source_type,
            "source_root": str(source_root),
            "num_available_complete_cases": len(cases),
            "num_selected_cases": len(copied),
            "notes": notes,
        },
    )

    message = [
        f"Selected {len(copied)} cases from {source_root}",
        f"Wrote {selected_cases_csv()}",
        *notes,
    ]
    text = f"invoked command: {invoked}\n" + "\n".join(message) + "\n"
    print(text)
    log_text("01_prepare_data.log", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
