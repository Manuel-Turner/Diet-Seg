from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANUAL_CASES_DIR = DATA_DIR / "manual_cases"
SUBSET_DIR = DATA_DIR / "subset"
KAIST_INPUT_DIR = DATA_DIR / "kaist_input"
MODELS_DIR = ROOT / "models"
MONAI_BUNDLE_DIR = MODELS_DIR / "monai_bundle"
OUTPUTS_DIR = ROOT / "outputs"
MONAI_PRED_DIR = OUTPUTS_DIR / "monai_pred"
KAIST_PRED_DIR = OUTPUTS_DIR / "kaist_pred"
NORM_DIR = OUTPUTS_DIR / "normalized_pred"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = ROOT / "logs"

MODALITIES = ("flair", "t1", "t1ce", "t2")
MONAI_MODALITY_ORDER = ("t1ce", "t1", "t2", "flair")
BRATS_LABELS = {0, 1, 2, 4}
REGIONS = {
    "WT": (1, 2, 4),
    "TC": (1, 4),
    "ET": (4,),
}


def ensure_project_dirs() -> None:
    for path in [
        RAW_DIR,
        MANUAL_CASES_DIR,
        SUBSET_DIR,
        KAIST_INPUT_DIR,
        MONAI_BUNDLE_DIR,
        MONAI_PRED_DIR,
        KAIST_PRED_DIR,
        NORM_DIR / "monai",
        NORM_DIR / "kaist",
        METRICS_DIR,
        FIGURES_DIR / "all_cases",
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def log_text(name: str, text: str) -> Path:
    ensure_project_dirs()
    path = LOGS_DIR / name
    path.write_text(text, encoding="utf-8")
    return path


def append_log(name: str, text: str) -> Path:
    ensure_project_dirs()
    path = LOGS_DIR / name
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    return path


def run_command(command: Sequence[str], log_name: str, cwd: Optional[Path] = None, check: bool = False):
    ensure_project_dirs()
    started = datetime.now().isoformat(timespec="seconds")
    header = [
        f"started: {started}",
        f"cwd: {cwd or ROOT}",
        "command: " + " ".join(str(part) for part in command),
        "",
    ]
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd or ROOT),
            text=True,
            capture_output=True,
            check=check,
        )
        body = [
            f"returncode: {completed.returncode}",
            "",
            "STDOUT:",
            completed.stdout,
            "",
            "STDERR:",
            completed.stderr,
        ]
        log_text(log_name, "\n".join(header + body))
        return completed
    except Exception as exc:
        log_text(log_name, "\n".join(header + [f"exception: {type(exc).__name__}: {exc}"]))
        raise


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def selected_cases_csv() -> Path:
    return METRICS_DIR / "selected_cases.csv"


def load_selected_cases() -> List[Dict[str, str]]:
    path = selected_cases_csv()
    if not path.exists():
        raise FileNotFoundError(f"Selected cases file not found: {path}. Run scripts/01_prepare_data.py first.")
    return read_csv(path)


def find_case_files(case_dir: Path, case_id: Optional[str] = None) -> Optional[Dict[str, Path]]:
    if case_id is None:
        case_id = case_dir.name
    files: Dict[str, Path] = {}
    for modality in MODALITIES:
        matches = sorted(case_dir.glob(f"*_{modality}.nii.gz")) + sorted(case_dir.glob(f"*_{modality}.nii"))
        if not matches:
            return None
        files[modality] = matches[0]
    seg_matches = sorted(case_dir.glob("*_seg.nii.gz")) + sorted(case_dir.glob("*_seg.nii"))
    if not seg_matches:
        return None
    files["seg"] = seg_matches[0]
    files["case_id"] = Path(case_id)  # sentinel stored as Path for compact typing
    return files


def discover_brats_cases(root: Path) -> List[Dict[str, str]]:
    cases: List[Dict[str, str]] = []
    for grade in ("HGG", "LGG"):
        for case_dir in sorted(root.rglob(grade + "/*")):
            if not case_dir.is_dir():
                continue
            found = find_case_files(case_dir)
            if found is None:
                continue
            cases.append(
                {
                    "case_id": case_dir.name,
                    "grade_group": grade,
                    "case_dir": str(case_dir),
                    "flair": str(found["flair"]),
                    "t1": str(found["t1"]),
                    "t1ce": str(found["t1ce"]),
                    "t2": str(found["t2"]),
                    "seg": str(found["seg"]),
                }
            )
    if cases:
        return cases

    for case_dir in sorted(root.rglob("*")):
        if not case_dir.is_dir():
            continue
        found = find_case_files(case_dir)
        if found is None:
            continue
        grade = "HGG" if "HGG" in [p.name for p in case_dir.parents] else "LGG" if "LGG" in [p.name for p in case_dir.parents] else "unknown"
        cases.append(
            {
                "case_id": case_dir.name,
                "grade_group": grade,
                "case_dir": str(case_dir),
                "flair": str(found["flair"]),
                "t1": str(found["t1"]),
                "t1ce": str(found["t1ce"]),
                "t2": str(found["t2"]),
                "seg": str(found["seg"]),
            }
        )
    return cases


def copy_case_to_subset(case: Dict[str, str]) -> Dict[str, str]:
    dest = SUBSET_DIR / case["case_id"]
    dest.mkdir(parents=True, exist_ok=True)
    copied = {"case_id": case["case_id"], "grade_group": case["grade_group"]}
    for key in ("flair", "t1", "t1ce", "t2", "seg"):
        src = Path(case[key])
        dst = dest / src.name
        if not dst.exists() or src.stat().st_size != dst.stat().st_size:
            shutil.copy2(src, dst)
        copied[key] = str(dst)
    return copied


def kaggle_credentials_path() -> Path:
    return Path.home() / ".kaggle" / "kaggle.json"


def kaggle_help_text() -> str:
    return (
        "Kaggle credentials are missing or Kaggle CLI is unavailable.\n"
        "\n"
        "Manual mini dataset fallback:\n"
        "Place cases in this exact format before running again:\n"
        "data/manual_cases/\n"
        "  <case_id>/\n"
        "    <case_id>_flair.nii.gz\n"
        "    <case_id>_t1.nii.gz\n"
        "    <case_id>_t1ce.nii.gz\n"
        "    <case_id>_t2.nii.gz\n"
        "    <case_id>_seg.nii.gz\n"
        "\n"
        "Manual fix:\n"
        "1. Create a Kaggle account.\n"
        "2. Go to Account settings.\n"
        "3. Create API token.\n"
        "4. Put kaggle.json in ~/.kaggle/kaggle.json on Linux/WSL or "
        "C:/Users/<USER>/.kaggle/kaggle.json on Windows.\n"
        "5. Run the command again.\n"
    )


def parse_models_arg(models: str) -> List[str]:
    parsed = [item.strip().lower() for item in models.split(",") if item.strip()]
    invalid = [item for item in parsed if item not in {"monai", "kaist"}]
    if invalid:
        raise ValueError(f"Unsupported model name(s): {invalid}. Use monai, kaist, or monai,kaist.")
    return parsed or ["monai", "kaist"]


def executable_available(name: str) -> bool:
    return shutil.which(name) is not None


def format_cmd(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)
