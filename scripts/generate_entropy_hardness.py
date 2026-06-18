import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.entropy_hardness import (
    compute_entropy_hardness,
    entropy_stats,
    load_probability_map,
    normalize_per_volume,
    save_hardness_map,
)
from utils.repro_config import case_id_from_path, format_fold_path, load_config, load_split_cases


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fold-wise teacher entropy hardness maps")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--teacher_pred_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    fold = args.fold if args.fold is not None else int(config["experiment"]["fold_index"])

    split_file = Path(config["data"]["split_file"])
    if not split_file.is_absolute():
        split_file = Path(args.config).resolve().parent.parent / split_file
    train_cases = set(load_split_cases(split_file, fold, "train"))

    teacher_pred_dir = Path(args.teacher_pred_dir or format_fold_path(config["teacher"]["checkpoint_dir"], fold))
    output_dir = Path(args.output_dir or format_fold_path(config["hardness"]["save_dir"], fold))
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(
        [
            *teacher_pred_dir.glob("*.nii"),
            *teacher_pred_dir.glob("*.nii.gz"),
            *teacher_pred_dir.glob("*.npy"),
            *teacher_pred_dir.glob("*.npz"),
            *teacher_pred_dir.glob("*.pt"),
            *teacher_pred_dir.glob("*.pth"),
        ]
    )
    pred_files = [path for path in pred_files if case_id_from_path(path, config["data"]["modalities"]) in train_cases]
    if not pred_files:
        raise ValueError(f"No teacher probability maps found for training cases in {teacher_pred_dir}")

    stats = []
    for path in tqdm(pred_files, desc=f"fold {fold} entropy hardness"):
        case_id = case_id_from_path(path, config["data"]["modalities"])
        suffix = ".nii.gz" if path.name.endswith(".nii.gz") else ".npy"
        output_path = output_dir / f"{case_id}{suffix}"
        if output_path.exists() and not args.overwrite:
            continue
        probabilities, reference = load_probability_map(path)
        hardness = compute_entropy_hardness(probabilities)
        if config["hardness"].get("normalize", True):
            hardness = normalize_per_volume(hardness)
        save_hardness_map(hardness, output_path, reference)
        stats.append(entropy_stats(hardness))

    if stats:
        mins, maxs, means = zip(*stats)
        print(f"fold id: {fold}")
        print(f"training cases processed: {len(stats)}")
        print(f"output directory: {output_dir}")
        print(
            "entropy min/max/mean range: "
            f"min {min(mins):.6f}-{max(mins):.6f}, "
            f"max {min(maxs):.6f}-{max(maxs):.6f}, "
            f"mean {min(means):.6f}-{max(means):.6f}"
        )
    else:
        print(f"fold id: {fold}")
        print("training cases processed: 0 (all outputs already existed)")
        print(f"output directory: {output_dir}")


if __name__ == "__main__":
    main()
