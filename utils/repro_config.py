import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml


REQUIRED_CONFIG_KEYS = (
    "experiment",
    "data",
    "preprocessing",
    "sampling",
    "augmentation",
    "model",
    "teacher",
    "hardness",
    "loss",
    "optimizer",
    "training",
    "inference",
    "evaluation",
)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    patch_size = config["sampling"]["patch_size"]
    roi_size = config["inference"]["roi_size"]
    if patch_size != [128, 128, 128]:
        raise ValueError(f"Expected patch_size [128, 128, 128], got {patch_size}")
    if roi_size != [128, 128, 128]:
        raise ValueError(f"Expected roi_size [128, 128, 128], got {roi_size}")
    if float(config["inference"]["overlap"]) != 0.5:
        raise ValueError("Expected inference overlap 0.5")

    optimizer = config["optimizer"]
    expected_optimizer = {
        "name": "AdamW",
        "learning_rate": 1.0e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "weight_decay": 1.0e-5,
    }
    for key, expected in expected_optimizer.items():
        if optimizer[key] != expected:
            raise ValueError(f"Expected optimizer.{key}={expected}, got {optimizer[key]}")

    if config["data"]["modalities"] != ["t1", "t1ce", "t2", "flair"]:
        raise ValueError("Expected modality order [t1, t1ce, t2, flair]")
    if config["hardness"]["type"] != "teacher_entropy":
        raise ValueError("Revised release requires hardness.type=teacher_entropy")
    if config["hardness"]["weight_form"] != "1+h":
        raise ValueError("Revised release requires hardness weight_form=1+h")


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_fold_path(path_template: str, fold: int) -> str:
    return path_template.format(fold_index=fold, fold=fold)


def load_split_cases(split_file: str | Path, fold: int, split_name: str = "train") -> List[str]:
    path = Path(split_file)
    with open(path, "r", encoding="utf-8") as f:
        split = json.load(f)

    fold_keys = [str(fold), f"fold{fold}", f"fold_{fold}"]
    fold_data = None
    if isinstance(split, dict):
        if "folds" in split:
            folds = split["folds"]
            if isinstance(folds, list):
                fold_data = folds[fold]
            elif isinstance(folds, dict):
                for key in fold_keys:
                    if key in folds:
                        fold_data = folds[key]
                        break
        else:
            for key in fold_keys:
                if key in split:
                    fold_data = split[key]
                    break
    elif isinstance(split, list):
        fold_data = split[fold]

    if fold_data is None:
        raise ValueError(f"Could not find fold {fold} in split file {path}")
    if split_name not in fold_data:
        raise ValueError(f"Fold {fold} does not contain split '{split_name}'")
    return [str(case_id) for case_id in fold_data[split_name]]


def create_adamw(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.AdamW:
    optimizer_cfg = config["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["learning_rate"]),
        betas=tuple(float(v) for v in optimizer_cfg["betas"]),
        eps=float(optimizer_cfg["epsilon"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )


def case_id_from_path(path: str | Path, modalities: Iterable[str] = ("t1", "t1ce", "t2", "flair")) -> str:
    name = Path(path).name
    for suffix in (".nii.gz", ".nii", ".npz", ".npy", ".pt", ".pth"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for suffix in [f"_{mod}" for mod in modalities] + ["_seg", "_prob", "_probs", "_pred"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name
