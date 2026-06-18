"""Run a tiny Diet-Seg preprocessing demo without training."""

import argparse
from pathlib import Path

import numpy as np

try:
    from .dataset import BratsPreprocessConfig, load_nifti_case, preprocess_arrays
except ImportError:  # pragma: no cover - supports direct execution
    from dataset import BratsPreprocessConfig, load_nifti_case, preprocess_arrays


def random_demo():
    shape = (155, 192, 192)
    rng = np.random.default_rng(1234)
    volumes = {
        "t1": rng.normal(size=shape).astype(np.float32),
        "t1ce": rng.normal(size=shape).astype(np.float32),
        "t2": rng.normal(size=shape).astype(np.float32),
        "flair": rng.normal(size=shape).astype(np.float32),
    }
    for volume in volumes.values():
        volume[rng.random(shape) < 0.35] = 0.0

    label = np.zeros(shape, dtype=np.int16)
    label[40:80, 60:110, 70:120] = 1
    label[55:95, 75:135, 85:145] = 2
    label[65:85, 90:115, 95:125] = 4
    return preprocess_arrays(volumes, label, BratsPreprocessConfig())


def main():
    parser = argparse.ArgumentParser(description="Diet-Seg BraTS preprocessing demo")
    parser.add_argument("--case_dir", default=None, help="Directory containing one BraTS case")
    parser.add_argument("--case_id", default=None, help="BraTS case id, for example BraTS19_2013_2_1")
    args = parser.parse_args()

    if args.case_dir and args.case_id:
        sample = load_nifti_case(Path(args.case_dir), args.case_id)
    else:
        sample = random_demo()

    image = sample["image"]
    label = sample["label"]
    print(f"image shape: {image.shape}  # [4, 128, 128, 128]")
    print(f"label shape: {label.shape}  # [3, 128, 128, 128] = ET, WT, TC")
    print(f"image dtype: {image.dtype}")
    print(f"label foreground voxels: ET={int(label[0].sum())}, WT={int(label[1].sum())}, TC={int(label[2].sum())}")


if __name__ == "__main__":
    main()
