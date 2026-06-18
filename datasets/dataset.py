"""Minimal BraTS preprocessing helpers for Diet-Seg demos."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


BRATS_MODALITIES = ("t1", "t1ce", "t2", "flair")
BRATS_REGIONS = {
    "ET": (4,),
    "WT": (1, 2, 4),
    "TC": (1, 4),
}


@dataclass
class BratsPreprocessConfig:
    modalities: List[str] = field(default_factory=lambda: list(BRATS_MODALITIES))
    patch_size: Tuple[int, int, int] = (128, 128, 128)
    intensity_normalization: str = "nonzero_zscore"


def nonzero_zscore(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize only non-zero brain voxels."""
    volume = volume.astype(np.float32, copy=False)
    mask = volume != 0
    if not np.any(mask):
        return np.zeros_like(volume, dtype=np.float32)
    values = volume[mask]
    std = float(values.std())
    if std < 1.0e-8:
        std = 1.0
    output = np.zeros_like(volume, dtype=np.float32)
    output[mask] = (values - float(values.mean())) / std
    return output


def crop_or_pad_center(array: np.ndarray, target_size: Sequence[int]) -> np.ndarray:
    """Center crop or zero-pad a 3D/4D array to target spatial size."""
    target = tuple(int(v) for v in target_size)
    channel_first = array.ndim == 4
    spatial = array.shape[1:] if channel_first else array.shape

    pad_width = [(0, 0)] if channel_first else []
    for current, desired in zip(spatial, target):
        missing = max(desired - current, 0)
        pad_width.append((missing // 2, missing - missing // 2))
    padded = np.pad(array, pad_width, mode="constant")

    spatial = padded.shape[1:] if channel_first else padded.shape
    slices = []
    for current, desired in zip(spatial, target):
        start = max((current - desired) // 2, 0)
        slices.append(slice(start, start + desired))
    return padded[(slice(None), *slices)] if channel_first else padded[tuple(slices)]


def remap_brats_regions(label: np.ndarray) -> np.ndarray:
    """Map BraTS labels to ET, WT, TC region channels."""
    return np.stack(
        [np.isin(label, BRATS_REGIONS[name]).astype(np.float32) for name in ("ET", "WT", "TC")],
        axis=0,
    )


def preprocess_arrays(
    modality_volumes: Dict[str, np.ndarray],
    label: np.ndarray,
    config: BratsPreprocessConfig | None = None,
) -> Dict[str, np.ndarray]:
    """Preprocess in-memory BraTS arrays for a Diet-Seg model demo."""
    config = config or BratsPreprocessConfig()
    images = []
    for modality in config.modalities:
        if modality not in modality_volumes:
            raise KeyError(f"Missing modality: {modality}")
        volume = modality_volumes[modality]
        if config.intensity_normalization == "nonzero_zscore":
            volume = nonzero_zscore(volume)
        else:
            raise ValueError(f"Unsupported normalization: {config.intensity_normalization}")
        images.append(volume)

    image = np.stack(images, axis=0)
    image = crop_or_pad_center(image, config.patch_size)
    label = crop_or_pad_center(label, config.patch_size)
    regions = remap_brats_regions(label)
    return {"image": image.astype(np.float32), "label": regions.astype(np.float32)}


def load_nifti_case(case_dir: str | Path, case_id: str, config: BratsPreprocessConfig | None = None):
    """Load one BraTS-style case and run the demo preprocessing pipeline."""
    import nibabel as nib

    config = config or BratsPreprocessConfig()
    case_dir = Path(case_dir)
    volumes = {}
    for modality in config.modalities:
        path = case_dir / f"{case_id}_{modality}.nii.gz"
        if not path.exists():
            path = case_dir / f"{case_id}_{modality}.nii"
        volumes[modality] = nib.load(str(path)).get_fdata(dtype=np.float32)

    seg_path = case_dir / f"{case_id}_seg.nii.gz"
    if not seg_path.exists():
        seg_path = case_dir / f"{case_id}_seg.nii"
    label = nib.load(str(seg_path)).get_fdata(dtype=np.float32).astype(np.int16)
    return preprocess_arrays(volumes, label, config)
