import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


BRATS_REGIONS = {
    "ET": [4],
    "WT": [1, 2, 4],
    "TC": [1, 4],
}


@dataclass
class BratsConfig:
    """BraTS dataset configuration with backward-compatible defaults."""

    train_folder: str = r"C:\Users\admin\Desktop\MICCAI_BraTS_2019_Data_Training\HGG"
    data_multip: int = 1
    target_size: Tuple[int, int, int] = (128, 128, 128)
    modalities: List[str] = field(default_factory=lambda: ["t1", "t1ce", "t2", "flair"])
    debug: bool = False
    normalisation: str = "nonzero_zscore"
    is_gz: bool = True
    case_ids: Optional[List[str]] = None
    hardness_dir: Optional[str] = None
    augment: bool = False
    foreground_crop: bool = True
    foreground_aware_sampling: bool = True
    foreground_ratio: float = 0.67
    output_regions: Dict[str, List[int]] = field(default_factory=lambda: dict(BRATS_REGIONS))


def config_from_yaml(config: Dict, split_cases: Optional[List[str]] = None, training: bool = True) -> BratsConfig:
    return BratsConfig(
        train_folder=config["data"]["root"],
        target_size=tuple(config["sampling"]["patch_size"]),
        modalities=list(config["data"]["modalities"]),
        normalisation=config["preprocessing"]["intensity_normalization"],
        case_ids=split_cases,
        hardness_dir=config["hardness"]["save_dir"],
        augment=training and bool(config["augmentation"]["enabled"]),
        foreground_crop=bool(config["preprocessing"]["foreground_crop"]),
        foreground_aware_sampling=bool(config["sampling"]["foreground_aware_sampling"]),
        foreground_ratio=float(config["sampling"]["foreground_ratio"]),
        output_regions=dict(config["data"]["output_regions"]),
    )


class ImageProcessor:
    @staticmethod
    def nonzero_zscore(image: np.ndarray) -> np.ndarray:
        mask = image != 0
        if not np.any(mask):
            return image.astype(np.float32)
        values = image[mask]
        mean = values.mean()
        std = values.std()
        if std < 1.0e-8:
            std = 1.0
        normalized = np.zeros_like(image, dtype=np.float32)
        normalized[mask] = (image[mask] - mean) / std
        return normalized

    @staticmethod
    def minmax(image: np.ndarray) -> np.ndarray:
        min_value = float(np.min(image))
        max_value = float(np.max(image))
        if max_value - min_value < 1.0e-8:
            return np.zeros_like(image, dtype=np.float32)
        return ((image - min_value) / (max_value - min_value)).astype(np.float32)

    @staticmethod
    def foreground_bbox(label: np.ndarray) -> Tuple[slice, slice, slice]:
        coords = np.where(label > 0)
        if len(coords[0]) == 0:
            return (slice(0, label.shape[0]), slice(0, label.shape[1]), slice(0, label.shape[2]))
        return tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coords)

    @staticmethod
    def pad_to_size(array: np.ndarray, target_size: Sequence[int], channel_first: bool) -> np.ndarray:
        spatial_shape = array.shape[1:] if channel_first else array.shape
        pad_width = [(0, 0)] if channel_first else []
        for dim, target in zip(spatial_shape, target_size):
            missing = max(int(target) - int(dim), 0)
            left = missing // 2
            pad_width.append((left, missing - left))
        return np.pad(array, pad_width, mode="constant")

    @staticmethod
    def crop_patch(
        image: np.ndarray,
        label: np.ndarray,
        target_size: Sequence[int],
        foreground_aware: bool,
        foreground_ratio: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        spatial_shape = np.asarray(image.shape[1:])
        target = np.asarray(target_size)
        max_start = np.maximum(spatial_shape - target, 0)

        use_foreground = foreground_aware and np.any(label > 0) and random.random() < foreground_ratio
        if use_foreground:
            voxels = np.argwhere(label > 0)
            center = voxels[random.randrange(len(voxels))]
            start = center - target // 2
            start = np.minimum(np.maximum(start, 0), max_start)
        else:
            start = np.asarray([random.randint(0, int(v)) if v > 0 else 0 for v in max_start])

        slices = tuple(slice(int(s), int(s + t)) for s, t in zip(start, target))
        return image[(slice(None),) + slices], label[slices]


class BratsDataset(Dataset):
    """BraTS loader with fixed modality order and ET/WT/TC region labels."""

    def __init__(self, config: BratsConfig = BratsConfig()):
        super().__init__()
        self.config = config
        self.image_processor = ImageProcessor()
        self.data_list = self._load_data_list()
        self.data_count = len(self.data_list)

    def _load_data_list(self) -> List[Dict[str, Path]]:
        base_folder = Path(self.config.train_folder)
        if not base_folder.exists():
            raise FileNotFoundError(f"BraTS data path does not exist: {base_folder}")

        allowed = set(self.config.case_ids) if self.config.case_ids else None
        data_list = []
        for patient_dir in sorted(base_folder.iterdir()):
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name
            if allowed is not None and patient_id not in allowed:
                continue
            ext = ".nii.gz" if self.config.is_gz else ".nii"
            paths = {
                mod: patient_dir / f"{patient_id}_{mod}{ext}"
                for mod in self.config.modalities
            }
            seg_path = patient_dir / f"{patient_id}_seg{ext}"
            if not all(path.exists() for path in paths.values()) or not seg_path.exists():
                continue
            data_list.append({"id": patient_id, **paths, "seg": seg_path})
        if self.config.debug:
            data_list = data_list[:3]
        if not data_list:
            raise ValueError(f"No BraTS cases found in {base_folder}")
        return data_list

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        if self.config.normalisation == "nonzero_zscore":
            return self.image_processor.nonzero_zscore(image)
        if self.config.normalisation in {"minmax", "irm"}:
            return self.image_processor.minmax(image)
        raise ValueError(f"Unsupported normalization: {self.config.normalisation}")

    def _process_regions(self, label: np.ndarray) -> np.ndarray:
        regions = []
        for region_name in ["ET", "WT", "TC"]:
            labels = self.config.output_regions[region_name]
            regions.append(np.isin(label, labels).astype(np.float32))
        return np.stack(regions, axis=0)

    def _load_hardness(self, patient_id: str, target_size: Sequence[int]) -> torch.Tensor:
        if not self.config.hardness_dir:
            return torch.zeros(tuple(target_size), dtype=torch.float32)
        base = Path(self.config.hardness_dir)
        candidates = [
            base / f"{patient_id}.npy",
            base / f"{patient_id}.nii.gz",
            base / f"{patient_id}.nii",
        ]
        for path in candidates:
            if path.exists():
                if path.suffix == ".npy":
                    return torch.from_numpy(np.load(path)).float()
                return torch.from_numpy(nib.load(path).get_fdata(dtype=np.float32)).float()
        return torch.zeros(tuple(target_size), dtype=torch.float32)

    def _augment(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config.augment:
            return image, label
        for axis in range(3):
            if random.random() < 0.2:
                image = np.flip(image, axis=axis + 1).copy()
                label = np.flip(label, axis=axis + 1).copy()
        if random.random() < 0.2:
            axes = random.choice([(1, 2), (1, 3), (2, 3)])
            k = random.randint(1, 3)
            image = np.rot90(image, k=k, axes=axes).copy()
            label = np.rot90(label, k=k, axes=axes).copy()
        if random.random() < 0.1:
            image = image * random.uniform(0.9, 1.1)
        if random.random() < 0.1:
            image = image + random.uniform(-0.1, 0.1)
        return image, label

    def __len__(self) -> int:
        return self.data_count * self.config.data_multip

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        actual_idx = idx % self.data_count
        data_dict = self.data_list[actual_idx]
        patient_id = data_dict["id"]

        images = [self._normalize(nib.load(data_dict[mod]).get_fdata(dtype=np.float32)) for mod in self.config.modalities]
        image = np.stack(images, axis=0)
        raw_label = nib.load(data_dict["seg"]).get_fdata(dtype=np.float32).astype(np.int16)

        if self.config.foreground_crop:
            bbox = self.image_processor.foreground_bbox(raw_label)
            image = image[(slice(None),) + bbox]
            raw_label = raw_label[bbox]

        image = self.image_processor.pad_to_size(image, self.config.target_size, channel_first=True)
        raw_label = self.image_processor.pad_to_size(raw_label, self.config.target_size, channel_first=False)
        image, raw_label = self.image_processor.crop_patch(
            image,
            raw_label,
            self.config.target_size,
            self.config.foreground_aware_sampling,
            self.config.foreground_ratio,
        )
        label = self._process_regions(raw_label)
        image, label = self._augment(image, label)

        hardness = self._load_hardness(patient_id, self.config.target_size)
        if tuple(hardness.shape) != tuple(self.config.target_size):
            hardness = torch.zeros(tuple(self.config.target_size), dtype=torch.float32)

        return {
            "image": torch.from_numpy(image.copy()).float(),
            "label": torch.from_numpy(label.copy()).float(),
            "hardness": hardness.float(),
            "patient_id": patient_id,
        }


def get_brats_dataset(config: BratsConfig = BratsConfig()) -> BratsDataset:
    return BratsDataset(config)
