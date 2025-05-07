import os
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional, Union
import nibabel as nib
import numpy as np
import torch
import random
from dataclasses import dataclass, field
from torch.utils.data.dataset import Dataset

@dataclass
class BratsConfig:
    """BraTS dataset configuration class"""
    train_folder: str = r"C:\Users\admin\Desktop\MICCAI_BraTS_2019_Data_Training\HGG"
    data_multip: int = 5
    target_size: Tuple[int, int, int] = (128, 128, 128)
    # Use field and default_factory to set default list
    modalities: List[str] = field(default_factory=lambda: ["t1", "t1ce", "t2", "flair"])
    debug: bool = False
    normalisation: str = "minmax"
    is_gz: bool = True

class ImageProcessor:
    """Image processing utility class"""
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """Min-max normalization"""
        min_ = np.min(image)
        max_ = np.max(image)
        return (image - min_) / (max_ - min_)

    @staticmethod
    def irm_min_max_preprocess(image: np.ndarray, low_perc: int = 1, high_perc: int = 99) -> np.ndarray:
        """Normalization after percentile clipping"""
        non_zeros = image > 0
        low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
        return ImageProcessor.normalize(np.clip(image, low, high))

    @staticmethod
    def get_crop_params(array_shape: Tuple[int, ...], target_size: Tuple[int, ...]) -> List[slice]:
        """Calculate cropping parameters"""
        slices = []
        for dim, target in zip(array_shape, target_size):
            if dim > target:
                crop_extent = dim - target
                left = random.randint(0, crop_extent)
                slices.append(slice(left, left + target))
            else:
                slices.append(slice(0, dim))
        return slices

    @staticmethod
    def get_pad_params(array_shape: Tuple[int, ...], target_size: Tuple[int, ...]) -> List[Tuple[int, int]]:
        """Calculate padding parameters"""
        pad_params = [(0, 0)]  # channel dimension
        for dim, target in zip(array_shape[1:], target_size):
            if dim < target:
                pad_extent = target - dim
                left = random.randint(0, pad_extent)
                pad_params.append((left, pad_extent - left))
            else:
                pad_params.append((0, 0))
        return pad_params

class BratsDataset(Dataset):
    """BraTS dataset class"""
    def __init__(self, config: BratsConfig = BratsConfig()):
        super().__init__()
        self.config = config
        self.image_processor = ImageProcessor()
        self.data_list = self._load_data_list()
        self.data_count = len(self.data_list)
        self.hardness_dict = self._load_hardness_dict()

    def _load_hardness_dict(self) -> Dict[str, np.ndarray]:
        """Load hardness dictionary"""
        try:
            with open("./hardness_dict_f16.pkl", 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("Warning: Hardness dictionary file does not exist, will use default hardness value")
            return {}

    def _load_data_list(self) -> List[Dict[str, Path]]:
        """Load data list"""
        base_folder = Path(self.config.train_folder)
        assert base_folder.exists(), f"Training data path does not exist: {base_folder}"
        
        data_list = []
        for patient_dir in sorted(base_folder.iterdir()):
            if patient_dir.is_dir():
                patient_id = patient_dir.name
                ext = ".nii.gz" if self.config.is_gz else ".nii"
                data_dict = {
                    'id': patient_id,
                    **{mod: patient_dir / f"{patient_id}_{mod}{ext}" 
                       for mod in self.config.modalities},
                    'seg': patient_dir / f"{patient_id}_seg{ext}"
                }
                data_list.append(data_dict)
        return data_list[:3] if self.config.debug else data_list

    def _process_label(self, label: np.ndarray) -> Tuple[np.ndarray, int]:
        """Process segmentation labels, generate single-channel class labels suitable for CrossEntropyLoss
        Args:
            label: Original label array
            
        Returns:
            Processed label array and foreground voxel count
        """
        # Convert labels to single-channel class labels (0=background, 1=necrotic core, 2=edema, 3=enhancing tumor)
        processed_label = np.zeros_like(label)
        processed_label[label == 1] = 1  # Necrotic core
        processed_label[label == 2] = 2  # Edema
        processed_label[label == 4] = 3  # Enhancing tumor
        
        # Calculate foreground voxel count
        foreground_voxels = np.sum(processed_label > 0)
        
        return processed_label, foreground_voxels

    def __len__(self) -> int:
        return self.data_count * self.config.data_multip

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image, label, and hardness information
        """
        # Calculate actual index
        actual_idx = idx % self.data_count
        data_dict = self.data_list[actual_idx]
        patient_id = data_dict['id']
        
        # Load image data
        images = []
        for modality in self.config.modalities:
            img_path = data_dict[modality]
            img = nib.load(img_path).get_fdata()
            
            # Normalization
            if self.config.normalisation == "minmax":
                img = self.image_processor.normalize(img)
            elif self.config.normalisation == "irm":
                img = self.image_processor.irm_min_max_preprocess(img)
            
            images.append(img)
        
        # Load segmentation label
        seg_path = data_dict['seg']
        label = nib.load(seg_path).get_fdata()
        processed_label, _ = self._process_label(label)
        
        # Random crop to target size
        crop_slices = self.image_processor.get_crop_params(images[0].shape, self.config.target_size)
        
        # Apply cropping
        images = [img[crop_slices[0], crop_slices[1], crop_slices[2]] for img in images]
        processed_label = processed_label[crop_slices[0], crop_slices[1], crop_slices[2]]
        
        # Convert to tensors
        image_tensor = torch.stack([torch.from_numpy(img).float() for img in images])
        label_tensor = torch.from_numpy(processed_label).long() - 1  # Convert to 0-indexed
        
        # Get hardness value
        hardness = self.hardness_dict.get(patient_id, 0.5)  # Default hardness is 0.5
        hardness_tensor = torch.tensor(hardness).float()
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'hardness': hardness_tensor,
            'patient_id': patient_id
        }

# Helper function to create dataset
def get_brats_dataset(config: BratsConfig = BratsConfig()) -> BratsDataset:
    """Create BraTS dataset instance
    
    Args:
        config: Dataset configuration
        
    Returns:
        BratsDataset instance
    """
    return BratsDataset(config)