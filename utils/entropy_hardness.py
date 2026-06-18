from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def compute_entropy_hardness(probabilities: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Compute normalized voxel-wise entropy from teacher probability maps.

    Expected input shape is [C, ...] or [B, C, ...]. The function uses teacher
    probabilities, not hard masks or ground-truth errors.
    """
    if probabilities.ndim < 2:
        raise ValueError("Probability tensor must include a class/channel dimension")
    class_dim = 1 if probabilities.ndim >= 5 else 0
    classes = probabilities.shape[class_dim]
    if classes <= 1:
        raise ValueError("Entropy hardness requires at least two classes/regions")

    probs = probabilities.float().clamp_min(0)
    denom = probs.sum(dim=class_dim, keepdim=True).clamp_min(eps)
    probs = probs / denom
    entropy = -(probs * torch.log(probs + eps)).sum(dim=class_dim)
    normalized = entropy / math.log(classes)
    return normalized.clamp(0.0, 1.0)


def hardness_weight(hardness: torch.Tensor) -> torch.Tensor:
    return 1.0 + hardness.detach()


def normalize_per_volume(hardness: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    min_value = hardness.amin()
    max_value = hardness.amax()
    return (hardness - min_value) / (max_value - min_value + eps)


def entropy_stats(hardness: torch.Tensor) -> Tuple[float, float, float]:
    return float(hardness.min()), float(hardness.max()), float(hardness.mean())


def load_probability_map(path: str | Path) -> Tuple[torch.Tensor, object]:
    path = Path(path)
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path)), None
    if path.suffix == ".npz":
        archive = np.load(path)
        key = "probabilities" if "probabilities" in archive else archive.files[0]
        return torch.from_numpy(archive[key]), None
    if path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            data = data.get("probabilities", data.get("probs", data.get("pred", data)))
        if isinstance(data, dict):
            raise ValueError(f"Could not find probability tensor in {path}")
        return torch.as_tensor(data), None

    import nibabel as nib

    image = nib.load(str(path))
    array = image.get_fdata(dtype=np.float32)
    tensor = torch.from_numpy(array)
    if tensor.ndim >= 4 and tensor.shape[-1] <= 8:
        tensor = tensor.movedim(-1, 0)
    return tensor, image


def save_hardness_map(hardness: torch.Tensor, path: str | Path, reference_image: object = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = hardness.cpu().numpy().astype(np.float32)
    if path.suffix == ".npy" or reference_image is None:
        np.save(path.with_suffix(".npy"), array)
        return

    import nibabel as nib

    image = nib.Nifti1Image(array, reference_image.affine, reference_image.header)
    nib.save(image, str(path))
