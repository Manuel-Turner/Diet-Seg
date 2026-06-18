"""Legacy prediction-vs-ground-truth Dice-error hardness.

This utility is not used for the revised Diet-Seg main experiments. The
revision uses fold-wise teacher-derived entropy hardness computed from teacher
softmax probability maps without ground-truth error.
"""

import glob
import os
import pickle

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


class HardnessCalculator:
    """Calculate legacy sample hardness as 1 - average Dice."""

    def __init__(self, pred_folder, gt_folder, use_gz=False):
        self.pred_folder = pred_folder
        self.gt_folder = gt_folder
        self.use_gz = use_gz
        self.generated = glob.glob(pred_folder + "/*")
        self.origin_imgs = list(
            filter(
                lambda x: "_seg" not in x,
                glob.glob(gt_folder + ("/**/*.nii.gz" if use_gz else "/**/*.nii"), recursive=True),
            )
        )
        self.ground_truth = glob.glob(
            gt_folder + ("/**/*_seg.nii.gz" if use_gz else "/**/*_seg.nii"),
            recursive=True,
        )
        self.file_ids = [os.path.basename(f).split(".")[0] for f in self.generated]

    def read_data(self, data_idx):
        data = nib.load(self.generated[data_idx]).get_fdata()
        origin = nib.load(self.origin_imgs[data_idx]).get_fdata()
        label = nib.load(self.ground_truth[data_idx]).get_fdata()
        probs = torch.from_numpy(data).squeeze(-2).permute(3, 2, 0, 1)
        prob_thresh = probs > 0.5
        return probs, prob_thresh, label, origin

    def calculate_hardness(self, save_path="./hardness_dict.pkl"):
        hardness_dict = {}
        for i in tqdm(range(len(self.file_ids))):
            _, prob_thresh, label, _ = self.read_data(i)
            label_tensor = torch.from_numpy(label).long()
            label_tensor = F.one_hot(label_tensor, num_classes=4).permute(3, 0, 1, 2).float()
            label_tensor = label_tensor[1:]

            dice_scores = []
            for c in range(3):
                pred = prob_thresh[c].float()
                target = label_tensor[c]
                intersection = (pred * target).sum()
                union = pred.sum() + target.sum()
                dice = 2 * intersection / union if union > 0 else torch.tensor(1.0)
                dice_scores.append(dice.item())

            hardness_dict[self.file_ids[i]] = 1.0 - float(np.mean(dice_scores))

        with open(save_path, "wb") as f:
            pickle.dump(hardness_dict, f)
        return hardness_dict

    def visualize_hardness_distribution(self, hardness_dict=None):
        if hardness_dict is None:
            with open("./hardness_dict.pkl", "rb") as f:
                hardness_dict = pickle.load(f)

        hardness_values = list(hardness_dict.values())
        plt.figure(figsize=(10, 6))
        plt.hist(hardness_values, bins=20, alpha=0.7)
        plt.xlabel("Legacy hardness score")
        plt.ylabel("Frequency")
        plt.title("Legacy prediction-error hardness distribution")
        plt.grid(True, alpha=0.3)
        plt.savefig("legacy_hardness_distribution.png")
        plt.show()
        print(f"Average Hardness: {np.mean(hardness_values):.4f}")
        print(f"Median Hardness: {np.median(hardness_values):.4f}")
        print(f"Min Hardness: {np.min(hardness_values):.4f}")
        print(f"Max Hardness: {np.max(hardness_values):.4f}")
