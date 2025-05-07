import torch
import matplotlib.pyplot as plt
import glob
import nibabel as nib
import numpy as np
import os
from torch.nn import functional as F
import pickle
from tqdm import tqdm


class HardnessCalculator:
    """
    Utility class for calculating sample difficulty in segmentation tasks
    Difficulty is calculated based on the difference between prediction results and ground truth labels
    """
    def __init__(self, pred_folder, gt_folder, use_gz=False):
        """
        Initialize hardness calculator
        
        Args:
            pred_folder: Prediction results folder path
            gt_folder: Ground truth labels folder path
            use_gz: Whether to use .nii.gz format (default is .nii)
        """
        self.pred_folder = pred_folder
        self.gt_folder = gt_folder
        self.use_gz = use_gz
        
        # 获取文件列表
        self.generated = glob.glob(pred_folder + "/*")
        self.origin_imgs = list(filter(lambda x: '_seg' not in x, glob.glob(
            gt_folder + ("/**/*.nii.gz" if use_gz else "/**/*.nii"), recursive=True)))
        self.ground_truth = glob.glob(gt_folder + ("/**/*_seg.nii.gz" if use_gz else "/**/*_seg.nii"), recursive=True)
        self.file_ids = [os.path.basename(f).split('.')[0] for f in self.generated]
    
    def read_data(self, data_idx):
        """
        Read data at specified index
        
        Args:
            data_idx: Data index
            
        Returns:
            Prediction probabilities, prediction binary map, ground truth label, original image
        """
        data = nib.load(self.generated[data_idx]).get_fdata()
        origin = nib.load(self.origin_imgs[data_idx]).get_fdata()
        label = nib.load(self.ground_truth[data_idx]).get_fdata()

        probs = torch.from_numpy(data).squeeze(-2).permute(3, 2, 0, 1) # [3, 155, 240, 240]
        prob_thresh = probs > 0.5
        
        return probs, prob_thresh, label, origin
    
    def calculate_hardness(self, save_path="./hardness_dict.pkl"):
        """
        Calculate difficulty for all samples and save
        
        Args:
            save_path: Save path
            
        Returns:
            Hardness dictionary, keys are file IDs, values are hardness values
        """
        hardness_dict = {}
        
        for i in tqdm(range(len(self.file_ids))):
            probs, prob_thresh, label, _ = self.read_data(i)
            
            # Convert label format
            label_tensor = torch.from_numpy(label).long()
            label_tensor = F.one_hot(label_tensor, num_classes=4).permute(3, 0, 1, 2).float()
            label_tensor = label_tensor[1:] # 去除背景类
            
            # Calculate Dice coefficient
            dice_scores = []
            for c in range(3): # 3个前景类
                pred = prob_thresh[c].float()
                target = label_tensor[c]
                
                intersection = (pred * target).sum()
                union = pred.sum() + target.sum()
                
                if union > 0:
                    dice = 2 * intersection / union
                else:
                    dice = torch.tensor(1.0) # If this class doesn't exist, set to 1
                
                dice_scores.append(dice.item())
            
            # Calculate hardness (1 - average Dice)
            avg_dice = np.mean(dice_scores)
            hardness = 1.0 - avg_dice
            
            # Store hardness value
            hardness_dict[self.file_ids[i]] = hardness
        
        # Save hardness dictionary
        with open(save_path, 'wb') as f:
            pickle.dump(hardness_dict, f)
        
        return hardness_dict
    
    def visualize_hardness_distribution(self, hardness_dict=None):
        """
        可视化难度分布
        
        Args:
            hardness_dict: 难度字典，如果为None则加载已保存的字典
        """
        if hardness_dict is None:
            with open("./hardness_dict.pkl", 'rb') as f:
                hardness_dict = pickle.load(f)
        
        hardness_values = list(hardness_dict.values())
        
        plt.figure(figsize=(10, 6))
        plt.hist(hardness_values, bins=20, alpha=0.7)
        plt.xlabel('Hardness Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Segmentation Hardness')
        plt.grid(True, alpha=0.3)
        plt.savefig('hardness_distribution.png')
        plt.show()
        
        # 打印统计信息
        print(f"Average Hardness: {np.mean(hardness_values):.4f}")
        print(f"Median Hardness: {np.median(hardness_values):.4f}")
        print(f"Min Hardness: {np.min(hardness_values):.4f}")
        print(f"Max Hardness: {np.max(hardness_values):.4f}")


# 使用示例
if __name__ == "__main__":
    src_folder = r'path/to/predictions'
    gt_folder = r"path/to/ground_truth"
    
    calculator = HardnessCalculator(src_folder, gt_folder)
    hardness_dict = calculator.calculate_hardness("./hardness_dict.pkl")
    calculator.visualize_hardness_distribution(hardness_dict)