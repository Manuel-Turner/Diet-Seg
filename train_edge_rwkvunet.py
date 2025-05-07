import os
import sys

# 添加RWKV-UNet子模块到Python搜索路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RWKV-UNet'))

import torch
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from datasets.dataset import BratsConfig, get_brats_dataset
from models.rwkv_unet_edge import RWKV_UNet
from tensorboardX import SummaryWriter
import random
import argparse

class DiceLoss(torch.nn.Module):
    """Dice loss function"""
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, inputs, targets, mask=None, softmax=True, one_hot=False, weight=None):
        if softmax:
            inputs = F.softmax(inputs, dim=1)
        
        if one_hot and targets.dtype == torch.long:
            targets = F.one_hot(targets, self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        # 计算每个类别的Dice系数
        dice_scores = []
        for i in range(self.num_classes):
            if mask is not None:
                pred = inputs[:, i][mask[:, 0]]
                target = targets[:, i][mask[:, 0]]
            else:
                pred = inputs[:, i]
                target = targets[:, i]
            
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            
            if union > 0:
                dice = 2 * intersection / union
            else:
                dice = torch.tensor(1.0, device=inputs.device)
            
            if weight is not None:
                dice = dice * weight[i]
            
            dice_scores.append(dice)
        
        # 计算平均Dice系数
        avg_dice = torch.stack(dice_scores).mean()
        
        # 返回Dice损失和各类别的Dice系数
        return 1 - avg_dice, dice_scores

def parse_args():
    parser = argparse.ArgumentParser(description='Train hardness-based RWKV-UNet edge detection segmentation model')
    parser.add_argument('--root_path', type=str, default=r"C:\Users\admin\Desktop\MICCAI_BraTS2020_TrainingData",
                        help='Dataset root directory')
    parser.add_argument('--img_size', type=int, default=160, help='Image size')
    parser.add_argument('--img_channels', type=int, default=160, help='Image channels')
    parser.add_argument('--img_modals', type=int, default=4, help='Image modalities')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--max_epochs', type=int, default=400, help='Maximum training epochs')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of segmentation classes')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of data loading threads')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path to load')
    parser.add_argument('--exp_name', type=str, default='exp_edge', help='Experiment name')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    return parser.parse_args()

def worker_init_fn(worker_id):
    """Set random seed for worker threads"""
    random.seed(args.seed + worker_id)

def train_step(model, image_batch, label_batch, hardness_batch, weights, optimizer, dice_loss, args):
    """Training step"""
    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
    weights = weights.mean(dim=0).cuda()
    hardness_batch = hardness_batch.cuda()
    
    # Forward propagation
    outputs = model(
        image_batch.view(-1, args.img_channels * args.img_modals, args.img_size, args.img_size)
    ).view(-1, args.num_classes, args.img_channels, args.img_size, args.img_size)
    
    # Process labels
    valid_mask = (label_batch != -1)
    background_mask = torch.logical_not(valid_mask)
    label_no_background = label_batch.clone()
    label_no_background[background_mask] = 0
    
    # Calculate cross entropy loss (weighted by hardness)
    loss_ce = F.cross_entropy(outputs, label_no_background, weight=weights[1:], reduction='none')
    loss_ce = (loss_ce * (1 + hardness_batch))[valid_mask].mean()
    
    # Calculate cross entropy loss with background
    batch_size_actual = image_batch.shape[0]
    zeros_cache = torch.zeros(batch_size_actual, 1, args.img_channels, args.img_size, args.img_size).cuda()
    outputs_with_background = torch.cat([zeros_cache, outputs], dim=1)
    label_with_background = (label_batch + 1)
    loss_ce_with_background = F.cross_entropy(
        outputs_with_background, label_with_background, weight=weights, reduction='none').mean()
    
    # Calculate Dice loss
    loss_dice, dices = dice_loss(
        outputs, label_no_background, mask=valid_mask,
        softmax=True, one_hot=True, weight=weights[1:]
    )
    
    # Total loss
    loss = loss_ce + loss_dice + loss_ce_with_background
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return outputs, (loss, loss_ce, loss_dice, loss_ce_with_background), dices

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create experiment directory
    snapshot_path = f"./{args.exp_name}"
    os.makedirs(snapshot_path, exist_ok=True)
    
    # Create TensorBoard log
    writer = SummaryWriter(snapshot_path + '/log')
    
    # Create loss function
    dice_loss = DiceLoss(args.num_classes)
    
    # Create dataset
    dataset_config = BratsConfig()
    dataset_config.train_folder = args.root_path
    dataset_config.is_gz = False
    dataset_config.target_size = (args.img_channels, args.img_size, args.img_size)
    brats_dataset = get_brats_dataset(dataset_config)
    
    # Create data loader
    trainloader = DataLoader(
        brats_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    # Create model
    model = RWKV_UNet(
        args.img_modals * args.img_channels,
        args.num_classes * args.img_channels,
        args.img_size
    )
    
    # Load checkpoint (if available)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Move model to GPU
    model.float().cuda()
    model.train()
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr)
    
    # Set class weights
    weights = torch.tensor([0.1, 0.3, 0.3, 0.4])  # Background, necrotic core, edema, enhancing tumor
    
    # Training loop
    iter_num = 0
    max_iterations = args.max_epochs * len(trainloader)
    
    for epoch in range(args.max_epochs):
        print(f"Starting training epoch {epoch+1}/{args.max_epochs}...")
        
        for i_batch, sampled_batch in enumerate(trainloader):
            # Get data
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            hardness_batch = sampled_batch['hardness']
            
            # Training step
            outputs, losses, dices = train_step(
                model, image_batch, label_batch, hardness_batch,
                weights, optimizer, dice_loss, args
            )
            
            # Record losses
            loss, loss_ce, loss_dice, loss_ce_with_background = losses
            writer.add_scalar('loss/total', loss.item(), iter_num)
            writer.add_scalar('loss/ce', loss_ce.item(), iter_num)
            writer.add_scalar('loss/dice', loss_dice.item(), iter_num)
            writer.add_scalar('loss/ce_with_bg', loss_ce_with_background.item(), iter_num)
            
            # Record Dice coefficients
            for i, dice in enumerate(dices):
                writer.add_scalar(f'metrics/dice_class{i+1}', dice.item(), iter_num)
            
            # Print progress
            if iter_num % 10 == 0:
                print(
                    f"Iteration {iter_num}/{max_iterations}, Loss: {loss.item():.4f}, "
                    f"Dice: {[d.item():.4f for d in dices]}"
                )
            
            iter_num += 1
        
        # Save model
        save_path = os.path.join(snapshot_path, f'epoch_{epoch}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()