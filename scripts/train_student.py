import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset import BratsDataset, config_from_yaml
from models.rwkv_unet_edge import RWKV_UNet
from utils.entropy_hardness import hardness_weight
from utils.repro_config import create_adamw, format_fold_path, load_config, load_split_cases, set_reproducible_seed


class SliceWiseRWKVUNet3D(torch.nn.Module):
    """Apply the 2D RWKV-UNet+EdgeNet backbone slice-wise to 128^3 BraTS patches."""

    def __init__(self, in_channels: int, out_channels: int, img_size: int):
        super().__init__()
        self.backbone = RWKV_UNet(in_channels, out_channels, img_size)

    def forward(self, x):
        batch, channels, depth, height, width = x.shape
        slices = x.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width)
        logits = self.backbone(slices)
        return logits.reshape(batch, depth, -1, height, width).permute(0, 2, 1, 3, 4)


def dice_loss(logits, targets, eps=1.0e-6):
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3, 4)
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def hardness_weighted_region_ce(logits, targets, hardness):
    weights = hardness_weight(hardness).unsqueeze(1).to(logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * weights).mean()


def mean_dice(logits, targets, eps=1.0e-6):
    preds = (torch.sigmoid(logits) > 0.5).float()
    dims = (0, 2, 3, 4)
    intersection = (preds * targets).sum(dim=dims)
    denominator = preds.sum(dim=dims) + targets.sum(dim=dims)
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diet-Seg RWKV-UNet student from YAML config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    return parser.parse_args()


def build_loaders(config, config_path, fold):
    split_file = Path(config["data"]["split_file"])
    if not split_file.is_absolute():
        split_file = Path(config_path).resolve().parent.parent / split_file
    train_cases = load_split_cases(split_file, fold, "train")
    val_cases = load_split_cases(split_file, fold, "val")

    hardness_dir = format_fold_path(config["hardness"]["save_dir"], fold)
    train_cfg = config_from_yaml(config, train_cases, training=True)
    train_cfg.hardness_dir = hardness_dir
    val_cfg = config_from_yaml(config, val_cases, training=False)
    val_cfg.hardness_dir = None

    train_loader = DataLoader(
        BratsDataset(train_cfg),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        BratsDataset(val_cfg),
        batch_size=1,
        shuffle=False,
        num_workers=max(1, int(config["training"]["num_workers"]) // 2),
        pin_memory=True,
    )
    return train_loader, val_loader


def main():
    args = parse_args()
    config = load_config(args.config)
    fold = args.fold if args.fold is not None else int(config["experiment"]["fold_index"])
    set_reproducible_seed(int(config["experiment"]["seed"]), bool(config["experiment"]["deterministic"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SliceWiseRWKVUNet3D(
        int(config["model"]["in_channels"]),
        int(config["model"]["out_channels"]),
        int(config["model"]["img_size"]),
    ).to(device)
    optimizer = create_adamw(model, config)
    scaler = GradScaler(enabled=bool(config["training"]["amp"]) and device.type == "cuda")
    train_loader, val_loader = build_loaders(config, args.config, fold)

    checkpoint_dir = Path(args.checkpoint_dir or f"checkpoints/student/{config['experiment']['dataset'].lower()}/fold{fold}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0

    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            hardness = batch["hardness"].to(device, non_blocking=True).detach()

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                logits = model(images)
                loss = hardness_weighted_region_ce(logits, labels, hardness) + dice_loss(logits, labels)
            scaler.scale(loss).backward()
            if config["training"]["gradient_clip_norm"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()

        if (epoch + 1) % int(config["training"]["validation_interval"]) == 0:
            model.eval()
            dices = []
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device, non_blocking=True)
                    labels = batch["label"].to(device, non_blocking=True)
                    with autocast(enabled=scaler.is_enabled()):
                        dices.append(float(mean_dice(model(images), labels)))
            val_dice = sum(dices) / max(len(dices), 1)
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({"model": model.state_dict(), "config": config, "fold": fold}, checkpoint_dir / "best.pt")
                print(f"epoch {epoch + 1}: saved best checkpoint, mean_dice={val_dice:.6f}")


if __name__ == "__main__":
    main()
