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
from scripts.train_student import dice_loss, mean_dice
from utils.repro_config import create_adamw, format_fold_path, load_config, load_split_cases, set_reproducible_seed


def build_teacher(in_channels, out_channels):
    try:
        from monai.networks.nets import UNet

        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    except ImportError as exc:
        raise ImportError("MONAI is required for the configured 3D U-Net teacher") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Train fold-wise 3D U-Net teacher from YAML config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    fold = args.fold if args.fold is not None else int(config["experiment"]["fold_index"])
    set_reproducible_seed(int(config["experiment"]["seed"]), bool(config["experiment"]["deterministic"]))

    split_file = Path(config["data"]["split_file"])
    if not split_file.is_absolute():
        split_file = Path(args.config).resolve().parent.parent / split_file
    train_cases = load_split_cases(split_file, fold, "train")
    val_cases = load_split_cases(split_file, fold, "val")

    train_cfg = config_from_yaml(config, train_cases, training=True)
    train_cfg.hardness_dir = None
    val_cfg = config_from_yaml(config, val_cases, training=False)
    val_cfg.hardness_dir = None

    train_loader = DataLoader(
        BratsDataset(train_cfg),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(BratsDataset(val_cfg), batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_teacher(int(config["model"]["in_channels"]), int(config["model"]["out_channels"])).to(device)
    optimizer = create_adamw(model, config)
    scaler = GradScaler(enabled=bool(config["training"]["amp"]) and device.type == "cuda")

    checkpoint_dir = Path(format_fold_path(config["teacher"]["checkpoint_dir"], fold))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0

    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                logits = model(images)
                loss = F.binary_cross_entropy_with_logits(logits, labels) + dice_loss(logits, labels)
            scaler.scale(loss).backward()
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
                print(f"epoch {epoch + 1}: saved best teacher checkpoint, mean_dice={val_dice:.6f}")


if __name__ == "__main__":
    main()
