import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset import BratsDataset, config_from_yaml
from scripts.train_student import SliceWiseRWKVUNet3D
from utils.repro_config import load_config, load_split_cases, set_reproducible_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Diet-Seg sliding-window inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    fold = args.fold if args.fold is not None else int(config["experiment"]["fold_index"])
    set_reproducible_seed(int(config["experiment"]["seed"]), bool(config["experiment"]["deterministic"]))

    if config["inference"]["test_time_augmentation"]:
        raise ValueError("Reproducibility config disables test-time augmentation")
    if config["inference"]["post_processing"] != "none":
        raise ValueError("Reproducibility config disables model-specific post-processing")

    split_file = Path(config["data"]["split_file"])
    if not split_file.is_absolute():
        split_file = Path(args.config).resolve().parent.parent / split_file
    cases = load_split_cases(split_file, fold, args.split)
    data_cfg = config_from_yaml(config, cases, training=False)
    data_cfg.hardness_dir = None
    dataset = BratsDataset(data_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SliceWiseRWKVUNet3D(
        int(config["model"]["in_channels"]),
        int(config["model"]["out_channels"]),
        int(config["model"]["img_size"]),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
    model.eval()

    output_dir = Path(args.output_dir or f"predictions/{config['experiment']['dataset'].lower()}/fold{fold}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from monai.inferers import sliding_window_inference
    except ImportError as exc:
        raise ImportError("MONAI is required for sliding-window inference") from exc

    roi_size = tuple(config["inference"]["roi_size"])
    overlap = float(config["inference"]["overlap"])
    for sample in dataset:
        image = sample["image"].unsqueeze(0).to(device)
        case_id = sample["patient_id"]
        with torch.no_grad():
            logits = sliding_window_inference(
                image,
                roi_size=roi_size,
                sw_batch_size=int(config["inference"]["sw_batch_size"]),
                predictor=model,
                overlap=overlap,
                mode=config["inference"]["mode"],
            )
            regions = (torch.sigmoid(logits)[0].cpu().numpy() > 0.5).astype(np.uint8)

        output = np.zeros(regions.shape[1:], dtype=np.uint8)
        output[regions[1] > 0] = 2
        output[regions[2] > 0] = 1
        output[regions[0] > 0] = 4
        nib.save(nib.Nifti1Image(output, np.eye(4)), output_dir / f"{case_id}.nii.gz")

    print(f"saved predictions to {output_dir}")


if __name__ == "__main__":
    main()
