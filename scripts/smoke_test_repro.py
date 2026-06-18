import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_student import SliceWiseRWKVUNet3D
from utils.entropy_hardness import compute_entropy_hardness, hardness_weight
from utils.repro_config import create_adamw, load_config


def assert_close(actual, expected, name):
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def main():
    configs = [
        ROOT / "configs" / "dietseg_brats2019_repro.yaml",
        ROOT / "configs" / "dietseg_brats2020_repro.yaml",
    ]
    parsed = [load_config(path) for path in configs]
    print("YAML configs parsed and schema validated: OK")

    config = parsed[0]
    model = SliceWiseRWKVUNet3D(
        int(config["model"]["in_channels"]),
        int(config["model"]["out_channels"]),
        int(config["model"]["img_size"]),
    )
    print("RWKV-UNet+EdgeNet student instantiated: OK")

    optimizer = create_adamw(model, config)
    group = optimizer.param_groups[0]
    assert_close(group["lr"], 1.0e-4, "AdamW lr")
    assert_close(group["betas"], (0.9, 0.999), "AdamW betas")
    assert_close(group["eps"], 1.0e-8, "AdamW eps")
    assert_close(group["weight_decay"], 1.0e-5, "AdamW weight_decay")
    print("AdamW parameter values verified: OK")

    probabilities = torch.rand(3, 4, 5, 6)
    probabilities = probabilities / probabilities.sum(dim=0, keepdim=True)
    hardness = compute_entropy_hardness(probabilities)
    if float(hardness.min()) < 0.0 or float(hardness.max()) > 1.0:
        raise AssertionError("entropy hardness out of [0, 1]")
    print("Entropy hardness output range [0, 1]: OK")

    weight = hardness_weight(hardness)
    if not torch.allclose(weight, 1.0 + hardness):
        raise AssertionError("hardness weight must be 1+h")
    print("Hardness weight form 1+h verified: OK")

    assert_close(config["inference"]["roi_size"], [128, 128, 128], "inference roi_size")
    assert_close(float(config["inference"]["overlap"]), 0.5, "inference overlap")
    assert_close(config["inference"]["test_time_augmentation"], False, "TTA")
    assert_close(config["inference"]["post_processing"], "none", "post_processing")
    print("Sliding-window inference config verified: OK")


if __name__ == "__main__":
    main()
