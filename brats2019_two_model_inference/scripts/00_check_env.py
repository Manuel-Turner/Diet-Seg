from __future__ import annotations

import platform
import subprocess
import sys

from common import ensure_project_dirs, executable_available, log_text


def optional_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return "not installed"


def command_status(command):
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=15)
        if completed.returncode == 0:
            first_line = (completed.stdout or completed.stderr).strip().splitlines()
            return "available" + (f" ({first_line[0]})" if first_line else "")
        return f"found but returned {completed.returncode}"
    except FileNotFoundError:
        return "not found"
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"


def main() -> int:
    ensure_project_dirs()
    lines = []
    lines.append(f"Python version: {platform.python_version()} ({sys.executable})")

    try:
        import torch

        lines.append(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        lines.append(f"CUDA available: {cuda_available}")
        lines.append(f"GPU name: {torch.cuda.get_device_name(0) if cuda_available else 'none'}")
    except Exception as exc:
        lines.append(f"PyTorch version: not installed ({type(exc).__name__}: {exc})")
        lines.append("CUDA available: unavailable")
        lines.append("GPU name: unavailable")

    lines.append(f"Docker availability: {command_status(['docker', '--version'])}")
    lines.append(f"nvidia-smi availability: {command_status(['nvidia-smi'])}")
    lines.append(f"Kaggle API availability: {command_status(['kaggle', '--version']) if executable_available('kaggle') else 'not found'}")
    lines.append(f"MONAI availability: {optional_version('monai')}")

    if not executable_available("docker"):
        lines.append(
            "KAIST Docker inference cannot run until Docker and NVIDIA Container Toolkit are available."
        )

    text = "\n".join(lines)
    print(text)
    log_text("00_check_env.log", text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
