#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

echo ""
echo "If the CUDA install fails, use the official PyTorch selector to choose the command for your driver/CUDA setup:"
echo "https://pytorch.org/get-started/locally/"
