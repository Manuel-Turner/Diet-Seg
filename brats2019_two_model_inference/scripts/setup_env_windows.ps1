$ErrorActionPreference = "Stop"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

Write-Host ""
Write-Host "If the CUDA install fails, use the official PyTorch selector to choose the command for your driver/CUDA setup:"
Write-Host "https://pytorch.org/get-started/locally/"
