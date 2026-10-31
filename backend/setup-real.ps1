# Install REAL-mode inference deps into .venv (run on the GPU machine).
# Usage:  .\setup-real.ps1  [-Cuda cu124]
param([string]$Cuda = "cu124")   # match your CUDA toolkit: cu121 / cu124 / cu128 ...

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) { python -m venv .venv }
$py = ".venv\Scripts\python.exe"

& $py -m pip install --upgrade pip

# 1) API + mock deps.
& $py -m pip install -r requirements.txt

# 2) torch matching your CUDA (see https://pytorch.org for the right index-url).
& $py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"

# 3) MiniMax-H3 lives on diffusers `main` for now.
& $py -m pip install "git+https://github.com/huggingface/diffusers.git"
& $py -m pip install "transformers>=4.57" accelerate av

# 4) Only for H3_DEVICE_MODE=consumer (int8 on 24-32GB cards):
& $py -m pip install torchao

Write-Host ""
Write-Host "Done. Now set H3_MOCK=0 in .env, then run:  .\run.ps1" -ForegroundColor Green
Write-Host "Quick self-check of the real pipeline (downloads weights, slow):" -ForegroundColor Cyan
Write-Host "  `$env:H3_MOCK='0'; $py -c `"from app import model; model.warmup(); print('pipeline loaded OK')`""
