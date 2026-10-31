#!/usr/bin/env bash
# Install REAL-mode inference deps into .venv (run on the GPU machine).
# Usage:  ./setup-real.sh [cuda]     e.g. ./setup-real.sh cu124
set -euo pipefail
cd "$(dirname "$0")"
CUDA="${1:-cu124}"   # match your CUDA toolkit: cu121 / cu124 / cu128 ...

[ -d .venv ] || python3 -m venv .venv
PY=./.venv/bin/python

$PY -m pip install --upgrade pip
$PY -m pip install -r requirements.txt
$PY -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA}"
$PY -m pip install "git+https://github.com/huggingface/diffusers.git"
$PY -m pip install "transformers>=4.57" accelerate av
$PY -m pip install torchao   # only for H3_DEVICE_MODE=consumer

echo ""
echo "Done. Set H3_MOCK=0 in .env, then run ./run.sh"
echo "Self-check (downloads weights, slow):"
echo "  H3_MOCK=0 $PY -c 'from app import model; model.warmup(); print(\"pipeline loaded OK\")'"
