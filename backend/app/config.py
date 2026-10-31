"""Runtime configuration, read from environment variables.

Nothing here imports torch/diffusers, so it stays cheap to import.
"""
from __future__ import annotations

import os
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# Where generated videos and uploads live.
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.getenv("H3_OUTPUT_DIR", BASE_DIR / "outputs"))
UPLOAD_DIR = Path(os.getenv("H3_UPLOAD_DIR", BASE_DIR / "uploads"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Which inference engine the jobs use:
#   "comfy"     -> drive a running ComfyUI instance via its HTTP API (light; runs the
#                  quantized hybrid model + turbo 8-step LoRA already set up in ComfyUI).
#   "diffusers" -> load MiniMax-H3 locally with diffusers (full bf16, heavy).
BACKEND = os.getenv("H3_BACKEND", "comfy").strip().lower()

# ComfyUI server (for BACKEND=comfy).
COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")

# Folder holding the exported ComfyUI API workflow(s) + mapping.json.
WORKFLOW_DIR = Path(os.getenv("H3_WORKFLOW_DIR", BASE_DIR / "workflows"))
WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face repo id for the weights (BACKEND=diffusers).
MODEL_ID = os.getenv("H3_MODEL_ID", "MiniMaxAI/MiniMax-H3")

# Real generation by default. H3_MOCK=1 is an opt-in that synthesizes a placeholder
# clip (no model, no GPU) purely for UI smoke-testing on a machine without a GPU.
MOCK = _bool("H3_MOCK", default=False)

# How to place the model on hardware when MOCK is off:
#   "auto"     -> single accelerator + ComponentsManager auto CPU offload (needs ~80GB VRAM).
#   "consumer" -> int8 quantization + block/leaf group offload (24-32GB VRAM, ~75GB host RAM).
DEVICE_MODE = os.getenv("H3_DEVICE_MODE", "auto").strip().lower()

# Accelerator device string passed to diffusers ("cuda", "mps", "xpu", "cpu").
DEVICE = os.getenv("H3_DEVICE", "cuda")

# Reserve margin for the "auto" ComponentsManager offloader.
MEMORY_RESERVE_MARGIN = os.getenv("H3_MEMORY_RESERVE_MARGIN", "12GB")

# CORS origins for the Vite dev server.
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "H3_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if o.strip()
]

# Generation limits (24 fps, 5-15 s per the model card).
FPS = 24
MIN_FRAMES = 120  # ~5 s
MAX_FRAMES = 360  # ~15 s
