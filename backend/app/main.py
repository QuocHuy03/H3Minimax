"""FastAPI entrypoint for the MiniMax-H3 video UI backend."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import comfy, config, jobs, model

app = FastAPI(title="MiniMax-H3 Video UI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

_ALLOWED_IMAGE = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def get_config():
    info = {
        "backend": config.BACKEND,
        "mock": config.MOCK,
        "fps": config.FPS,
        "min_frames": config.MIN_FRAMES,
        "max_frames": config.MAX_FRAMES,
    }
    if config.BACKEND == "comfy":
        info["comfy_url"] = config.COMFY_URL
        info["comfy_online"] = comfy.ping()
    else:
        info["model_id"] = config.MODEL_ID
        info["device_mode"] = config.DEVICE_MODE
        info["device"] = config.DEVICE
    return info


def _save_upload(upload: UploadFile) -> str:
    if upload.content_type not in _ALLOWED_IMAGE:
        raise HTTPException(400, f"Unsupported image type: {upload.content_type}")
    suffix = Path(upload.filename or "").suffix or ".png"
    dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(dest)


@app.post("/api/generate")
async def generate(
    mode: str = Form(...),  # "t2v" | "i2v"
    prompt: str = Form(...),
    num_frames: int = Form(124),
    num_inference_steps: int = Form(50),
    seed: Optional[int] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    image: Optional[UploadFile] = File(None),
    last_image: Optional[UploadFile] = File(None),
):
    if mode not in {"t2v", "i2v"}:
        raise HTTPException(400, "mode must be 't2v' or 'i2v'")
    if not prompt.strip():
        raise HTTPException(400, "prompt is required")
    if mode == "i2v" and image is None and last_image is None:
        raise HTTPException(400, "image-to-video needs a first and/or last frame")

    params = {
        "prompt": prompt.strip(),
        "num_frames": model.snap_num_frames(num_frames),
        "num_inference_steps": max(1, min(100, int(num_inference_steps))),
        "seed": int(seed) if seed is not None else None,
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "image_path": _save_upload(image) if (mode == "i2v" and image) else None,
        "last_image_path": _save_upload(last_image) if (mode == "i2v" and last_image) else None,
    }

    job = jobs.submit(mode, params)
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": [j.to_dict() for j in jobs.recent()]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.to_dict()


@app.get("/api/outputs/{filename}")
def get_output(filename: str):
    # Prevent path traversal: only a bare basename is allowed.
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "bad filename")
    path = config.OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)
