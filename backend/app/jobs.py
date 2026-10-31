"""A tiny single-worker job queue.

Generation is serialized because there is one model on one accelerator. Jobs
and their status live in memory; the output mp4 lives on disk under OUTPUT_DIR.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import comfy, config, model

_executor = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()
_jobs: "Dict[str, Job]" = {}


@dataclass
class Job:
    id: str
    mode: str  # "t2v" | "i2v"
    params: Dict[str, Any]
    status: str = "queued"  # queued | running | done | error
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    output_file: Optional[str] = None  # basename under OUTPUT_DIR

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "status": self.status,
            "params": self.params,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "logs": self.logs[-40:],
            "error": self.error,
            "output_file": self.output_file,
            "video_url": f"/api/outputs/{self.output_file}" if self.output_file else None,
        }


def _log(job: Job, msg: str) -> None:
    with _lock:
        job.logs.append(f"{time.strftime('%H:%M:%S')}  {msg}")


def _run(job: Job) -> None:
    with _lock:
        job.status = "running"
        job.started_at = time.time()
    _log(job, "Job started.")
    try:
        out_name = f"{job.id}.mp4"
        out_path = config.OUTPUT_DIR / out_name
        p = job.params
        common = dict(
            prompt=p["prompt"],
            num_frames=p["num_frames"],
            num_inference_steps=p["num_inference_steps"],
            seed=p.get("seed"),
            width=p.get("width"),
            height=p.get("height"),
            image_path=p.get("image_path"),
            last_image_path=p.get("last_image_path"),
            output_path=out_path,
            on_log=lambda m: _log(job, m),
        )
        if config.BACKEND == "comfy":
            comfy.generate(mode=job.mode, **common)
        else:
            model.generate(**common)
        with _lock:
            job.output_file = out_name
            job.status = "done"
            job.finished_at = time.time()
        _log(job, "Job finished OK.")
    except Exception as exc:  # noqa: BLE001 - surface everything to the UI
        with _lock:
            job.status = "error"
            job.error = str(exc)
            job.finished_at = time.time()
        _log(job, "ERROR: " + str(exc))
        _log(job, traceback.format_exc())


def submit(mode: str, params: Dict[str, Any]) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], mode=mode, params=params)
    with _lock:
        _jobs[job.id] = job
    _executor.submit(_run, job)
    return job


def get(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def recent(limit: int = 50) -> List[Job]:
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
    return jobs[:limit]
