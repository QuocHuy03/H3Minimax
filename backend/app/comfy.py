"""Drive a running ComfyUI instance via its HTTP API.

The UI sends plain params (prompt, frames, steps, seed, images). This module:
  1. loads an exported ComfyUI **API-format** workflow for the mode,
  2. patches the chosen nodes with those params (and uploads any images),
  3. queues it, waits for completion, and downloads the produced video.

You never hand-edit the graph here — you describe, once, *which* node fields to
override in ``workflows/mapping.json``. That keeps us decoupled from the exact
graph (which node does what), so your turbo/quantized H3 workflow works as-is.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

from . import config


class ComfyError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Workflow + mapping loading
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_workflow(mode: str) -> Dict[str, Any]:
    """Load the API-format workflow for ``mode`` (falls back to workflow.json)."""
    candidates = [
        config.WORKFLOW_DIR / f"{mode}.json",
        config.WORKFLOW_DIR / "workflow.json",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise ComfyError(
            f"No workflow file. Put your ComfyUI API-format JSON at "
            f"{config.WORKFLOW_DIR / f'{mode}.json'} (or workflow.json)."
        )
    wf = _read_json(path)
    # UI-format export has a top-level "nodes" array; the API needs the flat dict.
    if isinstance(wf, dict) and "nodes" in wf and isinstance(wf["nodes"], list):
        raise ComfyError(
            f"{path.name} looks like a UI export, not API format. In ComfyUI enable "
            "Dev mode, then use 'Save (API Format)'."
        )
    if not isinstance(wf, dict) or not all(isinstance(v, dict) and "class_type" in v for v in wf.values()):
        raise ComfyError(f"{path.name} is not a valid ComfyUI API workflow.")
    return wf


def load_mapping() -> Dict[str, Any]:
    path = config.WORKFLOW_DIR / "mapping.json"
    if not path.is_file():
        raise ComfyError(
            f"Missing {path}. Copy mapping.example.json to mapping.json and fill in "
            "the node ids/titles for your workflow."
        )
    return _read_json(path)


def _resolve_node_id(workflow: Dict[str, Any], spec: Dict[str, Any]) -> Optional[str]:
    """A spec targets a node by "node" (id) or by "title" (_meta.title)."""
    if "node" in spec:
        nid = str(spec["node"])
        return nid if nid in workflow else None
    if "title" in spec:
        for nid, node in workflow.items():
            if node.get("_meta", {}).get("title") == spec["title"]:
                return nid
    return None


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def ping() -> bool:
    try:
        r = requests.get(f"{config.COMFY_URL}/system_stats", timeout=4)
        return r.ok
    except requests.RequestException:
        return False


def _upload_image(path: str) -> str:
    p = Path(path)
    with p.open("rb") as f:
        files = {"image": (p.name, f, "application/octet-stream")}
        r = requests.post(
            f"{config.COMFY_URL}/upload/image",
            files=files,
            data={"overwrite": "true"},
            timeout=30,
        )
    if not r.ok:
        raise ComfyError(f"Image upload failed: {r.status_code} {r.text}")
    data = r.json()
    name = data.get("name")
    sub = data.get("subfolder")
    return f"{sub}/{name}" if sub else name


def _queue(workflow: Dict[str, Any], client_id: str) -> str:
    r = requests.post(
        f"{config.COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    if r.status_code != 200:
        raise ComfyError(f"Queue rejected ({r.status_code}): {r.text}")
    data = r.json()
    if data.get("node_errors"):
        raise ComfyError(f"Workflow node errors: {json.dumps(data['node_errors'])[:800]}")
    return data["prompt_id"]


def _wait(prompt_id: str, on_log: Callable[[str], None], timeout_s: int = 1800) -> Dict[str, Any]:
    t0 = time.time()
    last = -1.0
    while True:
        if time.time() - t0 > timeout_s:
            raise ComfyError("Timed out waiting for ComfyUI.")
        try:
            r = requests.get(f"{config.COMFY_URL}/history/{prompt_id}", timeout=10)
            if r.ok and prompt_id in r.json():
                return r.json()[prompt_id]
        except requests.RequestException:
            pass
        now = time.time()
        if now - last >= 5:
            on_log(f"Waiting on ComfyUI… {int(now - t0)}s")
            last = now
        time.sleep(1.5)


_VIDEO_EXT = (".mp4", ".webm", ".mkv", ".mov", ".gif")


def _find_output(history: Dict[str, Any], output_node: Optional[str]) -> Dict[str, str]:
    """Return {filename, subfolder, type} of the produced video from history outputs."""
    outputs = history.get("outputs", {})
    nodes = [output_node] if output_node and output_node in outputs else list(outputs.keys())
    best: Optional[Dict[str, str]] = None
    for nid in nodes:
        for items in outputs[nid].values():  # 'gifs' / 'videos' / 'images' lists
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict) or "filename" not in it:
                    continue
                fn = it["filename"].lower()
                cand = {
                    "filename": it["filename"],
                    "subfolder": it.get("subfolder", ""),
                    "type": it.get("type", "output"),
                }
                if fn.endswith(_VIDEO_EXT):
                    return cand  # a real video wins immediately
                best = best or cand  # fall back to an image if nothing else
    if best is None:
        raise ComfyError("ComfyUI finished but produced no downloadable file.")
    return best


def _download(ref: Dict[str, str], dest: Path) -> None:
    r = requests.get(
        f"{config.COMFY_URL}/view",
        params={"filename": ref["filename"], "subfolder": ref["subfolder"], "type": ref["type"]},
        timeout=120,
    )
    if not r.ok:
        raise ComfyError(f"Download failed: {r.status_code}")
    dest.write_bytes(r.content)


# --------------------------------------------------------------------------- #
# Public entrypoint (same shape as model.generate)
# --------------------------------------------------------------------------- #
def generate(
    *,
    mode: str,
    prompt: str,
    num_frames: int,
    num_inference_steps: int,
    seed: Optional[int],
    width: Optional[int],
    height: Optional[int],
    image_path: Optional[str],
    last_image_path: Optional[str],
    output_path: Path,
    on_log: Callable[[str], None] = lambda _m: None,
) -> Path:
    if not ping():
        raise ComfyError(f"ComfyUI not reachable at {config.COMFY_URL}. Is it running?")

    workflow = load_workflow(mode)
    mapping = load_mapping()
    conf = mapping.get(mode) or mapping.get("default") or {}

    values: Dict[str, Any] = {
        "prompt": prompt,
        "seed": seed if seed is not None else uuid.uuid4().int % (2**31),
        "steps": num_inference_steps,
        "num_frames": num_frames,
        "frames": num_frames,
        "width": width,
        "height": height,
    }

    # 1) scalar overrides
    for spec in conf.get("set", []):
        nid = _resolve_node_id(workflow, spec)
        if nid is None:
            on_log(f"[warn] node not found for {spec}; skipping")
            continue
        raw = spec.get("value")
        val = values.get(raw[1:], raw) if isinstance(raw, str) and raw.startswith("$") else raw
        if val is None:
            continue
        workflow[nid]["inputs"][spec["field"]] = val

    # 2) image inputs (upload then set the node field to the uploaded name)
    chosen = {"first": image_path, "last": last_image_path}
    for spec in conf.get("images", []):
        src = chosen.get(spec.get("from", "first"))
        if not src:
            continue
        nid = _resolve_node_id(workflow, spec)
        if nid is None:
            on_log(f"[warn] image node not found for {spec}; skipping")
            continue
        name = _upload_image(src)
        on_log(f"Uploaded {spec.get('from')} frame -> {name}")
        workflow[nid]["inputs"][spec["field"]] = name

    client_id = uuid.uuid4().hex
    on_log("Queuing workflow on ComfyUI…")
    prompt_id = _queue(workflow, client_id)
    on_log(f"Queued (prompt_id={prompt_id}).")

    history = _wait(prompt_id, on_log)
    ref = _find_output(history, conf.get("output_node"))
    on_log(f"Downloading {ref['filename']}…")
    _download(ref, output_path)
    on_log("Done.")
    return output_path
