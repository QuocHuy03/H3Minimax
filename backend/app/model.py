"""MiniMax-H3 inference wrapper.

The model is integrated in diffusers as Modular Diffusers blocks (there is no
plain ``DiffusionPipeline``). One repository holds two transformer partitions:

  * ``transformer/``     -> ``t2va`` (text only) and ``fl2va`` (first/last keyframe)
  * ``transformer_ref/`` -> ``ref2va`` (omni references, not used by this UI)

We create the pipeline WITHOUT a fixed workflow so it picks ``t2va`` vs ``fl2va``
per call from the inputs (a bare ``prompt`` -> t2va; ``prompt`` + ``image`` ->
fl2va). Loading ``workflow="t2va"`` fetches only ``transformer/`` + the shared
components, which is enough to serve both of our tasks.

Everything heavy is imported lazily inside ``_load`` so the API process and the
MOCK path never touch torch/diffusers.
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from . import config

# The pipeline object once loaded (real mode). Guarded by _load_lock.
_pipe = None
_load_lock = threading.Lock()


def snap_num_frames(num_frames: int) -> int:
    """Snap a frame count to the next ``17 * n + 5`` the video VAE can decode,
    clamped to the model's 5-15 s window."""
    num_frames = max(config.MIN_FRAMES, min(config.MAX_FRAMES, int(num_frames)))
    n = math.ceil((num_frames - 5) / 17)
    snapped = 17 * n + 5
    # Keep inside the window after snapping up.
    while snapped > config.MAX_FRAMES:
        n -= 1
        snapped = 17 * n + 5
    return snapped


def seconds_to_frames(seconds: float) -> int:
    return snap_num_frames(round(seconds * config.FPS))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load():
    """Load the real pipeline once. Returns the cached pipeline on later calls."""
    global _pipe
    if _pipe is not None:
        return _pipe

    with _load_lock:
        if _pipe is not None:
            return _pipe

        import torch  # noqa: WPS433 (lazy heavy import)
        from diffusers import ComponentsManager, ModularPipeline

        manager = ComponentsManager()

        if config.DEVICE_MODE == "consumer":
            _pipe = _load_consumer(torch, ModularPipeline)
        else:
            pipe = ModularPipeline.from_pretrained(
                config.MODEL_ID, components_manager=manager
            )
            pipe.load_components(workflow="t2va", dtype=torch.bfloat16)
            manager.enable_auto_cpu_offload(
                device=config.DEVICE,
                memory_reserve_margin=config.MEMORY_RESERVE_MARGIN,
            )
            _pipe = pipe

        return _pipe


def _load_consumer(torch, ModularPipeline):
    """int8 weight-only quant + group offload for 24-32GB cards (~75GB host RAM)."""
    from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
    from diffusers.hooks import apply_group_offloading
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    pipe = ModularPipeline.from_pretrained(config.MODEL_ID)
    pipe.update_components(
        transformer=MiniMaxH3Transformer3DModel.from_pretrained(
            config.MODEL_ID,
            subfolder="transformer",
            dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "proj_in", "audio_proj_in", "context_embedder",
                    "time_embedder", "time_proj", "token_refiner",
                    "norm_out", "proj_out", "audio_proj_out",
                ],
            ),
            low_cpu_mem_usage=False,
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            config.MODEL_ID,
            subfolder="text_encoder",
            dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "model.visual",
                    "model.language_model.embed_tokens",
                    "model.language_model.norm",
                    "lm_head",
                ],
            ),
        ),
    )
    pipe.load_components(workflow="t2va", dtype=torch.bfloat16)

    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    offload = dict(
        onload_device=torch.device(config.DEVICE),
        offload_device=torch.device("cpu"),
        use_stream=True,
    )
    pipe.transformer.enable_group_offload(
        offload_type="block_level", num_blocks_per_group=1, **offload
    )
    apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
    pipe.vae.to(config.DEVICE)
    pipe.audio_vae.to(config.DEVICE)
    return pipe


def warmup():
    """Eagerly load the model (real mode). No-op in MOCK mode."""
    if config.MOCK:
        return
    _load()


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate(
    *,
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
    """Run one generation and write an mp4 (with soundtrack) to ``output_path``.

    A bare prompt runs the ``t2va`` workflow; a prompt plus a first/last keyframe
    runs ``fl2va``.
    """
    if config.MOCK:
        return _generate_mock(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            image_path=image_path,
            output_path=output_path,
            on_log=on_log,
        )

    import torch
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    pipe = _load()
    on_log("Model ready, building request...")

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))

    kwargs = dict(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output=["videos", "audio", "sampling_rate"],
    )
    if width:
        kwargs["width"] = width
    if height:
        kwargs["height"] = height
    if image_path:
        kwargs["image"] = load_image(image_path)
    if last_image_path:
        kwargs["last_image"] = load_image(last_image_path)

    on_log(f"Denoising ({num_inference_steps} steps, {num_frames} frames)...")
    t0 = time.time()
    results = pipe(**kwargs)
    on_log(f"Denoised in {time.time() - t0:.1f}s, muxing video + audio...")

    encode_video(
        results["videos"][0],
        fps=config.FPS,
        output_path=str(output_path),
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    on_log("Done.")
    return output_path


def _generate_mock(
    *,
    prompt: str,
    num_frames: int,
    width: Optional[int],
    height: Optional[int],
    image_path: Optional[str],
    output_path: Path,
    on_log: Callable[[str], None],
) -> Path:
    """Synthesize a short animated placeholder clip (no model, no GPU).

    Lets you exercise the full UI/job/playback flow. Uses imageio+ffmpeg if
    available; otherwise raises a clear error telling you how to enable it.
    """
    on_log("[MOCK] Rendering placeholder clip (set H3_MOCK=0 for the real model)...")
    try:
        import numpy as np
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "MOCK mode needs numpy + imageio[ffmpeg]: "
            "pip install numpy imageio imageio-ffmpeg"
        ) from exc

    w = int(width or 960)
    h = int(height or 544)
    # Keep the mock snappy regardless of requested duration.
    frames_to_make = max(24, min(num_frames, 120))

    base = None
    if image_path:
        try:
            img = imageio.imread(image_path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img = img[..., :3]
            from PIL import Image

            pil = Image.fromarray(img).resize((w, h))
            base = np.asarray(pil, dtype=np.float32)
        except Exception:
            base = None

    writer = imageio.get_writer(
        str(output_path), fps=config.FPS, codec="libx264",
        quality=7, macro_block_size=None,
    )
    try:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for i in range(frames_to_make):
            t = i / max(1, frames_to_make - 1)
            if base is not None:
                shade = 0.65 + 0.35 * math.sin(2 * math.pi * t)
                frame = np.clip(base * shade, 0, 255).astype(np.uint8)
            else:
                r = (0.5 + 0.5 * np.sin(2 * math.pi * (t + xx / w))) * 255
                g = (0.5 + 0.5 * np.sin(2 * math.pi * (t + yy / h + 0.33))) * 255
                b = np.full_like(r, (0.5 + 0.5 * math.sin(2 * math.pi * (t + 0.66))) * 255)
                frame = np.stack([r, g, b], axis=-1).astype(np.uint8)
            writer.append_data(frame)
            if i % 12 == 0:
                on_log(f"[MOCK] frame {i + 1}/{frames_to_make}")
    finally:
        writer.close()

    on_log("[MOCK] Done.")
    return output_path
