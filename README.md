# MiniMax-H3 Video Studio

A simple self-hosted UI for the open-weight **MiniMax-H3 (Hailuo 3.0)** model —
text→video and image→video, with native stereo audio.

- **Backend**: FastAPI, lazy model load, single-worker job queue, serves the mp4.
- **Frontend**: React + Vite + TypeScript. Two tabs (T2V / I2V), live job status, preview + download.
- **Runs the real model by default** (`H3_MOCK=0`). An optional `H3_MOCK=1` renders a
  placeholder clip only to smoke-test the UI on a machine with no GPU.

```
E:\H3Minimax
├─ backend\      FastAPI + inference wrapper
└─ frontend\     React + Vite UI
```

## 1. Start the backend

```powershell
cd E:\H3Minimax\backend
copy .env.example .env      # real mode by default (H3_MOCK=0)
.\run.ps1                   # creates .venv, installs deps, serves :8000
# NOTE: real mode needs torch + diffusers + the weights — see DEPLOY.md.
# No GPU and just want to see the UI? set H3_MOCK=1 in .env first.
```

(Git Bash / Linux / macOS: `./run.sh`.)

Check it: open http://127.0.0.1:8000/api/config

## 2. Start the frontend

```powershell
cd E:\H3Minimax\frontend
npm install
npm run dev                 # http://localhost:5173
```

The Vite dev server proxies `/api` → `:8000`, so you only open **http://localhost:5173**.

Real mode generates actual MiniMax-H3 video + audio. (Only if you set `H3_MOCK=1`
does "Generate" instead produce a placeholder clip for UI testing without a GPU.)

## 3. Run via ComfyUI — recommended for consumer GPUs (default)

This is the default backend (`H3_BACKEND=comfy`). The API drives a **running ComfyUI**
that already has your quantized hybrid H3 model + turbo 8-step LoRA — the light path that
fits a 16 GB card (e.g. **Quadro RTX 5000 / RTX 4080**). No heavy diffusers load here.

On Turing cards (RTX 5000) there is **no fp8** — use the **INT8 / GGUF** quantized
weights and enable group offload to host RAM. fp8 nodes will error or crawl.

Steps:

1. Install ComfyUI (≥ 0.30.0) and load your H3 graph (hybrid quant model + turbo
   8-step LoRA + VAE + text encoder). Generate one clip in ComfyUI to confirm it works.
2. In ComfyUI: Settings → enable **Dev mode** → **Save (API Format)**. Save the export
   into `backend\workflows\` as `t2v.json` and `i2v.json` (or one `workflow.json`).
3. Copy `backend\workflows\mapping.example.json` → `mapping.json` and point each field at
   the node ids/titles from your export. See `backend\workflows\README.md` for the field list.
4. In `backend\.env`: `H3_BACKEND=comfy` and `COMFY_URL=http://127.0.0.1:8188`.
5. Start the backend + frontend (steps 1–2 above). The **Live** badge lights up when
   ComfyUI is reachable; hit Generate and the UI gets the real video.

The backend only patches prompt/seed/frames/size/images into your graph and downloads the
result — your turbo/quantized settings stay exactly as you built them.

## 4. Run the real model with diffusers (`H3_BACKEND=diffusers`, heavy)

MiniMax-H3 is integrated in **diffusers** as Modular Diffusers blocks. It is large:
the transformer is 61.7 GB and the Qwen3-VL conditioner 62.1 GB in bfloat16.

Install inference deps into `backend\.venv` (install the matching **torch** from
<https://pytorch.org> first), then uncomment the block in `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install "git+https://github.com/huggingface/diffusers.git" transformers accelerate imageio imageio-ffmpeg
# consumer int8 path also needs:  pip install torchao
```

Set the placement in `.env`:

| `H3_DEVICE_MODE` | Hardware | Notes |
|---|---|---|
| `auto` | one ~80 GB accelerator | `ComponentsManager` auto CPU offload |
| `consumer` | 24–32 GB VRAM | int8 weight-only quant + group offload, **~75 GB host RAM** |

```
H3_MOCK=0
H3_DEVICE_MODE=auto
H3_DEVICE=cuda
```

Restart the backend. The first request downloads the weights from Hugging Face and
loads the pipeline (slow, once); later requests reuse it.

### Model facts the UI respects
- 24 fps, **5–15 s**; `num_frames` is snapped to the next `17·n + 5`.
- Short edge 768 (native) or 544 (fast); canvases are multiples of 32.
- **No** negative prompt and **no** guidance scale — the checkpoint is guidance-distilled.
- Video + stereo audio (32 kHz) are generated jointly and muxed with `encode_video`.
- 2K output needs the separate H3-Regenerate-2K step (not included here).

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/config` | mode / device info |
| POST | `/api/generate` | multipart: `mode`, `prompt`, `num_frames`, `num_inference_steps`, `seed`, `width`, `height`, `image`, `last_image` |
| GET | `/api/jobs` | recent jobs |
| GET | `/api/jobs/{id}` | one job (status, logs) |
| GET | `/api/outputs/{file}` | the generated mp4 |

## Notes
- Check MiniMax-H3's community license before self-hosting in your region.
- Generation is serialized (one model, one GPU); jobs queue behind each other.
- Job history is in memory — it resets when the backend restarts (mp4s stay on disk in `backend\outputs`).
