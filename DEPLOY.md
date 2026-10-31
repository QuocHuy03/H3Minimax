# Deploy & run on the GPU machine (REAL mode)

Full steps to run MiniMax-H3 for real on a strong NVIDIA box.

## 0. Prerequisites

- **GPU**: ~80 GB VRAM for `auto` (A100/H100 80G), or 24–32 GB for `consumer` (+ ~75 GB host RAM).
- **Disk**: ~150 GB free for the weights download (transformer ~62 GB + Qwen3-VL ~62 GB + VAEs).
- **Driver/CUDA**: recent NVIDIA driver. Know your CUDA (e.g. `nvidia-smi` top-right) → pick `cu121` / `cu124` / `cu128`.
- **Software**: Python 3.12, Node 18+, Git, ffmpeg (bundled via imageio-ffmpeg).
- A **Hugging Face account** that has **accepted the MiniMax-H3 community license** on the model page, and an HF token.

## 1. Copy the project

Copy the whole `H3Minimax` folder over. Do NOT copy these (recreate on the GPU box):
`backend\.venv`, `frontend\node_modules`, `backend\outputs`, `backend\uploads`.

## 2. Hugging Face login (weights are gated)

Open <https://huggingface.co/MiniMaxAI/MiniMax-H3> → accept the license. Then:

```powershell
pip install -U "huggingface_hub[cli]"
huggingface-cli login          # paste your HF token
```

Optional — put the big cache on a roomy disk (do this BEFORE downloading):

```powershell
setx HF_HOME "D:\hf-cache"     # reopen the terminal after setx
```

## 3. Backend — install real-mode deps

```powershell
cd H3Minimax\backend
copy .env.example .env
.\setup-real.ps1 -Cuda cu124   # <-- match YOUR CUDA
```

`setup-real.ps1` creates `.venv` and installs: fastapi stack, torch (CUDA wheel),
`diffusers` from `main` (H3 isn't in a tagged release yet), transformers, accelerate,
av, imageio, and torchao (for consumer mode).

## 4. Pick the hardware mode in `.env`

```ini
H3_MOCK=0
H3_DEVICE=cuda
# one 80GB card:
H3_DEVICE_MODE=auto
H3_MEMORY_RESERVE_MARGIN=12GB
# OR 24-32GB card (needs ~75GB RAM):
# H3_DEVICE_MODE=consumer
```

## 5. ⭐ Self-check before anything else

This downloads the weights (slow, once) and loads the exact pipeline the UI uses:

```powershell
$env:H3_MOCK='0'
.\.venv\Scripts\python.exe -c "from app import model; model.warmup(); print('pipeline loaded OK')"
```

- Prints **`pipeline loaded OK`** → the model is correct and you're good to run.
- Errors here are almost always `diffusers main` API drift → send the traceback, it's a 1–2 line fix in `app\model.py`.

## 6. Run

```powershell
# terminal A - backend
cd H3Minimax\backend
.\run.ps1                       # http://127.0.0.1:8000

# terminal B - frontend
cd H3Minimax\frontend
npm install
npm run dev                     # http://localhost:5173
```

Open **http://localhost:5173**, type a prompt, hit **Generate**. First real job is slow
(model may offload layers); watch progress in the job's log panel. Output mp4 (with
stereo audio) lands in `backend\outputs` and plays in the Result panel.

## 7. Open the UI from another computer (if the GPU box is headless/remote)

Run the frontend with `--host` so your laptop's browser can reach it:

```powershell
npm run dev -- --host          # then open http://<GPU-IP>:5173
```

The Vite dev server proxies `/api` to the backend on the same machine, so you only
expose port **5173** (open it in the firewall). No CORS changes needed.

For a cleaner setup, SSH-tunnel instead of opening ports:

```bash
ssh -L 5173:localhost:5173 user@gpu-box
# open http://localhost:5173 on your laptop
```

## Tips

- **Speed**: `Quality = Fast (544)` is ~2.3x faster per step than native 768. Lower **Steps** (e.g. 30) for drafts.
- **Duration**: 5–15 s only; the slider already snaps to valid frame counts (`17·n+5`).
- **Queueing**: one model on one GPU → jobs run one at a time; others wait in History.
- **2K**: not included — needs the separate H3-Regenerate-2K step.
- First run downloads ~130 GB; later runs reuse the HF cache instantly.
