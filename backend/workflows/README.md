# ComfyUI workflows for the H3 UI

The backend (with `H3_BACKEND=comfy`) drives your ComfyUI by loading a workflow
here, patching a few node fields with the UI's params, queuing it, and downloading
the result. Your quantized + turbo-8step H3 graph runs unchanged.

## Setup (3 steps)

1. **Export your workflow as API format.**
   In ComfyUI: Settings → enable **Dev mode**, then **Save (API Format)**.
   Save it here as:
   - `t2v.json`  — text-to-video graph
   - `i2v.json`  — image-to-video graph
   - (or a single `workflow.json` used for both)

2. **Create `mapping.json`** from `mapping.example.json` and point each field at the
   right node. Open your exported JSON: each node is `"<id>": { "class_type": ..., "inputs": {...}, "_meta": {"title": ...} }`.
   - Target a node by its **id** (`"node": "40"`) or by its **title** (`"title": "H3 Sampler"`), titles survive re-numbering.
   - Map the UI params you want to control; leave the rest out to keep the graph's own
     values (e.g. don't map `steps` and your 8-step turbo stays 8).
   - For image-to-video, point `images[].from: "first"/"last"` at your LoadImage nodes.
     The backend uploads the file to ComfyUI and sets that input to the uploaded name.
   - `output_node` is the id of your video saver (VHS VideoCombine / SaveVideo). Optional —
     if omitted, the backend scans all outputs and takes the first video file.

3. **Point the backend at ComfyUI**: set `COMFY_URL` in `.env` (default
   `http://127.0.0.1:8188`), set `H3_BACKEND=comfy`, start the backend.

The UI's **Live** badge turns on when ComfyUI is reachable.

## Placeholders usable in `value`
`$prompt` · `$seed` · `$steps` · `$num_frames` · `$width` · `$height`

A literal (non-`$`) value is set as-is. Numbers stay numbers, strings stay strings.
