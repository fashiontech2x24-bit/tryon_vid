# Wan 2.1 VACE — Video Try-On Demo

A thin web demo in front of your ComfyUI **Wan 2.1 VACE V2V** workflow:
upload a **reference image** + a **control video**, watch progress, preview and
download the generated video. ComfyUI stays as the inference engine; this app
just drives its API.

```
Browser (web/index.html)
   │  upload image + control video → poll progress → preview/download mp4
   ▼
FastAPI app  :8000   (server/app.py)
   │  /api/generate  /api/status/{id}  /api/preview/{id}  /api/result/{id}
   ▼
ComfyUI      :8188   (OUR own clone, started by setup.sh on a free port)
   models/  ← reused via extra_model_paths.yaml / download_v2v.sh
   workflow_api.json ← the graph driven per request
```

We install and run our **own** ComfyUI (not the Vast template's) so versions,
custom nodes, and the listening port are under our control. Models already on
the box are reused — no re-download.

## Run on a Vast.ai instance

```bash
git clone <this-repo> tryon_vid
cd tryon_vid
bash setup.sh
```

`setup.sh` will:
1. pick a python that already has CUDA torch,
2. clone **ComfyUI** to `$COMFY_DIR` (default `/workspace/ComfyUI`) + install
   **ComfyUI-Manager** and **VideoHelperSuite** (`VHS_LoadVideo`) — keeping the
   existing torch build untouched,
3. reuse existing models via `extra_model_paths.yaml` (else download them),
4. install the server deps,
5. start our ComfyUI headless on a free port (`COMFY_PORT`, auto-bumps if busy),
6. launch the demo on `0.0.0.0:8000`, pointed at our ComfyUI.

Then open the demo via the Vast.ai **mapped address for port 8000** (expose
port `8000` in the instance config). Stop everything with `bash stop.sh`.

Env / flags: `COMFY_DIR=`, `COMFY_PORT=`, `APP_PORT=`, `COMFY_REF=`,
`--skip-models`, `--skip-comfy-install`, `--fresh`.

## Files

| path | purpose |
|------|---------|
| `workflow_api.json` | the workflow in ComfyUI **API format** (active 14B graph) — the template driven per request |
| `server/app.py` | FastAPI: upload → patch workflow → queue → track progress → serve mp4 |
| `server/comfy_client.py` | ComfyUI HTTP/WS wrapper (upload, prompt, ws progress, history, view) |
| `web/index.html` | the demo UI (no build step) |
| `setup.sh` | clone+run our own ComfyUI (+Manager+VHS), reuse models, start app |
| `stop.sh` | stop the ComfyUI + app that setup.sh started |
| `download_v2v.sh` | model downloader (unchanged) |

## Updating the workflow

When you tune the graph in ComfyUI, re-export it (Settings → enable **Dev mode**
→ **Save (API Format)**) and overwrite `workflow_api.json`. The app patches these
nodes per request: `134` (image), `151` (control video), `3` (seed/steps/cfg),
`6`/`7` (prompts), `49` (width/height/length), `107` (LoRA strength), `68` (fps),
`48` (shift). If you renumber those nodes, update the `NODE_*` constants in
`server/app.py`.

## Local dev (without Vast)

Point the app at any reachable ComfyUI:

```bash
pip install -r server/requirements.txt
COMFY_URL=http://127.0.0.1:8188 python -m uvicorn app:app --app-dir server --port 8000
```
