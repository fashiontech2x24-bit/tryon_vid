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
ComfyUI       :8188  (Vast.ai template; managed by setup.sh)
   models/  ← download_v2v.sh        workflow_api.json ← the graph driven per request
```

## Run on a Vast.ai ComfyUI instance

```bash
git clone <this-repo> tryon_vid
cd tryon_vid
bash setup.sh
```

`setup.sh` will:
1. locate ComfyUI + its python venv,
2. install the **VideoHelperSuite** custom node (`VHS_LoadVideo`),
3. download all Wan VACE models (`download_v2v.sh`),
4. install the server deps,
5. (re)start ComfyUI headless on `127.0.0.1:8188`,
6. launch the demo on `0.0.0.0:8000`.

Then open the demo via the Vast.ai **mapped address for port 8000**.
Expose port `8000` in the instance config (Docker `-p 8000:8000` or the
template's open-ports field). Useful flags: `APP_PORT=9000 bash setup.sh`,
`--skip-models`, `--no-comfy-restart`.

## Files

| path | purpose |
|------|---------|
| `workflow_api.json` | the workflow in ComfyUI **API format** (active 14B graph) — the template driven per request |
| `server/app.py` | FastAPI: upload → patch workflow → queue → track progress → serve mp4 |
| `server/comfy_client.py` | ComfyUI HTTP/WS wrapper (upload, prompt, ws progress, history, view) |
| `web/index.html` | the demo UI (no build step) |
| `setup.sh` | one-shot bootstrap on the instance |
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
