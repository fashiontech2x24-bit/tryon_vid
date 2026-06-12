# Wan 2.1 VACE — Video Try-On Studio

A web app for the ComfyUI **Wan 2.1 VACE V2V** workflow, with three tabs:

* **Quick Generate** — the original single-preset flow: reference image +
  motion preset → control video → one VACE generation → seamless boomerang.
* **Control Studio** — build motion presets from raw recordings: upload a
  (e.g. 60fps) clip, trim it, **uniformly decimate** it (duration + fps knobs,
  frames removed evenly across the whole trim, snapped to Wan's 4k+1 grid,
  max 29), tune the **motion-transfer config** (pose mode, root motion,
  smoothing, foreshorten, blend, **head lock**, mirror preview), check the
  retargeted skeleton against a reference image, and save it all as a preset.
* **Compose** — multi-segment choreography: a list of segments
  (preset + mirror/reverse/boomerang/speed) is deduplicated into unique
  (preset, mirror) **generations** (same seed + reference for coherence),
  then stitched with **flow-morphed seams** (optional slow-mo "pause" per
  junction). The stitch stage re-runs in seconds via *Re-stitch only* — no
  regeneration — so seam/timing knobs can be iterated freely. The whole thing
  is captured as a JSON recipe (`runs/<job>/recipe.json`).

The default Compose recipe is the target choreography: *step-back reversed*
(→ steps forward) → slow-mo pause → *torso-rotation boomerang* → *the same
rotation mirrored at the skeleton level* (opposite direction without flipping
garment/face pixels). Head lock keeps the face toward camera while the torso
rotates.

```
Browser (web/index.html)
   │  upload reference image + pick motion preset → poll progress → preview/download
   ▼
FastAPI app  :8000   (server/app.py)
   │  /api/generate  /api/status/{id}  /api/run/{id}/<file>  /api/result/{id}
   │
   ├─[1] preprocess  pipeline/pose_pipeline.py  (DWPose retarget)
   │        reference image + motion preset → OpenPose control video
   ├─[2] generate    ComfyUI / Wan2.1 VACE  :8188
   │        reference image + control video → generated clip
   └─[3] postprocess pipeline/boomerang_api.py
            generated clip → seamless flow-eased boomerang  (final result)
```

**No manual control-video upload** — the motion comes from server-side presets
in `assets/motion_presets/`. Each preset's control poses are estimated once and
disk-cached (`.pose_cache/`), so per request only the user's image is estimated.
The preprocess/postprocess stages run inside the web app for now; they can be
moved onto the GPU worker / into ComfyUI after finetuning.

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
4. install the server + pipeline deps (rtmlib/onnxruntime/opencv + ffmpeg) and
   prefetch the DWPose ONNX models (~350 MB),
5. start our ComfyUI headless on a free port (`COMFY_PORT`, auto-bumps if busy),
6. launch the demo on `0.0.0.0:8000`, pointed at our ComfyUI.

### Motion presets

Drop (or symlink) a motion clip into `assets/motion_presets/` and it appears in
the UI dropdown on next load. Bundled: `shot3.mp4`, `shot1.mp4`, `full-pivot.mp4`.
Pose estimation runs on CPU by default (`POSE_DEVICE=cuda` for onnxruntime-gpu).

Then open the demo via the Vast.ai **mapped address for port 8000** (expose
port `8000` in the instance config). Stop everything with `bash stop.sh`.

Env / flags: `COMFY_DIR=`, `COMFY_PORT=`, `APP_PORT=`, `COMFY_REF=`,
`--skip-models`, `--skip-comfy-install`, `--fresh`.

## Files

| path | purpose |
|------|---------|
| `workflow_api.json` | the workflow in ComfyUI **API format** (active 14B graph) — the template driven per request |
| `server/app.py` | FastAPI: preprocess → ComfyUI generate → boomerang → serve mp4 |
| `server/comfy_client.py` | ComfyUI HTTP/WS wrapper (upload, prompt, ws progress, history, view) |
| `server/pipeline/pose_pipeline.py`, `pose_retarget.py` | preprocess: DWPose retarget → control video (mirror, head_lock, frame subsetting) |
| `server/pipeline/video_edit.py` | trim + uniform decimation (duration/fps knobs, 4k+1 snap) |
| `server/pipeline/boomerang_api.py` | postprocess: seamless flow-eased boomerang |
| `server/pipeline/stitch_api.py` | multi-segment composer: reverse/boomerang/speed per segment, flow-morphed seams with slow-mo pause |
| `assets/motion_presets/` | server-side motion clips + optional `.json` sidecars (retarget config per preset, written by Control Studio) |
| `assets/control_sources/` | uploaded raw recordings (workbench input; not committed) |
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

## API (new endpoints)

| endpoint | purpose |
|----------|---------|
| `POST /api/control/upload` · `GET /api/control/sources` | raw recording management |
| `POST /api/control/preview` | trim + decimate → preview clip (CPU only) |
| `POST /api/control/pose_preview` | retargeted-skeleton preview (first call per source estimates poses; disk-cached) |
| `POST /api/control/save_preset` | write decimated clip + retarget sidecar into `assets/motion_presets/` |
| `POST /api/compose` | image + recipe JSON → N generations → stitched `final.mp4` |
| `POST /api/restitch/{job_id}` | re-run only the stitch stage with new knobs |

Recipe shape: `{"segments": [{"preset", "mirror", "reverse", "boomerang",
"boomerang_window", "speed"}], "seams": [{"pause", "window"}],
"stitch": {"out_fps", "crf", "slowdown"}, "gen": {"seed", "steps", ...}}`.

## Local dev (without Vast)

Point the app at any reachable ComfyUI:

```bash
pip install -r server/requirements.txt
COMFY_URL=http://127.0.0.1:8188 python -m uvicorn app:app --app-dir server --port 8000
```

Or run the full stack with no GPU/models at all (mock ComfyUI + stubbed pose
estimation) and the e2e test:

```bash
.venv/bin/python -m uvicorn mock_comfy:app --app-dir localtest --port 8189 &
COMFY_URL=http://127.0.0.1:8189 APP_PORT=8010 .venv/bin/python localtest/app_fakepose.py &
APP_URL=http://127.0.0.1:8010 .venv/bin/python localtest/test_compose_e2e.py
```
