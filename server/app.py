"""FastAPI demo front-end for the Wan 2.1 VACE V2V ComfyUI workflow.

Flow per request:
  1. receive reference image + control video (+ optional tuning overrides)
  2. upload both into ComfyUI's input dir
  3. clone workflow_api.json, patch the input nodes + any overrides
  4. queue it and stream progress over the ComfyUI websocket (background thread)
  5. expose the resulting mp4 for inline preview + download
"""
import copy
import json
import os
import random
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from comfy_client import ComfyClient

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = ROOT / "workflow_api.json"
WEB_DIR = ROOT / "web"

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")

# Node ids in workflow_api.json (the active 14B graph)
NODE_LOAD_IMAGE = "134"      # LoadImage.image       -> reference image
NODE_LOAD_VIDEO = "151"      # VHS_LoadVideo.video   -> control video
NODE_KSAMPLER = "3"          # KSampler (seed/steps/cfg)
NODE_POSITIVE = "6"          # CLIPTextEncode (positive)
NODE_NEGATIVE = "7"          # CLIPTextEncode (negative)
NODE_VACE = "49"             # WanVaceToVideo (width/height/length)
NODE_LORA = "107"            # LoraLoader (strength_model)
NODE_CREATE_VIDEO = "68"     # CreateVideo (fps)
NODE_SHIFT = "48"            # ModelSamplingSD3 (shift)

app = FastAPI(title="Wan VACE V2V Demo")
comfy = ComfyClient(COMFY_URL)

with open(WORKFLOW_PATH) as f:
    WORKFLOW_TEMPLATE = json.load(f)

# in-memory job registry (single-box demo; fine without a DB)
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _set(job_id: str, **fields):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(fields)


def _get(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def build_workflow(image_name: str, video_name: str, opts: dict) -> dict:
    wf = copy.deepcopy(WORKFLOW_TEMPLATE)
    wf[NODE_LOAD_IMAGE]["inputs"]["image"] = image_name
    wf[NODE_LOAD_VIDEO]["inputs"]["video"] = video_name

    seed = opts.get("seed")
    wf[NODE_KSAMPLER]["inputs"]["seed"] = int(seed) if seed not in (None, "") else random.randint(0, 2**32 - 1)

    def apply(node, key, val, cast):
        if val not in (None, ""):
            wf[node]["inputs"][key] = cast(val)

    apply(NODE_POSITIVE, "text", opts.get("positive"), str)
    apply(NODE_NEGATIVE, "text", opts.get("negative"), str)
    apply(NODE_VACE, "width", opts.get("width"), int)
    apply(NODE_VACE, "height", opts.get("height"), int)
    apply(NODE_VACE, "length", opts.get("length"), int)
    apply(NODE_KSAMPLER, "steps", opts.get("steps"), int)
    apply(NODE_KSAMPLER, "cfg", opts.get("cfg"), float)
    apply(NODE_LORA, "strength_model", opts.get("lora_strength"), float)
    apply(NODE_CREATE_VIDEO, "fps", opts.get("fps"), int)
    apply(NODE_SHIFT, "shift", opts.get("shift"), float)
    return wf


def run_job(job_id: str, image_name: str, video_name: str, opts: dict):
    client_id = job_id
    try:
        workflow = build_workflow(image_name, video_name, opts)
        _set(job_id, status="running", progress=0.0, stage="queued")

        def on_progress(p):
            phase = p.get("phase")
            if phase == "sampling" and p.get("max"):
                _set(job_id, progress=round(p["value"] / p["max"], 3),
                     stage=f"sampling {p['value']}/{p['max']}")
            elif phase == "executing":
                _set(job_id, stage=f"node {p.get('node')}")
            elif phase == "queued":
                _set(job_id, stage="queued")

        history = comfy.run(workflow, client_id, on_progress)
        found = ComfyClient.find_output_video(history)
        if not found:
            _set(job_id, status="error", error="No output file found in ComfyUI history.")
            return
        filename, subfolder, ftype = found
        _set(job_id, status="done", progress=1.0, stage="done",
             output={"filename": filename, "subfolder": subfolder, "type": ftype})
    except Exception as e:  # surface any failure to the client
        _set(job_id, status="error", error=str(e))


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"comfy_up": comfy.is_up(), "comfy_url": COMFY_URL}


@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    video: UploadFile = File(...),
    seed: str = Form(""),
    positive: str = Form(""),
    negative: str = Form(""),
    width: str = Form(""),
    height: str = Form(""),
    length: str = Form(""),
    steps: str = Form(""),
    cfg: str = Form(""),
    lora_strength: str = Form(""),
    fps: str = Form(""),
    shift: str = Form(""),
):
    if not comfy.is_up():
        raise HTTPException(503, f"ComfyUI is not reachable at {COMFY_URL}")

    job_id = uuid.uuid4().hex
    img_bytes = await image.read()
    vid_bytes = await video.read()

    img_name = f"{job_id}_{Path(image.filename or 'ref.png').name}"
    vid_name = f"{job_id}_{Path(video.filename or 'control.mp4').name}"

    try:
        stored_img = comfy.upload_file(img_bytes, img_name, image.content_type or "image/png")
        stored_vid = comfy.upload_file(vid_bytes, vid_name, video.content_type or "video/mp4")
    except Exception as e:
        raise HTTPException(502, f"Upload to ComfyUI failed: {e}")

    opts = {
        "seed": seed, "positive": positive, "negative": negative,
        "width": width, "height": height, "length": length,
        "steps": steps, "cfg": cfg, "lora_strength": lora_strength,
        "fps": fps, "shift": shift,
    }
    _set(job_id, status="queued", progress=0.0, stage="starting")
    threading.Thread(target=run_job, args=(job_id, stored_img, stored_vid, opts), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = _get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return JSONResponse({k: v for k, v in job.items() if k != "output"} | {"has_output": "output" in job})


def _fetch_output(job_id: str) -> tuple[bytes, str]:
    job = _get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job.get("status") != "done" or "output" not in job:
        raise HTTPException(409, "result not ready")
    out = job["output"]
    data = comfy.get_file(out["filename"], out["subfolder"], out["type"])
    return data, out["filename"]


@app.get("/api/preview/{job_id}")
def preview(job_id: str):
    data, filename = _fetch_output(job_id)
    media = "video/mp4" if filename.lower().endswith(".mp4") else "application/octet-stream"
    return Response(content=data, media_type=media)


@app.get("/api/result/{job_id}")
def result(job_id: str):
    data, filename = _fetch_output(job_id)
    return Response(
        content=data,
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
