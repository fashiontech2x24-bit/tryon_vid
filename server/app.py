"""FastAPI app for the Wan 2.1 VACE try-on demo — full 3-stage pipeline.

Per request (user uploads ONLY a reference image + picks a motion preset):

  [1] preprocess  pose_pipeline.ControlVideoPipeline
        reference image + server-side motion preset -> OpenPose control video
        (the control clip's poses are estimated once and disk-cached, so each
         request only estimates the user's image + cheap retarget/render)
  [2] generate    ComfyUI / Wan2.1 VACE
        reference image + control video -> generated clip
  [3] postprocess boomerang_api.boomerang
        generated clip -> seamless flow-eased boomerang (final result)

There is no manual control-video upload — the motion comes from assets/motion_presets/.
"""
import copy
import json
import os
import random
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from comfy_client import ComfyClient

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = ROOT / "workflow_api.json"
WEB_DIR = ROOT / "web"
PIPELINE_DIR = Path(__file__).resolve().parent / "pipeline"

# make the copied pipeline modules importable (pose_pipeline does `import pose_retarget`)
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
import pose_pipeline                    # noqa: E402
from boomerang_api import boomerang     # noqa: E402

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
PRESETS_DIR = Path(os.environ.get("PRESETS_DIR", ROOT / "assets" / "motion_presets")).resolve()
RUNS_DIR = Path(os.environ.get("RUNS_DIR", ROOT / "runs")).resolve()
MODEL_DIR = os.environ.get("DWPOSE_MODEL_DIR", str(ROOT / "models" / "dwpose"))
CACHE_DIR = os.environ.get("POSE_CACHE_DIR", str(ROOT / ".pose_cache"))
POSE_DEVICE = os.environ.get("POSE_DEVICE", "cpu")
VID_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

RUNS_DIR.mkdir(parents=True, exist_ok=True)
PRESETS_DIR.mkdir(parents=True, exist_ok=True)

# Node ids in workflow_api.json (active 14B graph)
NODE_LOAD_IMAGE = "134"; NODE_LOAD_VIDEO = "151"; NODE_KSAMPLER = "3"
NODE_POSITIVE = "6"; NODE_NEGATIVE = "7"; NODE_VACE = "49"
NODE_LORA = "107"; NODE_CREATE_VIDEO = "68"; NODE_SHIFT = "48"

app = FastAPI(title="Wan VACE Try-On Pipeline")
comfy = ComfyClient(COMFY_URL)

with open(WORKFLOW_PATH) as f:
    WORKFLOW_TEMPLATE = json.load(f)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
# rtmlib / onnxruntime sessions are not thread-safe; serialise pose estimation.
POSE_LOCK = threading.Lock()
_PIPELINES: dict[str, "pose_pipeline.ControlVideoPipeline"] = {}


def _set(job_id, **fields):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(fields)


def _get(job_id):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        return dict(j) if j else None


def list_presets():
    return sorted(p.name for p in PRESETS_DIR.iterdir()
                  if p.suffix.lower() in VID_EXT and p.exists())


def get_pipeline(preset):
    """Lazily build + cache one ControlVideoPipeline per motion preset.
    First use estimates the clip's poses (disk-cached); later uses are cheap."""
    if preset not in _PIPELINES:
        clip = PRESETS_DIR / preset
        if not clip.exists():
            raise FileNotFoundError(f"unknown preset: {preset}")
        _PIPELINES[preset] = pose_pipeline.ControlVideoPipeline(
            str(clip), device=POSE_DEVICE, model_dir=MODEL_DIR, cache_dir=CACHE_DIR, verbose=True)
    return _PIPELINES[preset]


# --- workflow patching (same as before, minus the manual control video) ------
def build_workflow(image_name, video_name, opts):
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


def _fnum(v, default):
    try:
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def run_job(job_id, image_path, opts):
    run = RUNS_DIR / job_id
    try:
        # -- [1] preprocess: retarget motion preset onto the user image -------
        _set(job_id, status="running", progress=0.05, stage="preprocess: retargeting pose")
        control_path = run / "control.mp4"
        with POSE_LOCK:
            pipe = get_pipeline(opts["preset"])  # first call per preset estimates the clip (slow, cached)
            pipe.generate(
                str(image_path), str(control_path),
                pose_mode=opts.get("pose_mode", "relative"),
                root_motion=_fnum(opts.get("root_motion"), 1.0),
                smoothing=_fnum(opts.get("smoothing"), 0.4),
                foreshorten=_fnum(opts.get("foreshorten"), 0.0),
                blend_frames=int(_fnum(opts.get("blend_frames"), 0)))
        _set(job_id, control_url=f"/api/run/{job_id}/control.mp4")

        # -- [2] generate: drive ComfyUI / VACE -------------------------------
        _set(job_id, progress=0.12, stage="uploading to ComfyUI")
        img_name = f"{job_id}_{Path(image_path).name}"
        vid_name = f"{job_id}_control.mp4"
        stored_img = comfy.upload_file(image_path.read_bytes(), img_name, "image/png")
        stored_vid = comfy.upload_file(control_path.read_bytes(), vid_name, "video/mp4")

        workflow = build_workflow(stored_img, stored_vid, opts)

        def on_progress(p):
            if p.get("phase") == "sampling" and p.get("max"):
                frac = p["value"] / p["max"]
                _set(job_id, progress=round(0.15 + 0.65 * frac, 3), stage=f"generating {p['value']}/{p['max']}")
            elif p.get("phase") == "executing":
                _set(job_id, stage=f"generating (node {p.get('node')})")

        _set(job_id, progress=0.15, stage="generating (Wan VACE)")
        history = comfy.run(workflow, job_id, on_progress)
        found = ComfyClient.find_output_video(history)
        if not found:
            _set(job_id, status="error", error="No output produced by ComfyUI.")
            return
        filename, subfolder, ftype = found
        gen_path = run / "generated.mp4"
        gen_path.write_bytes(comfy.get_file(filename, subfolder, ftype))
        _set(job_id, progress=0.85, stage="fetched generated clip", generated_url=f"/api/run/{job_id}/generated.mp4")

        # -- [3] postprocess: seamless boomerang ------------------------------
        final_path = run / "final.mp4"
        loop = str(opts.get("loop", "true")).lower() in ("1", "true", "on", "yes")
        _set(job_id, progress=0.9, stage="postprocess: boomerang")
        try:
            boomerang(str(gen_path), str(final_path),
                      window=int(_fnum(opts.get("window"), 3)),
                      crf=int(_fnum(opts.get("crf"), 16)), loop=loop)
            final_name = "final.mp4"
        except ValueError as e:
            # window too large for a short clip → fall back to the raw generated clip
            final_path = gen_path
            final_name = "generated.mp4"
            _set(job_id, warning=f"boomerang skipped: {e}")

        _set(job_id, status="done", progress=1.0, stage="done",
             output=str(final_path), output_url=f"/api/run/{job_id}/{final_name}")
    except Exception as e:
        _set(job_id, status="error", error=str(e))


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"comfy_up": comfy.is_up(), "comfy_url": COMFY_URL,
            "device": POSE_DEVICE, "presets": list_presets()}


@app.get("/api/presets")
def presets():
    return {"presets": list_presets(), "device": POSE_DEVICE}


@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    preset: str = Form(...),
    # preprocess (pose retarget)
    pose_mode: str = Form("relative"),
    root_motion: str = Form(""), smoothing: str = Form(""),
    foreshorten: str = Form(""), blend_frames: str = Form(""),
    # generate (VACE)
    seed: str = Form(""), positive: str = Form(""), negative: str = Form(""),
    width: str = Form(""), height: str = Form(""), length: str = Form(""),
    steps: str = Form(""), cfg: str = Form(""), lora_strength: str = Form(""),
    fps: str = Form(""), shift: str = Form(""),
    # postprocess (boomerang)
    window: str = Form(""), crf: str = Form(""), loop: str = Form("true"),
):
    if preset not in list_presets():
        raise HTTPException(400, f"invalid or missing preset (have: {list_presets()})")
    if not comfy.is_up():
        raise HTTPException(503, f"ComfyUI is not reachable at {COMFY_URL}")

    job_id = uuid.uuid4().hex
    run = RUNS_DIR / job_id
    run.mkdir(parents=True, exist_ok=True)
    ext = Path(image.filename or "ref.png").suffix.lower() or ".png"
    image_path = run / f"reference{ext}"
    image_path.write_bytes(await image.read())

    opts = dict(preset=preset, pose_mode=pose_mode, root_motion=root_motion,
                smoothing=smoothing, foreshorten=foreshorten, blend_frames=blend_frames,
                seed=seed, positive=positive, negative=negative, width=width, height=height,
                length=length, steps=steps, cfg=cfg, lora_strength=lora_strength, fps=fps,
                shift=shift, window=window, crf=crf, loop=loop)
    _set(job_id, status="queued", progress=0.0, stage="starting")
    threading.Thread(target=run_job, args=(job_id, image_path, opts), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = _get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return JSONResponse({k: v for k, v in job.items() if k != "output"})


@app.get("/api/run/{job_id}/{filename}")
def run_file(job_id: str, filename: str):
    # guard against path traversal; only serve known artifacts
    if filename not in ("control.mp4", "generated.mp4", "final.mp4"):
        raise HTTPException(404, "unknown artifact")
    path = (RUNS_DIR / job_id / filename).resolve()
    if path.parent != (RUNS_DIR / job_id).resolve() or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4")


# Back-compat aliases: preview (inline) + result (download) of the final video.
@app.get("/api/preview/{job_id}")
def preview(job_id: str):
    job = _get(job_id) or {}
    if job.get("status") != "done":
        raise HTTPException(409, "result not ready")
    return FileResponse(job["output"], media_type="video/mp4")


@app.get("/api/result/{job_id}")
def result(job_id: str):
    job = _get(job_id) or {}
    if job.get("status") != "done":
        raise HTTPException(409, "result not ready")
    return FileResponse(job["output"], media_type="video/mp4",
                        filename=f"tryon_{job_id}.mp4")
