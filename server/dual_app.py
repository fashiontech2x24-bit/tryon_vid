"""FastAPI app for the dual-inference rotation product.

One reference image + a slowdown factor -> TWO seamless looping videos:
rotate-left and rotate-right.

  [1] pose-retarget   reference image + fixed rotation preset -> two control
                      videos (one normal, one skeleton-mirrored = opposite spin)
  [2] dual inference  the two controls run in PARALLEL on two ComfyUI servers
                      (A=right, B=left) on the one RTX 6000 Pro, shared seed
  [3] postprocess     per side: RIFE smooth slow-mo (factor) -> seamless boomerang,
                      then a hard-cut combine (right -> left) into one clip
  [4] serve+callback  left/right/combined mp4s downloadable; on completion all
                      three are multipart-POSTed to the callback_url (retry+backoff)

Only ONE job runs at a time (each job uses both ComfyUI servers for its L/R
pair); a second request while busy gets 409. Per-stage and total timings —
including the GPU inference time — are tracked and shown in the web app.
"""
import copy
import json
import os
import random
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from comfy_client import ComfyClient

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = ROOT / "workflow_api.json"
WEB_DIR = ROOT / "web"
PIPELINE_DIR = Path(__file__).resolve().parent / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
import pose_pipeline                    # noqa: E402
import rife_slow                        # noqa: E402
from boomerang_api import boomerang     # noqa: E402

import requests                         # noqa: E402

# --- config -----------------------------------------------------------------
COMFY_URL_A = os.environ.get("COMFY_URL_A", "http://127.0.0.1:8188")  # right
COMFY_URL_B = os.environ.get("COMFY_URL_B", "http://127.0.0.1:8189")  # left
PRESETS_DIR = Path(os.environ.get("PRESETS_DIR", ROOT / "assets" / "motion_presets")).resolve()
RUNS_DIR = Path(os.environ.get("RUNS_DIR", ROOT / "runs")).resolve()
MODEL_DIR = os.environ.get("DWPOSE_MODEL_DIR", str(ROOT / "models" / "dwpose"))
CACHE_DIR = os.environ.get("POSE_CACHE_DIR", str(ROOT / ".pose_cache"))
ROTATE_PRESET = os.environ.get("ROTATE_PRESET", "rotate.mp4")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
CALLBACK_SECRET = os.environ.get("CALLBACK_SECRET", "")
# --- asset-service /submit contract -----------------------------------------
# shared secret sent as X-Internal-Auth on the result callback (same secret the
# image box uses for /v1/vton/result).
ASSET_INTERNAL_SECRET = os.environ.get("ASSET_INTERNAL_SECRET", "")
# the asset-service IP changes between deploys; when set, the callback's host is
# rewritten to this IP (port + path from the incoming callback_url are kept).
CALLBACK_HOST_OVERRIDE = os.environ.get("CALLBACK_HOST_OVERRIDE", "").strip()

# slowdown knob bounds (web app: slider 1.0–2.0, default 1.2)
SLOWDOWN_MIN, SLOWDOWN_MAX, SLOWDOWN_DEFAULT = 1.0, 2.0, 1.2

# fixed generation params (the 14B graph is locked to 29 @ 12 fps)
GEN_LENGTH, GEN_FPS = 29, 12
# generation resolution — 9:16, kept at ~Wan's 720p training area (0.92M px) to
# avoid above-training-res artifacts; matches the 928x1664 (0.5577) input AR to
# within ~0.9%, so WanVaceToVideo's internal resize introduces no visible stretch.
GEN_WIDTH, GEN_HEIGHT = 720, 1280
# fixed boomerang params (UI only exposes slowdown; RIFE already did the slowing)
BM_WINDOW, BM_CRF, BM_LOOP = 3, 16, True

RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Node ids in workflow_api.json (active 14B graph) — same as the demo app.
NODE_LOAD_IMAGE = "134"; NODE_LOAD_VIDEO = "151"; NODE_KSAMPLER = "3"
NODE_VACE = "49"; NODE_CREATE_VIDEO = "68"

with open(WORKFLOW_PATH) as f:
    WORKFLOW_TEMPLATE = json.load(f)


def _resolve_pose_device(requested: str) -> str:
    req = (requested or "auto").strip().lower()
    if req in ("cpu", "mps"):
        return req
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except Exception:
        pass
    if req == "cuda":
        print("[dual_app] POSE_DEVICE=cuda requested but onnxruntime has no CUDA "
              "provider — falling back to cpu")
    return "cpu"


POSE_DEVICE = _resolve_pose_device(os.environ.get("POSE_DEVICE", "auto"))

app = FastAPI(title="Dual-Inference Rotation")
comfy_right = ComfyClient(COMFY_URL_A)
comfy_left = ComfyClient(COMFY_URL_B)

# --- job + single-flight state ----------------------------------------------
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
# rtmlib / onnxruntime sessions are not thread-safe; serialise pose estimation.
POSE_LOCK = threading.Lock()
# only one rotation job at a time (it needs both ComfyUI servers).
BUSY_LOCK = threading.Lock()
_BUSY = {"job_id": None}
_PIPELINE: dict[str, "pose_pipeline.ControlVideoPipeline"] = {}


def _set(job_id, **fields):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(fields)


def _get(job_id):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        return dict(j) if j else None


def _rotate_pipeline():
    """Lazily build + cache the rotation preset's pipeline (first use estimates
    the preset's poses, then disk-cached forever)."""
    clip = (PRESETS_DIR / ROTATE_PRESET).resolve()
    key = str(clip)
    if key not in _PIPELINE:
        if not clip.exists():
            raise FileNotFoundError(f"rotation preset not found: {clip}")
        _PIPELINE[key] = pose_pipeline.ControlVideoPipeline(
            key, device=POSE_DEVICE, model_dir=MODEL_DIR, cache_dir=CACHE_DIR,
            verbose=True)
    return _PIPELINE[key]


def _retarget_cfg() -> dict:
    """Retarget config from the preset's sidecar (falls back to config E)."""
    cfg = {"pose_mode": "absolute", "root_motion": 1.0, "smoothing": 0.4,
           "foreshorten": 1.0, "blend_frames": 0, "head_lock": 0.3}
    sidecar = PRESETS_DIR / (Path(ROTATE_PRESET).stem + ".json")
    if sidecar.exists():
        try:
            cfg.update(json.loads(sidecar.read_text()).get("retarget", {}))
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def build_workflow(image_name, video_name, seed):
    wf = copy.deepcopy(WORKFLOW_TEMPLATE)
    wf[NODE_LOAD_IMAGE]["inputs"]["image"] = image_name
    wf[NODE_LOAD_VIDEO]["inputs"]["video"] = video_name
    # VHS must load the control 1:1 (no resampling) so frame count == length
    wf[NODE_LOAD_VIDEO]["inputs"]["force_rate"] = GEN_FPS
    wf[NODE_KSAMPLER]["inputs"]["seed"] = int(seed)
    wf[NODE_VACE]["inputs"]["length"] = GEN_LENGTH
    wf[NODE_VACE]["inputs"]["width"] = GEN_WIDTH
    wf[NODE_VACE]["inputs"]["height"] = GEN_HEIGHT
    wf[NODE_CREATE_VIDEO]["inputs"]["fps"] = GEN_FPS
    return wf


def _generate_side(side, client, run, stored_img, control_path, seed, job_id, progress):
    """Upload + run one side on its ComfyUI; write gen_<side>.mp4. Returns its
    wall-clock seconds. Raises on any failure (job fails as a whole)."""
    t0 = time.monotonic()
    stored_vid = client.upload_file(control_path.read_bytes(),
                                    f"{job_id}_{side}_control.mp4", "video/mp4")
    wf = build_workflow(stored_img[client.base_url], stored_vid, seed)

    def on_progress(p):
        if p.get("phase") == "sampling" and p.get("max"):
            progress[side] = p["value"] / p["max"]
            _publish_gen_progress(job_id, progress)

    history = client.run(wf, f"{job_id}_{side}", on_progress)
    found = ComfyClient.find_output_video(history)
    if not found:
        raise RuntimeError(f"{side}: ComfyUI produced no output")
    filename, subfolder, ftype = found
    (run / f"gen_{side}.mp4").write_bytes(client.get_file(filename, subfolder, ftype))
    return time.monotonic() - t0


def _publish_gen_progress(job_id, progress):
    avg = (progress.get("left", 0.0) + progress.get("right", 0.0)) / 2.0
    _set(job_id, progress=round(0.15 + 0.55 * avg, 3),
         stage=f"generating (L {int(progress.get('left', 0) * 100)}% · "
               f"R {int(progress.get('right', 0) * 100)}%)")


def _postprocess_side(side, run, factor):
    """RIFE slow-mo (factor) -> seamless boomerang -> <side>_final.mp4."""
    gen = run / f"gen_{side}.mp4"
    slowed = rife_slow.slowmo(str(gen), str(run / f"slow_{side}.mp4"), factor,
                              fps=GEN_FPS, device="cuda")
    final = run / f"{side}_final.mp4"
    try:
        boomerang(slowed, str(final), window=BM_WINDOW, crf=BM_CRF, loop=BM_LOOP,
                  slowdown=1.0)
    except ValueError:
        # window too large for a very short clip -> fall back to the slowed clip
        import shutil
        shutil.copy(slowed, final)
    return final


def _combine(run):
    """Hard-cut concat of the two finals into one clip: right then left.
    Both finals share res/fps/codec, so this is a single clean re-encode."""
    right, left = run / "right_final.mp4", run / "left_final.mp4"
    out = run / "combined.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(right), "-i", str(left), "-filter_complex",
         "[0:v]format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS[a];"
         "[1:v]format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS[b];"
         "[a][b]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
         str(out)],
        check=True)
    return out


def _send_callback(job_id, callback_url, status, factor, run, timings, error=None):
    """Multipart-POST both finals (+ JSON metadata) to callback_url, with
    exponential backoff until a 2xx (or attempts exhausted)."""
    meta = {"job_id": job_id, "status": status, "slowdown": factor,
            "timings": timings}
    if error:
        meta["error"] = error
    headers = {"X-Callback-Secret": CALLBACK_SECRET} if CALLBACK_SECRET else {}
    delay = 1.0
    for attempt in range(1, 5):
        files = []
        opened = []
        try:
            data = {"meta": json.dumps(meta)}
            if status == "done":
                parts = [("left", run / "left_final.mp4"),
                         ("right", run / "right_final.mp4"),
                         ("combined", run / "combined.mp4")]
                for name, p in parts:
                    f = p.open("rb")
                    opened.append(f)
                    files.append((f"{name}_video", (f"{name}.mp4", f, "video/mp4")))
            r = requests.post(callback_url, data=data, files=files or None,
                              headers=headers, timeout=120)
            if 200 <= r.status_code < 300:
                _set(job_id, callback="delivered")
                return
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        finally:
            for f in opened:
                f.close()
        _set(job_id, callback=f"retry {attempt} ({last})")
        time.sleep(delay)
        delay *= 2
    _set(job_id, callback=f"failed after retries ({last})")


def _render_core(job_id, run, image_path, factor, seed):
    """Steps [1]-[3]: pose-retarget -> dual VACE inference -> postprocess +
    hard-cut combine. Updates job progress; returns the stage-timings dict
    (no total_s — the caller stamps that). Produces left/right/combined under
    `run`. Raises on any failure."""
    timings: dict[str, float] = {}
    cfg = _retarget_cfg()

    # -- [1] pose-retarget: one estimate of the user image, two renders ----
    _set(job_id, status="running", progress=0.05, stage="pose-retargeting")
    t0 = time.monotonic()
    control_right = run / "control_right.mp4"
    control_left = run / "control_left.mp4"
    with POSE_LOCK:
        pipe = _rotate_pipeline()
        pipe.generate(str(image_path), str(control_right), fps=GEN_FPS,
                      mirror=False, **cfg)
        pipe.generate(str(image_path), str(control_left), fps=GEN_FPS,
                      mirror=True, **cfg)
    timings["retarget_s"] = round(time.monotonic() - t0, 2)

    # -- [2] dual inference: both controls in PARALLEL, shared seed --------
    _set(job_id, progress=0.15, stage="generating (Wan VACE ×2)")
    img_bytes = image_path.read_bytes()
    stored_img = {
        comfy_right.base_url: comfy_right.upload_file(
            img_bytes, f"{job_id}_ref.png", "image/png"),
        comfy_left.base_url: comfy_left.upload_file(
            img_bytes, f"{job_id}_ref.png", "image/png"),
    }
    progress = {"left": 0.0, "right": 0.0}
    results: dict[str, object] = {}

    def worker(side, client, control):
        try:
            results[side] = _generate_side(
                side, client, run, stored_img, control, seed, job_id, progress)
        except Exception as e:  # noqa: BLE001 — captured, re-raised below
            results[side] = e

    t0 = time.monotonic()
    threads = [
        threading.Thread(target=worker, args=("right", comfy_right, control_right)),
        threading.Thread(target=worker, args=("left", comfy_left, control_left)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    timings["inference_s"] = round(time.monotonic() - t0, 2)
    for side in ("left", "right"):
        if isinstance(results[side], Exception):
            raise RuntimeError(f"{side} generation failed: {results[side]}")
        timings[f"gen_{side}_s"] = round(float(results[side]), 2)

    # -- [3] postprocess: RIFE slow-mo -> boomerang, per side --------------
    _set(job_id, progress=0.72, stage="postprocess (RIFE + boomerang)")
    t0 = time.monotonic()
    for side in ("right", "left"):
        _postprocess_side(side, run, factor)
    # hard-cut combine: right -> left, one downloadable clip
    _combine(run)
    timings["postprocess_s"] = round(time.monotonic() - t0, 2)
    return timings


def run_rotate(job_id, image_path, factor, callback_url):
    run = RUNS_DIR / job_id
    timings: dict[str, float] = {}
    t_start = time.monotonic()
    try:
        seed = random.randint(0, 2**32 - 1)
        timings = _render_core(job_id, run, image_path, factor, seed)
        timings["total_s"] = round(time.monotonic() - t_start, 2)
        _set(job_id, status="done", progress=1.0, stage="done", timings=timings,
             left_url=f"/api/rotate/result/{job_id}/left",
             right_url=f"/api/rotate/result/{job_id}/right",
             combined_url=f"/api/rotate/result/{job_id}/combined")

        if callback_url:
            _send_callback(job_id, callback_url, "done", factor, run, timings)
    except Exception as e:  # noqa: BLE001
        timings["total_s"] = round(time.monotonic() - t_start, 2)
        _set(job_id, status="error", error=str(e), timings=timings)
        if callback_url:
            _send_callback(job_id, callback_url, "error", factor, run, timings,
                           error=str(e))
    finally:
        with BUSY_LOCK:
            _BUSY["job_id"] = None


# --- asset-service /submit flow ---------------------------------------------
def _override_callback_host(url: str) -> str:
    """Rewrite the callback URL's host to CALLBACK_HOST_OVERRIDE (if set),
    keeping scheme, port and path. The asset-service IP changes per deploy;
    its port + endpoint stay constant."""
    if not CALLBACK_HOST_OVERRIDE:
        return url
    p = urlparse(url)
    netloc = CALLBACK_HOST_OVERRIDE + (f":{p.port}" if p.port else "")
    return urlunparse(p._replace(netloc=netloc))


def _download_image(url: str, dest: Path):
    """GET a presigned image URL (no creds) and write it to dest."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)


def _send_video_callback(callback_url, job_id, status, video_path=None, error=None):
    """multipart/form-data result POST to asset-service (/v1/video/result),
    mirroring the image box. SUCCESS attaches the mp4 as `video`; FAILED/TIMEOUT
    attach an `error`. Auth via X-Internal-Auth. Retries with backoff."""
    url = _override_callback_host(callback_url)
    headers = {"X-Internal-Auth": ASSET_INTERNAL_SECRET} if ASSET_INTERNAL_SECRET else {}
    delay = 1.0
    last = "no attempt"
    for attempt in range(1, 5):
        opened = None
        try:
            # send everything as multipart/form-data (text fields as (None, val))
            parts = [("job_id", (None, job_id)), ("status", (None, status))]
            if status == "SUCCESS" and video_path:
                opened = open(video_path, "rb")
                parts.append(("video", ("video.mp4", opened, "video/mp4")))
            elif error:
                parts.append(("error", (None, str(error))))
            r = requests.post(url, files=parts, headers=headers, timeout=120)
            if 200 <= r.status_code < 300:
                _set(job_id, video_callback="delivered")
                return
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        finally:
            if opened:
                opened.close()
        _set(job_id, video_callback=f"retry {attempt} ({last})")
        time.sleep(delay)
        delay *= 2
    _set(job_id, video_callback=f"failed after retries ({last})")


def run_submit(job_id, image_url, callback_url, seed=None):
    """Background runner for /submit: fetch the presigned image, render, and
    POST the combined mp4 (or the failure) back to asset-service."""
    run = RUNS_DIR / job_id
    run.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    try:
        _set(job_id, status="running", stage="fetching image")
        image_path = run / "reference.png"
        _download_image(image_url, image_path)

        s = int(seed) if seed is not None else random.randint(0, 2**32 - 1)
        timings = _render_core(job_id, run, image_path, SLOWDOWN_DEFAULT, s)
        timings["total_s"] = round(time.monotonic() - t_start, 2)
        _set(job_id, status="done", progress=1.0, stage="done", timings=timings,
             combined_url=f"/api/rotate/result/{job_id}/combined")
        _send_video_callback(callback_url, job_id, "SUCCESS",
                             video_path=run / "combined.mp4")
    except Exception as e:  # noqa: BLE001
        _set(job_id, status="error", error=str(e),
             timings={"total_s": round(time.monotonic() - t_start, 2)})
        _send_video_callback(callback_url, job_id, "FAILED", error=str(e))
    finally:
        with BUSY_LOCK:
            _BUSY["job_id"] = None


# --- API --------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "dual.html")


@app.get("/api/health")
def health():
    return {"comfy_right": comfy_right.is_up(), "comfy_left": comfy_left.is_up(),
            "comfy_right_url": COMFY_URL_A, "comfy_left_url": COMFY_URL_B,
            "device": POSE_DEVICE, "busy": _BUSY["job_id"] is not None,
            "slowdown": {"min": SLOWDOWN_MIN, "max": SLOWDOWN_MAX,
                         "default": SLOWDOWN_DEFAULT}}


class SubmitReq(BaseModel):
    job_id: str
    image_url: str
    callback_url: str
    prompt: str | None = None      # accepted but ignored
    remove_bg: bool | None = None  # accepted but ignored
    seed: int | None = None


@app.post("/submit")
def submit(req: SubmitReq):
    """asset-service entrypoint: accept a render job, run it in the background,
    and POST the result mp4 to callback_url. Returns 202 immediately, or 409 if
    a job is already running (single GPU)."""
    # single-flight: one render at a time (shares the lock with /api/rotate)
    with BUSY_LOCK:
        if _BUSY["job_id"] is not None:
            return JSONResponse(status_code=409, content={
                "job_id": req.job_id, "status": "BUSY",
                "error": "a render is already in progress; retry shortly"})
        _BUSY["job_id"] = req.job_id

    _set(req.job_id, status="queued", stage="submitted")
    threading.Thread(
        target=run_submit,
        args=(req.job_id, req.image_url, req.callback_url, req.seed),
        daemon=True).start()
    return JSONResponse(status_code=202, content={"video_job_id": req.job_id})


@app.post("/api/rotate")
async def rotate(
    image: UploadFile = File(...),
    slowdown: str = Form(""),
    callback_url: str = Form(""),
):
    if not (comfy_right.is_up() and comfy_left.is_up()):
        raise HTTPException(503, "both ComfyUI servers must be reachable "
                                 f"({COMFY_URL_A}, {COMFY_URL_B})")
    try:
        factor = float(slowdown) if slowdown not in (None, "") else SLOWDOWN_DEFAULT
    except ValueError:
        raise HTTPException(400, "slowdown must be a number")
    factor = max(SLOWDOWN_MIN, min(SLOWDOWN_MAX, factor))

    job_id = uuid.uuid4().hex
    # single-flight: reject if a job is already running
    with BUSY_LOCK:
        if _BUSY["job_id"] is not None:
            raise HTTPException(409, "a rotation job is already running; retry shortly")
        _BUSY["job_id"] = job_id

    try:
        run = RUNS_DIR / job_id
        run.mkdir(parents=True, exist_ok=True)
        ext = Path(image.filename or "ref.png").suffix.lower() or ".png"
        image_path = run / f"reference{ext}"
        image_path.write_bytes(await image.read())
    except Exception:
        with BUSY_LOCK:
            _BUSY["job_id"] = None
        raise

    _set(job_id, status="queued", progress=0.0, stage="starting", slowdown=factor)
    threading.Thread(target=run_rotate,
                     args=(job_id, image_path, factor, callback_url or None),
                     daemon=True).start()
    return {"job_id": job_id, "slowdown": factor}


@app.get("/api/rotate/status/{job_id}")
def status(job_id: str):
    job = _get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return JSONResponse(job)


@app.get("/api/rotate/result/{job_id}/{side}")
def result(job_id: str, side: str):
    if side not in ("left", "right", "combined"):
        raise HTTPException(404, "side must be left, right or combined")
    fname = "combined.mp4" if side == "combined" else f"{side}_final.mp4"
    path = (RUNS_DIR / job_id / fname).resolve()
    if path.parent != (RUNS_DIR / job_id).resolve() or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4",
                        filename=f"rotate_{side}_{job_id[:8]}.mp4")
