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
import hashlib
import json
import os
import random
import re
import sys
import threading
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
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
import video_edit                       # noqa: E402
from boomerang_api import boomerang     # noqa: E402
from stitch_api import compose as stitch_compose  # noqa: E402

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
PRESETS_DIR = Path(os.environ.get("PRESETS_DIR", ROOT / "assets" / "motion_presets")).resolve()
SOURCES_DIR = Path(os.environ.get("SOURCES_DIR", ROOT / "assets" / "control_sources")).resolve()
RUNS_DIR = Path(os.environ.get("RUNS_DIR", ROOT / "runs")).resolve()
PREVIEWS_DIR = RUNS_DIR / "_previews"
MODEL_DIR = os.environ.get("DWPOSE_MODEL_DIR", str(ROOT / "models" / "dwpose"))
CACHE_DIR = os.environ.get("POSE_CACHE_DIR", str(ROOT / ".pose_cache"))


def _resolve_pose_device(requested: str) -> str:
    """'auto' (the default) picks cuda whenever onnxruntime ships the CUDA
    provider (setup.sh installs onnxruntime-gpu on GPU boxes); otherwise cpu.
    An explicit cpu/mps/cuda request is honored, with a cpu fallback + warning
    when cuda isn't actually available."""
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
        print("[app] POSE_DEVICE=cuda requested but onnxruntime has no CUDA "
              "provider — falling back to cpu (install onnxruntime-gpu)")
    return "cpu"


POSE_DEVICE = _resolve_pose_device(os.environ.get("POSE_DEVICE", "auto"))
VID_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

RUNS_DIR.mkdir(parents=True, exist_ok=True)
PRESETS_DIR.mkdir(parents=True, exist_ok=True)
SOURCES_DIR.mkdir(parents=True, exist_ok=True)
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

# Pose-retarget config "E" — the validated default; preset sidecars override it.
RETARGET_DEFAULTS = {"pose_mode": "absolute", "root_motion": 1.0,
                     "smoothing": 0.4, "foreshorten": 1.0, "blend_frames": 8,
                     "head_lock": 0.0}

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


def _pipeline_for(clip: Path):
    """Lazily build + cache one ControlVideoPipeline per clip (preset OR
    uploaded source). First use estimates the clip's poses (disk-cached)."""
    key = str(clip.resolve())
    if key not in _PIPELINES:
        if not clip.exists():
            raise FileNotFoundError(f"clip not found: {clip}")
        _PIPELINES[key] = pose_pipeline.ControlVideoPipeline(
            key, device=POSE_DEVICE, model_dir=MODEL_DIR, cache_dir=CACHE_DIR, verbose=True)
    return _PIPELINES[key]


def get_pipeline(preset):
    return _pipeline_for(PRESETS_DIR / preset)


def load_preset_config(preset):
    """Retarget config for a preset: config E + the sidecar JSON's overrides
    (written by the control workbench's 'save preset')."""
    cfg = dict(RETARGET_DEFAULTS)
    sidecar = PRESETS_DIR / (Path(preset).stem + ".json")
    if sidecar.exists():
        try:
            cfg.update(json.loads(sidecar.read_text()).get("retarget", {}))
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


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
            # Config "E" (chosen from local testing): absolute pose, full root
            # motion + foreshortening, short blend-in. Overridable via form, but
            # the UI no longer sends these — E is the fixed default.
            pipe.generate(
                str(image_path), str(control_path),
                pose_mode=opts.get("pose_mode") or "absolute",
                root_motion=_fnum(opts.get("root_motion"), 1.0),
                smoothing=_fnum(opts.get("smoothing"), 0.4),
                foreshorten=_fnum(opts.get("foreshorten"), 1.0),
                blend_frames=int(_fnum(opts.get("blend_frames"), 8)))
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
                      crf=int(_fnum(opts.get("crf"), 16)), loop=loop,
                      slowdown=_fnum(opts.get("slowdown"), 1.5))
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
    # preprocess (pose retarget) — fixed to config "E" server-side; these stay
    # accepted for power users / API callers but the UI no longer sends them.
    pose_mode: str = Form("absolute"),
    root_motion: str = Form(""), smoothing: str = Form(""),
    foreshorten: str = Form(""), blend_frames: str = Form(""),
    # generate (VACE)
    seed: str = Form(""), positive: str = Form(""), negative: str = Form(""),
    width: str = Form(""), height: str = Form(""), length: str = Form(""),
    steps: str = Form(""), cfg: str = Form(""), lora_strength: str = Form(""),
    fps: str = Form(""), shift: str = Form(""),
    # postprocess (boomerang)
    window: str = Form(""), crf: str = Form(""), loop: str = Form("true"),
    slowdown: str = Form(""),
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
                shift=shift, window=window, crf=crf, loop=loop, slowdown=slowdown)
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
    # (compose jobs produce control_<k>.mp4 / gen_<k>.mp4 per unique control)
    if not re.fullmatch(r"(control|gen|generated|final)(_\d+)?\.mp4", filename):
        raise HTTPException(404, "unknown artifact")
    path = (RUNS_DIR / job_id / filename).resolve()
    if path.parent != (RUNS_DIR / job_id).resolve() or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4")


# ===========================================================================
# Control-video workbench: upload a source clip, trim + uniformly decimate it
# (duration/fps knobs), preview the retargeted skeleton, save as a preset.
# ===========================================================================
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")
_PROBE_CACHE: dict[tuple, dict] = {}


def _safe_name(name, fallback="clip"):
    return _SAFE_NAME.sub("_", name or "").strip("._") or fallback


def _probe_cached(path: Path) -> dict:
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _PROBE_CACHE:
        _PROBE_CACHE[key] = video_edit.probe_video(str(path))
    return _PROBE_CACHE[key]


def _source_path(source_id: str) -> Path:
    p = (SOURCES_DIR / source_id).resolve()
    if p.parent != SOURCES_DIR or not p.is_file():
        raise HTTPException(404, f"unknown source: {source_id}")
    return p


def _truthy(v):
    return str(v).lower() in ("1", "true", "on", "yes")


def _decimated_cached(src: Path, start, end, duration, fps, reverse=False):
    """Trim+decimate `src` into a deterministic cache file under _previews/.

    The file name is keyed on the exact frame selection, so repeat calls with
    the same knobs reuse it — and so does the pose pipeline's disk cache,
    meaning pose estimation only ever runs on the ≤29 decimated frames.
    Returns (path, plan).
    """
    plan = video_edit.plan_decimation(
        _probe_cached(src), start=_fnum(start, 0.0),
        end=_fnum(end, 0.0) or None, duration=_fnum(duration, 0.0) or None,
        fps=_fnum(fps, video_edit.DEFAULT_FPS), reverse=reverse)
    st = src.stat()
    key = hashlib.sha1(
        f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}|"
        f"{list(plan['indices'])}|{plan['fps']}".encode()).hexdigest()[:32]
    out = PREVIEWS_DIR / f"{key}_dec.mp4"
    if not out.exists():
        frames = video_edit.read_frames_at(str(src), plan["indices"])
        video_edit.write_frames(frames, str(out), plan["fps"])
    plan["indices"] = [int(i) for i in plan["indices"]]
    plan["path"] = str(out)
    return out, plan


def _retarget_from_form(pose_mode, root_motion, smoothing, foreshorten,
                        blend_frames, head_lock):
    cfg = dict(RETARGET_DEFAULTS)
    if pose_mode in ("relative", "absolute"):
        cfg["pose_mode"] = pose_mode
    cfg["root_motion"] = _fnum(root_motion, cfg["root_motion"])
    cfg["smoothing"] = _fnum(smoothing, cfg["smoothing"])
    cfg["foreshorten"] = _fnum(foreshorten, cfg["foreshorten"])
    cfg["blend_frames"] = int(_fnum(blend_frames, cfg["blend_frames"]))
    cfg["head_lock"] = _fnum(head_lock, cfg["head_lock"])
    return cfg


@app.post("/api/control/upload")
async def control_upload(video: UploadFile = File(...)):
    name = Path(video.filename or "source.mp4")
    if name.suffix.lower() not in VID_EXT:
        raise HTTPException(400, f"unsupported video type: {name.suffix}")
    source_id = f"{uuid.uuid4().hex[:8]}_{_safe_name(name.stem)}{name.suffix.lower()}"
    path = SOURCES_DIR / source_id
    path.write_bytes(await video.read())
    try:
        info = _probe_cached(path)
    except Exception as e:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"cannot probe video: {e}")
    return {"source_id": source_id, **info}


@app.get("/api/control/sources")
def control_sources():
    out = []
    for p in sorted(SOURCES_DIR.iterdir()):
        if p.suffix.lower() in VID_EXT:
            try:
                out.append({"source_id": p.name, **_probe_cached(p)})
            except Exception:
                continue
    return {"sources": out}


@app.get("/api/control/source/{source_id}")
def control_source(source_id: str):
    return FileResponse(_source_path(source_id), media_type="video/mp4")


@app.get("/api/wb/{filename}")
def workbench_file(filename: str):
    # workbench previews (uuid-named, written by the endpoints below)
    if not re.fullmatch(r"[a-f0-9]{32}_(dec|pose)\.mp4", filename):
        raise HTTPException(404, "unknown preview")
    path = PREVIEWS_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/control/preview")
async def control_preview(source_id: str = Form(...), start: str = Form(""),
                          end: str = Form(""), duration: str = Form(""),
                          fps: str = Form(""), reverse: str = Form("false")):
    """Trim + uniformly decimate -> preview clip (no GPU, no pose model)."""
    src = _source_path(source_id)
    try:
        out, plan = _decimated_cached(src, start, end, duration, fps,
                                      reverse=_truthy(reverse))
    except Exception as e:
        raise HTTPException(400, f"decimation failed: {e}")
    plan["url"] = f"/api/wb/{out.name}"
    return plan


@app.post("/api/control/pose_preview")
async def control_pose_preview(
    image: UploadFile = File(...), source_id: str = Form(""),
    preset: str = Form(""), start: str = Form(""), end: str = Form(""),
    duration: str = Form(""), fps: str = Form(""),
    pose_mode: str = Form(""), root_motion: str = Form(""),
    smoothing: str = Form(""), foreshorten: str = Form(""),
    blend_frames: str = Form(""), head_lock: str = Form(""),
    mirror: str = Form("false"), reverse: str = Form("false"),
):
    """Retargeted-skeleton preview of a (trimmed/decimated) source or preset
    against a reference image. The source is decimated FIRST, so pose
    estimation only runs on the ≤29 selected frames (disk-cached per
    trim/decimate setting)."""
    import cv2
    import numpy as np
    cfg = _retarget_from_form(pose_mode, root_motion, smoothing, foreshorten,
                              blend_frames, head_lock)
    if source_id:
        try:
            clip, plan = _decimated_cached(_source_path(source_id), start, end,
                                           duration, fps, reverse=_truthy(reverse))
        except Exception as e:
            raise HTTPException(400, f"decimation failed: {e}")
        indices = None  # the clip IS the selection — estimate it directly
    elif preset in list_presets():
        clip = PRESETS_DIR / preset
        plan = video_edit.plan_decimation(
            _probe_cached(clip), start=_fnum(start, 0.0),
            end=_fnum(end, 0.0) or None, duration=_fnum(duration, 0.0) or None,
            fps=_fnum(fps, video_edit.DEFAULT_FPS))
        indices = plan["indices"]
    else:
        raise HTTPException(400, "pass source_id or a valid preset")

    data = np.frombuffer(await image.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "cannot decode reference image")

    out = PREVIEWS_DIR / f"{uuid.uuid4().hex}_pose.mp4"
    try:
        with POSE_LOCK:
            pipe = _pipeline_for(clip)
            pipe.generate(img, str(out), fps=plan["fps"],
                          frame_indices=indices,
                          mirror=str(mirror).lower() in ("1", "true", "on"),
                          **cfg)
    except Exception as e:
        raise HTTPException(500, f"pose preview failed: {e}")
    return {"url": f"/api/wb/{out.name}", "n_frames": plan["n_frames"],
            "fps": plan["fps"], "duration": plan["duration"]}


@app.post("/api/control/save_preset")
async def control_save_preset(
    source_id: str = Form(...), name: str = Form(...), start: str = Form(""),
    end: str = Form(""), duration: str = Form(""), fps: str = Form(""),
    pose_mode: str = Form(""), root_motion: str = Form(""),
    smoothing: str = Form(""), foreshorten: str = Form(""),
    blend_frames: str = Form(""), head_lock: str = Form(""),
    reverse: str = Form("false"),
):
    """Write the decimated clip into assets/motion_presets/ plus a JSON
    sidecar holding its retarget config (used by compose jobs). The clip is
    baked pose0-first when `reverse` is set, so compose needs no extra step."""
    src = _source_path(source_id)
    stem = _safe_name(Path(name).stem, "preset")
    out = PRESETS_DIR / f"{stem}.mp4"
    rev = _truthy(reverse)
    try:
        plan = video_edit.decimate(
            str(src), str(out), start=_fnum(start, 0.0),
            end=_fnum(end, 0.0) or None, duration=_fnum(duration, 0.0) or None,
            fps=_fnum(fps, video_edit.DEFAULT_FPS), reverse=rev)
    except Exception as e:
        raise HTTPException(400, f"decimation failed: {e}")
    cfg = _retarget_from_form(pose_mode, root_motion, smoothing, foreshorten,
                              blend_frames, head_lock)
    sidecar = {"source": source_id,
               "decimate": {"start": _fnum(start, 0.0), "end": _fnum(end, 0.0),
                            "duration": _fnum(duration, 0.0), "reverse": rev,
                            "fps": plan["fps"], "n_frames": plan["n_frames"]},
               "retarget": cfg}
    (PRESETS_DIR / f"{stem}.json").write_text(json.dumps(sidecar, indent=2))
    # a previously cached pipeline for this preset name is now stale
    _PIPELINES.pop(str(out.resolve()), None)
    return {"preset": out.name, **{k: v for k, v in plan.items() if k != "indices"}}


# ===========================================================================
# Compose: N segments (preset + mirror/reverse/boomerang/speed) -> N unique
# generations (same seed + reference) -> flow-morphed stitch -> final.mp4
# ===========================================================================
def _parse_recipe(raw: str) -> dict:
    try:
        r = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"recipe is not valid JSON: {e}")
    segs = r.get("segments") or []
    if not segs:
        raise HTTPException(400, "recipe.segments is empty")
    have = list_presets()
    for s in segs:
        if s.get("preset") not in have:
            raise HTTPException(400, f"unknown preset {s.get('preset')!r} (have: {have})")
    seams = r.get("seams") or [{} for _ in segs[1:]]
    if len(seams) != len(segs) - 1:
        raise HTTPException(400, f"need {len(segs) - 1} seam configs, got {len(seams)}")
    return {"segments": segs, "seams": seams,
            "stitch": r.get("stitch") or {}, "gen": r.get("gen") or {}}


def _control_fps(preset, pipe):
    sidecar = PRESETS_DIR / (Path(preset).stem + ".json")
    if sidecar.exists():
        try:
            return float(json.loads(sidecar.read_text())["decimate"]["fps"])
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            pass
    return float(pipe.fps)


def _do_stitch(run: Path, recipe: dict) -> dict:
    segs, seg_gen = recipe["segments"], recipe["seg_gen"]
    seg_dicts = [{"path": str(run / f"gen_{seg_gen[i]}.mp4"),
                  "reverse": bool(s.get("reverse")),
                  "boomerang": bool(s.get("boomerang")),
                  "boomerang_window": int(s.get("boomerang_window", 3)),
                  "speed": float(s.get("speed", 1.0))}
                 for i, s in enumerate(segs)]
    st = recipe.get("stitch") or {}
    return stitch_compose(
        seg_dicts, str(run / "final.mp4"), seams=recipe["seams"],
        out_fps=_fnum(st.get("out_fps"), recipe.get("ctrl_fps", 12.0)),
        crf=int(_fnum(st.get("crf"), 16)),
        slowdown=_fnum(st.get("slowdown"), 1.0))


def run_compose(job_id, image_path, recipe):
    run = RUNS_DIR / job_id
    try:
        segs = recipe["segments"]
        # unique (preset, mirror) pairs -> one control video + generation each
        uniq, seg_gen = [], []
        for s in segs:
            key = (s["preset"], bool(s.get("mirror")))
            if key not in uniq:
                uniq.append(key)
            seg_gen.append(uniq.index(key))
        recipe["seg_gen"] = seg_gen

        gen = recipe["gen"]
        seed = gen.get("seed")
        gen["seed"] = int(seed) if seed not in (None, "") else random.randint(0, 2**32 - 1)

        # -- [1] one control video per unique (preset, mirror) ----------------
        controls = []
        for k, (preset, mirror) in enumerate(uniq):
            _set(job_id, status="running", progress=0.02 + 0.08 * k / len(uniq),
                 stage=f"preprocess {k + 1}/{len(uniq)}: retarget {preset}"
                       + (" (mirrored)" if mirror else ""))
            cfg = load_preset_config(preset)
            ctrl = run / f"control_{k}.mp4"
            with POSE_LOCK:
                pipe = get_pipeline(preset)
                n = len(pipe.control_poses)
                n_snap = video_edit.snap_length(n)
                idx = video_edit.pick_indices(0, n - 1, n_snap) if n_snap != n else None
                ctrl_fps = _control_fps(preset, pipe)
                pipe.generate(
                    str(image_path), str(ctrl), fps=ctrl_fps, mirror=mirror,
                    frame_indices=idx,
                    pose_mode=cfg["pose_mode"],
                    root_motion=_fnum(cfg.get("root_motion"), 1.0),
                    smoothing=_fnum(cfg.get("smoothing"), 0.4),
                    foreshorten=_fnum(cfg.get("foreshorten"), 1.0),
                    blend_frames=int(_fnum(cfg.get("blend_frames"), 8)),
                    head_lock=_fnum(cfg.get("head_lock"), 0.0))
            controls.append((ctrl, n_snap, ctrl_fps))
            _set(job_id, **{f"control_{k}_url": f"/api/run/{job_id}/control_{k}.mp4"})
        recipe["ctrl_fps"] = controls[0][2]

        # -- [2] one VACE generation per control (sequential, same seed) ------
        img_name = f"{job_id}_{Path(image_path).name}"
        stored_img = comfy.upload_file(image_path.read_bytes(), img_name, "image/png")
        m = len(controls)
        for k, (ctrl, n_frames, ctrl_fps) in enumerate(controls):
            stored_vid = comfy.upload_file(ctrl.read_bytes(),
                                           f"{job_id}_control_{k}.mp4", "video/mp4")
            opts = dict(gen)
            opts["length"] = n_frames
            opts["fps"] = int(round(ctrl_fps))
            wf = build_workflow(stored_img, stored_vid, opts)
            # VHS must load the control 1:1 (no resampling) or length mismatches
            wf[NODE_LOAD_VIDEO]["inputs"]["force_rate"] = int(round(ctrl_fps))
            base = 0.12 + 0.72 * k / m

            def on_progress(p, base=base, k=k):
                if p.get("phase") == "sampling" and p.get("max"):
                    frac = p["value"] / p["max"]
                    _set(job_id, progress=round(base + 0.72 / m * frac, 3),
                         stage=f"generating {k + 1}/{m}: {p['value']}/{p['max']}")

            _set(job_id, progress=base, stage=f"generating {k + 1}/{m} (Wan VACE)")
            history = comfy.run(wf, f"{job_id}_{k}", on_progress)
            found = ComfyClient.find_output_video(history)
            if not found:
                _set(job_id, status="error",
                     error=f"No output produced by ComfyUI for segment {k + 1}.")
                return
            filename, subfolder, ftype = found
            (run / f"gen_{k}.mp4").write_bytes(comfy.get_file(filename, subfolder, ftype))
            _set(job_id, **{f"gen_{k}_url": f"/api/run/{job_id}/gen_{k}.mp4"})

        # -- [3] stitch --------------------------------------------------------
        _set(job_id, progress=0.88, stage="stitching segments")
        (run / "recipe.json").write_text(json.dumps(recipe, indent=2))
        info = _do_stitch(run, recipe)
        _set(job_id, status="done", progress=1.0, stage="done",
             output=str(run / "final.mp4"),
             output_url=f"/api/run/{job_id}/final.mp4",
             n_generations=m, stitch_info=info)
    except Exception as e:
        _set(job_id, status="error", error=str(e))


@app.post("/api/compose")
async def compose_endpoint(image: UploadFile = File(...), recipe: str = Form(...)):
    parsed = _parse_recipe(recipe)
    if not comfy.is_up():
        raise HTTPException(503, f"ComfyUI is not reachable at {COMFY_URL}")
    job_id = uuid.uuid4().hex
    run = RUNS_DIR / job_id
    run.mkdir(parents=True, exist_ok=True)
    ext = Path(image.filename or "ref.png").suffix.lower() or ".png"
    image_path = run / f"reference{ext}"
    image_path.write_bytes(await image.read())
    _set(job_id, status="queued", progress=0.0, stage="starting",
         kind="compose")
    threading.Thread(target=run_compose, args=(job_id, image_path, parsed),
                     daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/restitch/{job_id}")
def restitch(job_id: str, payload: dict = Body(...)):
    """Re-run only the stitch stage of a finished compose job with new knobs
    (segment ops / seams / out_fps / crf / slowdown) — no regeneration."""
    run = RUNS_DIR / job_id
    recipe_path = run / "recipe.json"
    if not recipe_path.is_file():
        raise HTTPException(404, "no compose recipe for this job")
    recipe = json.loads(recipe_path.read_text())
    if not all((run / f"gen_{k}.mp4").is_file() for k in set(recipe["seg_gen"])):
        raise HTTPException(409, "generated segments not (yet) available")

    # merge: only stitch-stage fields may change; preset/mirror are baked into
    # the generated files and stay as generated.
    for i, s in enumerate(payload.get("segments") or []):
        if i < len(recipe["segments"]):
            for f in ("reverse", "boomerang", "boomerang_window", "speed"):
                if f in s:
                    recipe["segments"][i][f] = s[f]
    if payload.get("seams") is not None:
        if len(payload["seams"]) != len(recipe["segments"]) - 1:
            raise HTTPException(400, "wrong number of seam configs")
        recipe["seams"] = payload["seams"]
    recipe["stitch"] = {**recipe.get("stitch", {}), **(payload.get("stitch") or {})}

    try:
        info = _do_stitch(run, recipe)
    except Exception as e:
        raise HTTPException(500, f"stitch failed: {e}")
    recipe_path.write_text(json.dumps(recipe, indent=2))
    _set(job_id, output=str(run / "final.mp4"), stitch_info=info)
    return {"output_url": f"/api/run/{job_id}/final.mp4?v={uuid.uuid4().hex[:6]}",
            "stitch_info": info}


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
