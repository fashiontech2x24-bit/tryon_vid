# Swapping the rotation control video (dual-inference)

Runbook for replacing the **rotation motion preset** that drives the
`dual-inference` product (one reference image → rotate-left + rotate-right).
Self-contained: you should be able to follow this without any prior chat
context.

---

## What the preset is and where it lives

| Thing | Path / value |
|------|--------------|
| Baked control clip | `assets/motion_presets/rotate.mp4` |
| Retarget config sidecar | `assets/motion_presets/rotate.json` |
| Consumed by | `server/dual_app.py` → `_rotate_pipeline()` / `_retarget_cfg()` |
| Selected via env | `ROTATE_PRESET` (default `rotate.mp4`) |
| Pose cache | `.pose_cache/` (auto-keyed by clip content) |

`rotate.mp4` is **not** a rendered skeleton — it is a short, decimated clip of a
**real person rotating**. At request time the pipeline estimates that clip's
poses once (cached), then retargets the motion onto the user's reference image,
rendering an OpenPose control video. The same preset produces **both**
directions: `dual_app` renders it once normally (`mirror=False` → right) and
once skeleton-mirrored (`mirror=True` → left). So you only ever bake **one**
clip.

## Hard constraints (read before baking)

1. **Frame count must equal `GEN_LENGTH` in `server/dual_app.py` (currently
   `29`).** The app loads the control 1:1 (`force_rate=12`, `length=29`) and a
   mismatch breaks generation. If you bake a different length you MUST update
   `GEN_LENGTH` (and keep it on the grid below).
2. **Length must be on Wan's `4k+1` grid and ≤ 29** for the 14B graph:
   `5, 9, 13, 17, 21, 25, 29`. `video_edit.snap_length()` enforces this.
3. **fps is 12** (`video_edit.DEFAULT_FPS`, and `GEN_FPS` in `dual_app.py`).
4. **Direction matters.** The control clip should play **neutral pose →
   rotated** so the boomerang reads neutral→rotated→neutral. The original
   source (`IMG_5332.mp4`) was recorded rotated→neutral, so it was baked with
   `reverse=True`. Check your new clip's direction and set `reverse`
   accordingly.

## Step 1 — get the new source clip

Any video of a person doing the rotation you want. Note:
- which **time range** holds the good motion (e.g. `0–4 s`), and
- whether it plays **neutral→rotated** (`reverse=False`) or **rotated→neutral**
  (`reverse=True`).

Put it somewhere readable, e.g. `/path/to/new_rotation.mp4`.

## Step 2 — bake the preset

The bake uses `server/pipeline/video_edit.py::decimate(...)` (trim → uniform
decimation → 4k+1 snap → optional reverse). Use a Python that has `opencv` +
`numpy` and `ffmpeg` on PATH (locally that was `.venv/bin/python`; on the box,
the ComfyUI interpreter works too).

```bash
# from the repo root
.venv/bin/python - <<'PY'
import sys, json
sys.path.insert(0, "server/pipeline")
import video_edit as ve

SRC     = "/path/to/new_rotation.mp4"   # <-- your new source
OUT     = "assets/motion_presets/rotate.mp4"
START   = 0.0      # trim start (s)
END     = 4.0      # trim end (s); None = clip end
REVERSE = True     # True if the source plays rotated->neutral

plan = ve.decimate(SRC, OUT, start=START, end=END, duration=None,
                   fps=12.0, max_frames=29, reverse=REVERSE)
print("plan:", {k: v for k, v in plan.items() if k != "indices"})

# retarget config (validated default; keep blend_frames=0, head_lock keeps the
# face toward camera while the torso turns)
sidecar = {
    "source": SRC.split("/")[-1],
    "decimate": {"start": START, "end": END, "duration": 0.0,
                 "reverse": REVERSE, "fps": plan["fps"],
                 "n_frames": plan["n_frames"]},
    "retarget": {"pose_mode": "absolute", "root_motion": 1.0, "smoothing": 0.4,
                 "foreshorten": 1.0, "blend_frames": 0, "head_lock": 0.3},
}
open("assets/motion_presets/rotate.json", "w").write(json.dumps(sidecar, indent=2))
print("wrote rotate.json")
PY
```

### Retarget config knobs (`rotate.json` → `retarget`)
`dual_app._retarget_cfg()` reads this sidecar (falls back to these defaults).
- `pose_mode`: `absolute` (follow the clip's joint angles) vs `relative`.
- `root_motion` (0–1+): how much whole-body translation is transferred.
- `smoothing` (0–1): temporal smoothing of the retargeted skeleton.
- `foreshorten` (0–1): depth foreshortening as limbs rotate toward camera.
- `blend_frames`: keep **0** for this product (no blend-in ramp).
- `head_lock` (0–1): `0.3` pins the face toward camera while the body turns;
  `0.0` = head turns with the torso (natural full spin).

To tune these interactively, use the demo app's **Control Studio**
(`server/app.py` → `/api/control/pose_preview`): upload the source, trim, tweak
the retarget sliders against a reference image, watch the skeleton preview, then
copy the values into `rotate.json`.

## Step 3 — invalidate the pose cache

The pose cache key is derived from the clip's content (path + size + mtime), so
replacing `rotate.mp4` **auto-invalidates** it — the first request after the
swap re-estimates the new clip's poses (slow once, then cached). To force a
clean slate:

```bash
rm -f .pose_cache/control_*_wb.npz   # or: rm -rf .pose_cache
```

## Step 4 — verify

```bash
# frame count must be 29 (or your new GEN_LENGTH) and fps 12
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=nb_read_frames,r_frame_rate -of csv=p=0 \
  assets/motion_presets/rotate.mp4
```

Full orchestration smoke test (no GPU; mocks ComfyUI, stubs pose, RIFE bypassed
at slowdown 1.0):

```bash
.venv/bin/python localtest/test_dual_e2e.py
```

Then a real run: open the web app, upload a reference image, set slowdown > 1.0,
and confirm both rotate-left and rotate-right look correct (and that "left" vs
"right" map the way you expect — if reversed, the mirror simply maps to the
other side; relabel by swapping which control is sent to which ComfyUI in
`dual_app.run_rotate`, or accept the labels).

## Step 5 — commit the new asset

```bash
git add assets/motion_presets/rotate.mp4 assets/motion_presets/rotate.json
git commit -m "swap rotation preset (<describe new clip>)"
git push
```

`assets/motion_presets/*.mp4` and `*.json` are explicitly un-ignored in
`.gitignore`, so the preset is committed despite the global `*.mp4` rule.

---

## If you change the length (not just the clip)

1. Pick a `4k+1` value ≤ 29 (e.g. 25).
2. Bake with `max_frames=25` (and `duration`/`fps` such that the snap lands there).
3. Set `GEN_LENGTH = 25` in `server/dual_app.py`.
4. Re-run `localtest/test_dual_e2e.py`.

Going **above 29** means re-exporting the ComfyUI graph (`workflow_api.json`)
for a longer budget and more VRAM — out of scope for a preset swap.
