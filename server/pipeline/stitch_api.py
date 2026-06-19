#!/usr/bin/env python3
"""
stitch_api.py — compose separately-generated segment clips into one coherent
video.

Why this works seamlessly: every segment is generated from the SAME reference
image with a control video that starts at (or returns to) the reference pose,
so at each junction the two adjacent frames are nearly identical — same pose,
same framing. What remains is texture-level mismatch (garment folds, hair),
which a short bidirectional-optical-flow morph hides.

Per segment:  reverse | boomerang (flow-eased apex, from boomerang_api) | speed.
Per seam:     flow morph from last frame of A to first frame of B; `pause`
              stretches that morph into a slow-mo settle (the "pause" between
              the step-forward and the torso rotations).
Global:       out_fps, crf, optional whole-video slowdown (motion-interpolated).

Everything is done in memory — segments are ≤29 frames each, so a full
3-segment composition is well under 100 frames.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from boomerang_api import _Flow


def load_frames(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames, fps


def boomerang_frames(frames: list[np.ndarray], window: int = 3) -> list[np.ndarray]:
    """forward + flow-eased apex + reversed — same parabolic time-remap as
    boomerang_api, on in-memory frames. Ends back at frames[0]."""
    n = len(frames)
    window = int(np.clip(window, 1, (n - 1) // 2))
    apex = n - 1
    p0 = apex - window
    m = 4 * window + 1
    flow = _Flow(frames[p0:apex + 1])
    join = [flow.at(k - k * k / (m - 1)) for k in range(m)]
    return frames[:p0] + join + frames[p0 - 1::-1]


def seam_morph(fa: np.ndarray, fb: np.ndarray, n: int) -> list[np.ndarray]:
    """n intermediate frames morphing fa -> fb (bidirectional flow), with
    smoothstep easing so velocity is continuous at both ends."""
    if n <= 0:
        return []
    if fa.shape != fb.shape:
        fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]))
    flow = _Flow([fa, fb])
    out = []
    for i in range(1, n + 1):
        u = i / (n + 1)
        t = u * u * (3.0 - 2.0 * u)  # smoothstep
        out.append(flow.at(min(t, 1.0 - 1e-4)))
    return out


def retime_frames(frames: list[np.ndarray], speed: float) -> list[np.ndarray]:
    """Resample a clip to `1/speed` of its length with flow interpolation
    (speed < 1 = slow motion, > 1 = faster)."""
    if abs(speed - 1.0) < 1e-3 or len(frames) < 2:
        return frames
    n_out = max(2, int(round(len(frames) / speed)))
    flow = _Flow(frames)
    pos = np.linspace(0, len(frames) - 1, n_out)
    return [flow.at(min(float(p), len(frames) - 1 - 1e-4)) for p in pos]


def apply_segment_ops(frames: list[np.ndarray], *, reverse: bool = False,
                      boomerang: bool = False, boomerang_window: int = 3,
                      speed: float = 1.0) -> list[np.ndarray]:
    """Op order: reverse -> boomerang -> speed (reversal first so a
    boomerang of a reversed clip pivots on the right apex)."""
    if reverse:
        frames = frames[::-1]
    if boomerang:
        frames = boomerang_frames(frames, boomerang_window)
    if abs(speed - 1.0) > 1e-3:
        frames = retime_frames(frames, speed)
    return frames


def compose(segments: list[dict], output_path: str, *, seams: list[dict] | None = None,
            out_fps: float = 12.0, crf: int = 16, preset: str = "medium",
            slowdown: float = 1.0, final_fps: int = 30) -> dict:
    """Stitch segment clips into one video.

    segments : [{path, reverse?, boomerang?, boomerang_window?, speed?}, ...]
    seams    : len(segments)-1 entries [{window?, pause?}, ...]
               window = morph frames when pause is 0 (default 3);
               pause  = seconds of slow-mo settle at this junction
                        (overrides window: morph frames = pause * out_fps).
    slowdown : >1 slows the WHOLE result via motion-compensated interpolation
               to final_fps (same knob as the single-clip boomerang).

    Returns {path, n_frames, fps, seam_starts} — seam_starts are output frame
    indices where each junction begins (handy for seam preview in the UI).
    """
    seams = seams or [{} for _ in range(len(segments) - 1)]
    if len(seams) != len(segments) - 1:
        raise ValueError(f"need {len(segments) - 1} seam configs, got {len(seams)}")

    processed = []
    base_shape = None
    for seg in segments:
        frames, _ = load_frames(seg["path"])
        if base_shape is None:
            base_shape = frames[0].shape
        elif frames[0].shape != base_shape:
            frames = [cv2.resize(f, (base_shape[1], base_shape[0])) for f in frames]
        processed.append(apply_segment_ops(
            frames, reverse=bool(seg.get("reverse")),
            boomerang=bool(seg.get("boomerang")),
            boomerang_window=int(seg.get("boomerang_window", 3)),
            speed=float(seg.get("speed", 1.0))))

    timeline = list(processed[0])
    seam_starts = []
    for seg_frames, seam in zip(processed[1:], seams):
        pause = float(seam.get("pause", 0.0) or 0.0)
        n_join = (int(round(pause * out_fps)) if pause > 0
                  else int(seam.get("window", 3)))
        seam_starts.append(len(timeline) - 1)
        timeline += seam_morph(timeline[-1], seg_frames[0], n_join)
        timeline += seg_frames

    _encode(timeline, output_path, out_fps, crf=crf, preset=preset,
            slowdown=slowdown, final_fps=final_fps)
    return {"path": str(output_path), "n_frames": len(timeline),
            "fps": out_fps, "seam_starts": seam_starts}


def _encode(frames, path, fps, *, crf=16, preset="medium",
            slowdown=1.0, final_fps=30):
    h, w = frames[0].shape[:2]
    w2, h2 = w // 2 * 2, h // 2 * 2
    vf = "format=yuv420p"
    if slowdown and abs(slowdown - 1.0) > 1e-3:
        vf = (f"setpts={slowdown}*PTS,minterpolate=fps={final_fps}"
              f":mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=none,"
              + vf)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}",
         "-r", f"{fps:g}", "-i", "-",
         "-vf", vf,
         "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
         "-movflags", "+faststart", "-an", str(path)],
        stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f[:h2, :w2]).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed encoding {path}")


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Stitch segment clips: reverse/boomerang/speed per segment, "
                    "flow-morphed seams with optional slow-mo pause.")
    ap.add_argument("output")
    ap.add_argument("--recipe", required=True,
                    help='JSON: {"segments":[...], "seams":[...], "out_fps":12, '
                         '"crf":16, "slowdown":1.0}')
    a = ap.parse_args()
    r = json.loads(a.recipe)
    res = compose(r["segments"], a.output, seams=r.get("seams"),
                  out_fps=float(r.get("out_fps", 12)), crf=int(r.get("crf", 16)),
                  slowdown=float(r.get("slowdown", 1.0)))
    print(json.dumps(res))
