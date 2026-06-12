#!/usr/bin/env python3
"""
video_edit.py — control-source probing, trimming and uniform decimation.

The control-video workbench flow: the user uploads a (typically 60fps)
recording, trims it, and squashes it to a short low-fps control clip for
Wan VACE. Frames are removed UNIFORMLY across the whole trimmed range:

    n_out = round(duration_knob * fps_knob)          # capped + 4k+1 snapped
    indices = linspace(first_frame, last_frame, n_out)

so the entire trimmed motion always survives — turning the duration knob
down drops more frames (the motion plays faster), the fps knob trades
smoothness for frame budget. Wan VACE wants `length ≡ 1 (mod 4)`, hence the
4k+1 snap (…, 21, 25, 29).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

MAX_FRAMES = 29          # Wan VACE budget used by the current 14B graph
DEFAULT_FPS = 12.0


def probe_video(path: str) -> dict:
    """Return {frames, fps, width, height, duration} via one ffprobe call."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    width, height, rate, n = out  # csv order follows -show_entries order
    num, den = rate.split("/")
    fps = float(num) / float(den)
    frames = int(n)
    return {"frames": frames, "fps": fps, "width": int(width),
            "height": int(height), "duration": frames / fps if fps else 0.0}


def snap_length(n: int, max_frames: int = MAX_FRAMES) -> int:
    """Largest 4k+1 <= min(n, max_frames), floored at 5."""
    n = min(int(n), int(max_frames))
    return max(5, (n - 1) // 4 * 4 + 1)


def pick_indices(start_frame: int, end_frame: int, n_out: int) -> np.ndarray:
    """Uniformly spread n_out source-frame indices over [start, end] incl."""
    return np.linspace(start_frame, end_frame, int(n_out)).round().astype(int)


def plan_decimation(src: dict, *, start: float = 0.0, end: float | None = None,
                    duration: float | None = None, fps: float = DEFAULT_FPS,
                    max_frames: int = MAX_FRAMES) -> dict:
    """Resolve the trim/decimate knobs against a probed source.

    start/end   : trim range in source seconds (end None/<=0 = clip end).
    duration    : desired OUTPUT duration in seconds (None = trimmed length).
    fps         : output frame rate.
    Returns {indices, n_frames, fps, duration} — feed `indices` to the frame
    reader or to ControlVideoPipeline.generate(frame_indices=...).
    """
    src_fps, n_src = src["fps"], src["frames"]
    f0 = int(np.clip(round(start * src_fps), 0, n_src - 1))
    end_s = end if end and end > start else n_src / src_fps
    f1 = int(np.clip(round(end_s * src_fps) - 1, f0, n_src - 1))
    trimmed = (f1 - f0 + 1) / src_fps
    dur = duration if duration and duration > 0 else trimmed
    n = snap_length(min(round(dur * fps), f1 - f0 + 1, max_frames), max_frames)
    return {"indices": pick_indices(f0, f1, n), "n_frames": int(n),
            "fps": float(fps), "duration": n / float(fps)}


def read_frames_at(path: str, indices) -> list[np.ndarray]:
    """Decode exactly the requested frame indices (one sequential pass)."""
    wanted = {}
    for i in indices:
        wanted[int(i)] = wanted.get(int(i), 0) + 1  # linspace may duplicate
    cap = cv2.VideoCapture(str(path))
    frames, idx = {}, 0
    last_needed = max(wanted)
    while idx <= last_needed:
        ok = cap.grab()
        if not ok:
            break
        if idx in wanted:
            ok, f = cap.retrieve()
            if not ok:
                break
            frames[idx] = f
        idx += 1
    cap.release()
    missing = [i for i in wanted if i not in frames]
    if missing:
        raise RuntimeError(f"could not decode frames {missing[:5]}... from {path}")
    return [frames[int(i)] for i in indices]


def write_frames(frames, path: str, fps: float, crf: int = 18,
                 preset: str = "medium") -> str:
    """Encode BGR frames to a browser-friendly mp4 (libx264, yuv420p)."""
    h, w = frames[0].shape[:2]
    w2, h2 = w // 2 * 2, h // 2 * 2  # libx264/yuv420p needs even dimensions
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}",
         "-r", f"{fps:g}", "-i", "-",
         "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(path)],
        stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f[:h2, :w2].tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed encoding {path}")
    return str(path)


def decimate(input_path: str, output_path: str, *, start: float = 0.0,
             end: float | None = None, duration: float | None = None,
             fps: float = DEFAULT_FPS, max_frames: int = MAX_FRAMES,
             crf: int = 18) -> dict:
    """Trim + uniformly decimate a source video to a short control clip.

    Returns the plan dict (indices, n_frames, fps, duration) with `path` set.
    """
    plan = plan_decimation(probe_video(input_path), start=start, end=end,
                           duration=duration, fps=fps, max_frames=max_frames)
    frames = read_frames_at(input_path, plan["indices"])
    write_frames(frames, output_path, plan["fps"], crf=crf)
    plan["path"] = str(output_path)
    plan["indices"] = [int(i) for i in plan["indices"]]
    return plan


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Trim + uniformly decimate a video.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--start", type=float, default=0.0, help="trim start (s)")
    ap.add_argument("--end", type=float, default=None, help="trim end (s)")
    ap.add_argument("--duration", type=float, default=None,
                    help="output duration (s); smaller = more frames removed")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    a = ap.parse_args()
    res = decimate(a.input, a.output, start=a.start, end=a.end,
                   duration=a.duration, fps=a.fps, max_frames=a.max_frames)
    print(json.dumps(res))
