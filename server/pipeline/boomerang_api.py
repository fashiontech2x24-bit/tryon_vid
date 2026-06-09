#!/usr/bin/env python3
r"""
boomerang_api.py — seamless boomerang with a flow-eased join, importable as a function.

    from boomerang_api import boomerang
    out = boomerang("clip.mp4", loop=True)          # -> path to result

Efficiency design
-----------------
A boomerang is: forward frames, then the same frames reversed. The only place
quality work is needed is the apex, where the motion reverses direction. So:

  * ffmpeg builds the forward head and reversed tail natively (C, one pass) and
    does the single re-encode the reversal already requires anyway.
  * Python decodes ONLY the `window+1` frames around the apex — independent of
    clip length — and synthesizes the eased join with bidirectional optical flow.

Frames decoded in Python and frames synthesized are both O(window), not O(N).
The bulk of the video is never touched by Python or re-blended.

Join geometry: a parabolic time-remap (constant deceleration) over ±window
source frames, so velocity eases +1 -> 0 -> -1 like a pendulum at its peak
instead of flipping hard. Everything outside the window is bit-for-bit the source.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _probe(path: str) -> tuple[int, float]:
    """Return (frame_count, fps) via a single ffprobe call."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,r_frame_rate",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    rate, n = out  # csv order follows the -show_entries order
    num, den = rate.split("/")
    return int(n), float(num) / float(den)


def _read_window(path: str, start: int, count: int) -> list[np.ndarray]:
    """Decode exactly `count` frames starting at `start` (frame-accurate)."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) != count:
        raise RuntimeError(f"expected {count} frames at {start}, got {len(frames)}")
    return frames


class _Flow:
    """Synthesize a frame at a fractional position within a small frame list."""

    def __init__(self, frames: list[np.ndarray]):
        self.f = frames
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        h, w = frames[0].shape[:2]
        self.gx, self.gy = np.meshgrid(np.arange(w, dtype=np.float32),
                                       np.arange(h, dtype=np.float32))
        self._fwd: dict[int, np.ndarray] = {}
        self._bwd: dict[int, np.ndarray] = {}

    def _gray(self, i):
        return cv2.cvtColor(self.f[i], cv2.COLOR_BGR2GRAY)

    def at(self, pos: float) -> np.ndarray:
        i = int(np.floor(pos))
        t = pos - i
        if t < 1e-6:                      # integer position -> exact source frame
            return self.f[i]
        if i not in self._fwd:            # flows computed once per adjacent pair
            self._fwd[i] = self.dis.calc(self._gray(i), self._gray(i + 1), None)
            self._bwd[i] = self.dis.calc(self._gray(i + 1), self._gray(i), None)
        fab, fba = self._fwd[i], self._bwd[i]
        wa = cv2.remap(self.f[i], self.gx - t * fab[..., 0], self.gy - t * fab[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(self.f[i + 1], self.gx - (1 - t) * fba[..., 0],
                       self.gy - (1 - t) * fba[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.addWeighted(wa, 1 - t, wb, t, 0)


def boomerang(
    input_path: str,
    output_path: str | None = None,
    *,
    window: int = 3,
    loop: bool = False,
    crf: int = 16,
    preset: str = "slow",
) -> str:
    """Create a seamless boomerang with a flow-eased apex.

    Parameters
    ----------
    input_path  : source video.
    output_path : result path (default: '<input>_boomerang.mp4').
    window      : source frames on each side of the apex to retime (>=1).
    loop        : drop the final frame so the result loops perfectly.
    crf, preset : libx264 quality / speed knobs.

    Returns the output path.
    """
    src = Path(input_path)
    out = Path(output_path) if output_path else src.with_name(src.stem + "_boomerang.mp4")

    n, fps = _probe(str(src))
    if window < 1 or 2 * window >= n:
        raise ValueError(f"window must be in [1, {(n - 1) // 2}] for a {n}-frame clip")

    apex = n - 1
    p0 = apex - window               # parabola entry/exit source position
    last = 1 if loop else 0          # first frame kept in the reversed tail
    m = 4 * window + 1               # parabola length (velocity-continuous ends)

    # --- the only Python decode/synthesis: frames [p0 .. apex] = window+1 frames ---
    sub = _read_window(str(src), p0, window + 1)
    flow = _Flow(sub)
    h, w = sub[0].shape[:2]
    join = [flow.at(k - k * k / (m - 1)) for k in range(m)]   # local positions 0..window

    join_raw = Path(tempfile.mkstemp(suffix=".raw")[1])
    join_raw.write_bytes(b"".join(f.tobytes() for f in join))

    # --- ffmpeg: native head + reversed tail, splice the synthesized join between ---
    # input 0 = source; input 1 = raw join frames.
    fc = (
        f"[0:v]trim=end_frame={p0},setpts=PTS-STARTPTS,format=yuv420p,setsar=1[head];"
        f"[0:v]trim=start_frame={last}:end_frame={p0},reverse,setpts=PTS-STARTPTS,"
        f"format=yuv420p,setsar=1[tail];"
        f"[1:v]setpts=PTS-STARTPTS,format=yuv420p,setsar=1[join];"
        f"[head][join][tail]concat=n=3:v=1[v]"
    )
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-i", str(src),
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", f"{fps:g}", "-i", str(join_raw),
             "-filter_complex", fc, "-map", "[v]",
             "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
             str(out)],
            check=True)
    finally:
        join_raw.unlink(missing_ok=True)

    return str(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Seamless boomerang with flow-eased apex.")
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    a = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg/ffprobe not found on PATH")

    result = boomerang(a.input, a.output, window=a.window, loop=a.loop,
                       crf=a.crf, preset=a.preset)
    print(result)
