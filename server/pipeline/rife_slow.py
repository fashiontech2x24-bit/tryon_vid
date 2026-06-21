#!/usr/bin/env python3
"""
rife_slow.py — RIFE-based smooth slow motion (Practical-RIFE backend).

The dual-inference pipeline's "slowdown" knob is a continuous slow-mo *factor*
(1.0, 1.2, ...). We keep the output frame rate constant and synthesise extra
in-between frames, so a `factor` of 1.2 turns N frames into round(N * factor)
frames that play over the same fps — i.e. the clip lasts `factor`x longer and
moves in smooth slow motion.

Frames are produced at uniform positions across the source timeline; each
fractional position is interpolated from its two bracketing source frames with
RIFE at the exact sub-frame timestep (no nearest-frame stutter). `factor` 1.0
is a no-op (the input is returned untouched).

Backend: Practical-RIFE (https://github.com/hzwer/Practical-RIFE). Point at a
checkout via the RIFE_DIR env var (or the `rife_dir` arg); setup.sh installs it
under vendor/Practical-RIFE with its train_log/ weights. The torch model is
loaded once per (dir, device) and reused.
"""

from __future__ import annotations

import os
import sys
import threading

import cv2
import numpy as np

import video_edit  # reuse the even-dim / yuv420p mp4 encoder

DEFAULT_RIFE_DIR = os.environ.get(
    "RIFE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 "vendor", "Practical-RIFE"),
)

_MODELS: dict[tuple, "_RifeModel"] = {}
_MODELS_LOCK = threading.Lock()


class _RifeModel:
    """Thin holder around a loaded Practical-RIFE Model (torch, on `device`)."""

    def __init__(self, rife_dir: str, device: str | None = None):
        import torch

        rife_dir = os.path.abspath(rife_dir)
        if not os.path.isdir(os.path.join(rife_dir, "train_log")):
            raise FileNotFoundError(
                f"Practical-RIFE not found at {rife_dir} (expected a train_log/ "
                f"dir with weights). Set RIFE_DIR or run setup.sh.")
        if rife_dir not in sys.path:
            sys.path.insert(0, rife_dir)

        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Practical-RIFE ships the model class under train_log/ (RIFE_HDv3.py);
        # older checkouts expose it as model/RIFE_HDv3.py. Try both.
        Model = None
        for modpath in ("train_log.RIFE_HDv3", "model.RIFE_HDv3", "RIFE_HDv3"):
            try:
                Model = __import__(modpath, fromlist=["Model"]).Model
                break
            except Exception:
                continue
        if Model is None:
            raise ImportError(
                f"could not import Practical-RIFE Model from {rife_dir}")

        self.model = Model()
        self.model.load_model(os.path.join(rife_dir, "train_log"), -1)
        self.model.eval()
        if hasattr(self.model, "device"):
            self.model.device()

    def _to_tensor(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = self.torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def _to_bgr(self, t) -> np.ndarray:
        arr = (t[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0)
        rgb = arr.round().astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def interp(self, a_bgr: np.ndarray, b_bgr: np.ndarray, t: float) -> np.ndarray:
        """Frame at fractional timestep t in (0,1) between a and b (BGR)."""
        torch = self.torch
        h, w = a_bgr.shape[:2]
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        pad = (0, pw - w, 0, ph - h)  # (left, right, top, bottom)
        i0 = torch.nn.functional.pad(self._to_tensor(a_bgr), pad)
        i1 = torch.nn.functional.pad(self._to_tensor(b_bgr), pad)
        with torch.no_grad():
            mid = self.model.inference(i0, i1, t)
        return self._to_bgr(mid[:, :, :h, :w])


def _get_model(rife_dir: str, device: str | None) -> _RifeModel:
    key = (os.path.abspath(rife_dir), device or "auto")
    with _MODELS_LOCK:
        if key not in _MODELS:
            _MODELS[key] = _RifeModel(rife_dir, device)
        return _MODELS[key]


def _read_all_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


def slowmo(input_path: str, output_path: str, factor: float, *,
           fps: float | None = None, rife_dir: str | None = None,
           device: str | None = None, crf: int = 16) -> str:
    """Slow `input_path` down by `factor` via RIFE and write `output_path`.

    factor : >1 slows down (1.0 = no-op, returns input_path unchanged).
    fps    : output fps (default: source fps — more frames at the same fps is
             what produces the slow motion).
    Returns the path that downstream stages should consume.
    """
    factor = float(factor)
    if factor <= 1.0 + 1e-3:
        return input_path  # nothing to interpolate; caller uses the source clip

    src_frames = _read_all_frames(input_path)
    n = len(src_frames)
    if n < 2:
        return input_path

    out_fps = float(fps) if fps else video_edit.probe_video(input_path)["fps"]
    m = max(n, int(round(n * factor)))          # target frame count
    model = _get_model(rife_dir or DEFAULT_RIFE_DIR, device)

    out_frames: list[np.ndarray] = []
    for j in range(m):
        pos = j * (n - 1) / (m - 1)             # source-frame position [0, n-1]
        i = int(np.floor(pos))
        t = pos - i
        if i >= n - 1:                          # last frame, exact
            out_frames.append(src_frames[-1])
        elif t < 1e-4:                          # on a source frame, exact
            out_frames.append(src_frames[i])
        else:
            out_frames.append(model.interp(src_frames[i], src_frames[i + 1], t))

    video_edit.write_frames(out_frames, output_path, out_fps, crf=crf)
    return output_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RIFE smooth slow motion.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--factor", type=float, default=1.5)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--rife_dir", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    print(slowmo(a.input, a.output, a.factor, fps=a.fps,
                 rife_dir=a.rife_dir, device=a.device))
