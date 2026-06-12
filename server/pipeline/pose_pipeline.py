#!/usr/bin/env python3
"""
pose_pipeline.py — function API for the VTON control-video pipeline step.

Turns a user image + a fixed control clip (e.g. shot3.mp4) into a rendered
OpenPose skeleton **control video** (mp4) that a downstream model (Wan2.1
VACE) consumes as ``control_video``. The motion comes from the control clip;
the body proportions, scale and framing come from the user image — see
``pose_retarget.py`` for the retargeting maths.

Two entry points:

    # one-shot — simplest for a single call
    from pose_pipeline import generate_control_video
    out_mp4 = generate_control_video("user.png", "shot3.mp4", "control.mp4")

    # reusable — load the model + control clip ONCE, call many times
    from pose_pipeline import ControlVideoPipeline
    pipe = ControlVideoPipeline("shot3.mp4", device="cuda")   # heavy work here
    for user_img in batch:
        pipe.generate(user_img, f"out/{user_img}.mp4")        # cheap per call

Optimization: the control clip is constant across pipeline runs, so its
per-frame pose estimation (the expensive part) is done once and disk-cached
keyed by clip content + estimation params. Per call we only estimate the one
user image and run the cheap retargeting + render.

Dependencies:  pip install rtmlib onnxruntime-gpu opencv-python numpy
"""

import hashlib
import os

import cv2
import numpy as np

import pose_retarget as pr


# --------------------------------------------------------------------------
# Control-clip pose cache (estimate the constant control video only once)
# --------------------------------------------------------------------------
def _control_cache_key(control_video, skip_frames, frame_load_cap,
                       select_every_nth, kpt_thr, mode):
    """Stable key over the clip's content + every parameter that changes the
    estimated poses. Retarget params are NOT included — they are applied per
    user image, after the cache."""
    st = os.stat(control_video)
    raw = "|".join(str(x) for x in (
        os.path.abspath(control_video), st.st_size, st.st_mtime_ns,
        skip_frames, frame_load_cap, select_every_nth, kpt_thr, mode,
    ))
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _load_cached_poses(cache_path):
    if not os.path.exists(cache_path):
        return None
    data = np.load(cache_path)
    kpts, conf = data["kpts"], data["conf"]
    poses = [(kpts[i], conf[i]) for i in range(len(kpts))]
    return poses, float(data["fps"]), int(data["n_miss"])


def _save_cached_poses(cache_path, poses, fps, n_miss):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    kpts = np.stack([k for k, _ in poses]).astype(np.float32)
    conf = np.stack([c for _, c in poses]).astype(np.float32)
    # savez appends .npz unless the path already ends with it; tmp does, so the
    # written file is exactly `tmp` and the rename is atomic.
    tmp = cache_path + ".tmp.npz"
    np.savez(tmp, kpts=kpts, conf=conf, fps=np.float32(fps),
             n_miss=np.int32(n_miss))
    os.replace(tmp, cache_path)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
class ControlVideoPipeline:
    """Reusable control-video generator.

    Construct once (loads the ONNX pose model and estimates + caches the
    control clip's poses), then call :meth:`generate` per user image.

    Parameters
    ----------
    control_video : str
        Path to the constant motion clip (e.g. ``shot3.mp4``).
    device : str
        ``"cuda"`` (default, needs onnxruntime-gpu), ``"cpu"`` or ``"mps"``.
    model_dir : str
        Where the DWPose ONNX models are cached.
    cache_dir : str
        Where the control clip's estimated poses are cached.
    skip_frames, frame_load_cap, select_every_nth : int
        VHS-style frame selection on the control clip (see pose_retarget).
    kpt_thr : float
        Detector keypoint confidence threshold.
    """

    def __init__(self, control_video="shot3.mp4", *, device="cuda",
                 model_dir="models", cache_dir=".pose_cache",
                 skip_frames=0, frame_load_cap=0, select_every_nth=1,
                 kpt_thr=0.3, mode="dwpose", verbose=True):
        if not os.path.exists(control_video):
            raise FileNotFoundError(f"Control video not found: {control_video}")
        self.control_video = control_video
        self.verbose = verbose
        self._log(f"loading pose model (mode={mode}, device={device}) ...")
        self.estimator = pr.PoseEstimator(mode=mode, device=device,
                                          kpt_thr=kpt_thr, model_dir=model_dir)

        key = _control_cache_key(control_video, skip_frames, frame_load_cap,
                                 select_every_nth, kpt_thr, mode)
        cache_path = os.path.join(cache_dir, f"control_{key}.npz")
        cached = _load_cached_poses(cache_path)
        if cached is not None:
            self.control_poses, self.fps, n_miss = cached
            self._log(f"control poses loaded from cache "
                      f"({len(self.control_poses)} frames) -> {cache_path}")
        else:
            self._log(f"estimating control poses from {control_video} ...")
            self.control_poses, self.fps, n_miss = pr.estimate_video_poses(
                self.estimator, control_video, skip_frames,
                frame_load_cap, select_every_nth, progress=verbose)
            if not self.control_poses:
                raise RuntimeError(f"No frames loaded from {control_video}")
            _save_cached_poses(cache_path, self.control_poses, self.fps, n_miss)
            self._log(f"control poses cached -> {cache_path}")
        if n_miss:
            self._log(f"warning: no person in {n_miss}/{len(self.control_poses)} "
                      f"control frames (pose held)")

    def _log(self, msg):
        if self.verbose:
            print(f"[pose_pipeline] {msg}")

    def generate(self, user_image, output, *, pose_mode="relative",
                 root_motion=1.0, smoothing=0.4, blend_frames=0,
                 foreshorten=0.0, fps=None, mirror=False, head_lock=0.0,
                 frame_indices=None):
        """Retarget the control clip's motion onto ``user_image`` and render
        the skeleton control video.

        Parameters
        ----------
        user_image : str | np.ndarray
            Path to the user/VTON image, or a BGR image array. Defines the
            output skeleton's bone lengths, scale, position and canvas size.
        output : str
            Output mp4 path. The control video for the downstream model.
        pose_mode : str
            ``"relative"`` (default): keep the user's rest pose, add the
            control clip's motion on top. ``"absolute"``: follow the control
            clip's actual joint angles (user proportions/framing preserved).
        root_motion, smoothing, blend_frames, foreshorten
            Retargeting controls — see pose_retarget / README.
        fps : float, optional
            Output frame rate. Defaults to the control clip's fps.
        mirror : bool
            Flip the control motion left<->right (skeleton-level mirror) —
            e.g. turn a clockwise torso rotation into a counter-clockwise one.
        head_lock : float
            0..1 — pin the head to the user image's orientation while the
            body follows the control motion.
        frame_indices : sequence[int], optional
            Subset/reorder of the control clip's (cached) pose frames to use,
            e.g. from video_edit.pick_indices — decimation without
            re-estimating poses.

        Returns
        -------
        str
            The ``output`` path (the rendered control video).
        """
        img = user_image
        if isinstance(user_image, str):
            img = cv2.imread(user_image)
            if img is None:
                raise FileNotFoundError(f"Cannot read user image: {user_image}")
        H, W = img.shape[:2]

        tgt_kpts, tgt_conf = self.estimator.estimate(img)
        if tgt_kpts is None:
            raise RuntimeError("No person detected in the user image.")
        self._log(f"user skeleton: {int(tgt_conf.sum())}/18 keypoints, "
                  f"canvas {W}x{H}")

        poses = self.control_poses
        if frame_indices is not None:
            poses = [poses[int(i)] for i in frame_indices]
        frames = pr.retarget_poses(
            tgt_kpts, tgt_conf, poses,
            root_motion=root_motion, smoothing=smoothing,
            pose_mode=pose_mode, blend_frames=blend_frames,
            foreshorten=foreshorten, mirror=mirror, head_lock=head_lock)

        out_dir = os.path.dirname(output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pr.render_pose_video(frames, output, W, H, fps or self.fps)
        self._log(f"wrote {len(frames)} frames -> {output}")
        return output


def generate_control_video(user_image, control_video="shot3.mp4",
                           output="control.mp4", *, device="cuda", **kwargs):
    """One-shot convenience wrapper around :class:`ControlVideoPipeline`.

    Builds a pipeline (loads model + estimates control clip, using the disk
    cache if warm) and generates a single control video. For batch use,
    construct a :class:`ControlVideoPipeline` once and reuse it instead.

    ``kwargs`` are split between construction (frame selection, model/cache
    dirs, kpt_thr, mode, verbose) and generation (pose_mode, root_motion,
    smoothing, blend_frames, foreshorten, fps).
    """
    init_keys = {"model_dir", "cache_dir", "skip_frames", "frame_load_cap",
                 "select_every_nth", "kpt_thr", "mode", "verbose"}
    init_kwargs = {k: v for k, v in kwargs.items() if k in init_keys}
    gen_kwargs = {k: v for k, v in kwargs.items() if k not in init_keys}
    pipe = ControlVideoPipeline(control_video, device=device, **init_kwargs)
    return pipe.generate(user_image, output, **gen_kwargs)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Generate a VTON control video from a user image + control clip.")
    ap.add_argument("user_image", help="User / VTON result image")
    ap.add_argument("--control_video", default="shot3.mp4")
    ap.add_argument("--output", default="control.mp4")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--pose_mode", default="relative",
                    choices=["relative", "absolute"])
    ap.add_argument("--root_motion", type=float, default=1.0)
    ap.add_argument("--smoothing", type=float, default=0.4)
    ap.add_argument("--blend_frames", type=int, default=0)
    ap.add_argument("--foreshorten", type=float, default=0.0)
    a = ap.parse_args()
    out = generate_control_video(
        a.user_image, a.control_video, a.output, device=a.device,
        pose_mode=a.pose_mode, root_motion=a.root_motion,
        smoothing=a.smoothing, blend_frames=a.blend_frames,
        foreshorten=a.foreshorten)
    print(out)
