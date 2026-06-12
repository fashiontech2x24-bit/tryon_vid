#!/usr/bin/env python3
"""
pose_retarget.py — Dynamic 2D pose motion retargeting for VTON video pipelines.

Extracts MOTION (per-bone rotation deltas + scaled root translation) from a
control video and applies it to the skeleton of the person in a VTON result
image — preserving the user's bone lengths, proportions, position and framing.

Output is an OpenPose JSON (array of frames) in the comfyui_controlnet_aux
POSE_KEYPOINT convention:
    [{"people":[{"pose_keypoints_2d":[x,y,c, ...x18]}],
      "canvas_width":W, "canvas_height":H}, ...]
  - coordinates are in PIXEL space of the VTON image canvas
  - c is 1.0 for visible keypoints, 0.0 for invisible
    (RenderPeopleKps / decode_json_as_poses drops any keypoint with c < 1.0)

Compatible with:  Load Openpose JSON  ->  Render Pose JSON (Human)

Usage:
    python pose_retarget.py \
        --control_video motion.mp4 \
        --vton_image vton_result.png \
        --output pose_frames.json \
        --skip_frames 0 --frame_load_cap 81 --select_every_nth 1 \
        --root_motion 1.0 --smoothing 0.4 \
        --preview preview.mp4

Dependencies:  pip install rtmlib onnxruntime opencv-python numpy
(rtmlib auto-downloads its detection + pose ONNX models on first run)
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

# --------------------------------------------------------------------------
# OpenPose BODY_18 topology
# --------------------------------------------------------------------------
# 0 Nose, 1 Neck, 2 RShoulder, 3 RElbow, 4 RWrist, 5 LShoulder, 6 LElbow,
# 7 LWrist, 8 RHip, 9 RKnee, 10 RAnkle, 11 LHip, 12 LKnee, 13 LAnkle,
# 14 REye, 15 LEye, 16 REar, 17 LEar

ROOT = 1  # neck
HEAD = (0, 14, 15, 16, 17)  # nose, eyes, ears — the joints head_lock pins

# child -> parent
PARENTS = {
    0: 1, 14: 0, 15: 0, 16: 14, 17: 15,
    2: 1, 3: 2, 4: 3,
    5: 1, 6: 5, 7: 6,
    8: 1, 9: 8, 10: 9,
    11: 1, 12: 11, 13: 12,
}

# traversal order: every parent appears before its children
ORDER = [0, 14, 16, 15, 17, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# left/right counterpart, used to repair bones missing in the VTON image
MIRROR = {
    2: 5, 5: 2, 3: 6, 6: 3, 4: 7, 7: 4,
    8: 11, 11: 8, 9: 12, 12: 9, 10: 13, 13: 10,
    14: 15, 15: 14, 16: 17, 17: 16,
}

# COCO-17 index -> OpenPose-18 index (neck is synthesized from shoulders)
COCO_TO_OP18 = {
    0: 0,            # nose
    1: 15, 2: 14,    # eyes  (coco: L,R)
    3: 17, 4: 16,    # ears
    5: 5, 6: 2,      # shoulders
    7: 6, 8: 3,      # elbows
    9: 7, 10: 4,     # wrists
    11: 11, 12: 8,   # hips
    13: 12, 14: 9,   # knees
    15: 13, 16: 10,  # ankles
}

LIMB_SEQ = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]
LIMB_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]


def wrap_angle(a):
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------
# Pose estimation (rtmlib runtime + DWPose ONNX models, pure onnxruntime)
# --------------------------------------------------------------------------
# Same model files comfyui_controlnet_aux's DWPose node uses (HF-hosted);
# rtmlib's YOLOX/RTMPose classes natively support their raw/simcc formats.
DWPOSE_MODELS = {
    "det": ("yolox_l.onnx",
            "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"),
    "pose": ("dw-ll_ucoco_384.onnx",
             "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"),
}


def ensure_dwpose_models(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    paths = {}
    for key, (fname, url) in DWPOSE_MODELS.items():
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            import urllib.request
            print(f"      downloading {fname} ...")
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        paths[key] = path
    return paths


class PoseEstimator:
    def __init__(self, mode="dwpose", device="cpu", kpt_thr=0.3,
                 model_dir="models"):
        try:
            from rtmlib import Body
        except ImportError:
            sys.exit(
                "rtmlib is not installed. Run:\n"
                "    pip install rtmlib\n"
                "(it uses your existing onnxruntime)"
            )
        backend = "onnxruntime"
        if mode == "dwpose":
            paths = ensure_dwpose_models(model_dir)
            self.body = Body(det=paths["det"], det_input_size=(640, 640),
                             pose=paths["pose"], pose_input_size=(288, 384),
                             to_openpose=False, backend=backend, device=device)
        else:
            # rtmlib presets; downloads from download.openmmlab.com
            self.body = Body(mode=mode, to_openpose=False,
                             backend=backend, device=device)
        self.kpt_thr = kpt_thr

    def estimate(self, img_bgr):
        """Return (kpts (18,2) float32, conf (18,) {0|1}) or (None, None)."""
        keypoints, scores = self.body(img_bgr)
        if keypoints is None or len(keypoints) == 0:
            return None, None
        # pick the most confident, largest person
        best, best_metric = 0, -1.0
        for i in range(len(keypoints)):
            k, s = keypoints[i], scores[i]
            valid = s > self.kpt_thr
            if not valid.any():
                continue
            span = np.ptp(k[valid], axis=0)  # (w, h) of visible keypoints
            metric = float(s.mean()) * float(max(span[0], span[1], 1.0))
            if metric > best_metric:
                best_metric, best = metric, i
        # wholebody models (DWPose, 133 kpts): first 17 are COCO body
        return self._coco17_to_op18(keypoints[best][:17], scores[best][:17])

    def _coco17_to_op18(self, k17, s17):
        kp = np.zeros((18, 2), dtype=np.float32)
        cf = np.zeros(18, dtype=np.float32)
        for coco_i, op_i in COCO_TO_OP18.items():
            if s17[coco_i] > self.kpt_thr:
                kp[op_i] = k17[coco_i]
                cf[op_i] = 1.0
        # synthesize neck = midpoint of shoulders
        if s17[5] > self.kpt_thr and s17[6] > self.kpt_thr:
            kp[1] = (k17[5] + k17[6]) / 2.0
            cf[1] = 1.0
        return kp, cf


# --------------------------------------------------------------------------
# Motion retargeting
# --------------------------------------------------------------------------
def body_scale(kpts, conf):
    """Characteristic size: mean neck->hip distance, fallback shoulder width."""
    if conf[ROOT] < 1.0:
        return None
    torso = [np.linalg.norm(kpts[h] - kpts[ROOT]) for h in (8, 11) if conf[h] >= 1.0]
    if torso:
        return float(np.mean(torso))
    if conf[2] >= 1.0 and conf[5] >= 1.0:
        return float(np.linalg.norm(kpts[2] - kpts[5]))
    return None


def mirror_control_poses(poses):
    """Mirror a control pose stream left<->right: swap L/R joints and reflect
    x about the sequence's mean root x. Synthesizes the opposite-direction
    motion (e.g. a counter-rotation) from a single recorded clip — mirroring
    must happen at the skeleton level, since flipping generated pixels would
    flip the garment's asymmetries, prints and face."""
    xs = [k[ROOT][0] for k, c in poses if c[ROOT] >= 1.0]
    if not xs:
        xs = [float(k[c >= 1.0][:, 0].mean()) for k, c in poses if (c >= 1.0).any()]
    cx = float(np.mean(xs)) if xs else 0.0
    out = []
    for kpts, conf in poses:
        mk, mc = kpts.copy(), conf.copy()
        for a, b in MIRROR.items():
            if a < b:
                mk[[a, b]] = kpts[[b, a]]
                mc[[a, b]] = conf[[b, a]]
        mk[:, 0] = 2.0 * cx - mk[:, 0]
        out.append((mk, mc))
    return out


class Retargeter:
    """Transfers per-bone rotation deltas from a source pose stream onto a
    fixed target skeleton, preserving the target's bone lengths and root."""

    def __init__(self, target_kpts, target_conf, root_motion=1.0, smoothing=0.0,
                 pose_mode="relative", blend_frames=0, foreshorten=0.0,
                 head_lock=0.0):
        if target_conf[ROOT] < 1.0:
            raise ValueError("Neck not detected on the VTON image — cannot build target skeleton.")
        self.tgt_root = target_kpts[ROOT].copy()
        self.root_motion = float(root_motion)
        # smoothing: EMA factor in [0,1); 0 disables. alpha = weight of history.
        self.alpha = float(np.clip(smoothing, 0.0, 0.95))
        # relative: target keeps its rest pose, control adds motion deltas.
        # absolute: target's joints follow the control's actual angles
        #           (with the target's bone lengths / root / framing).
        self.pose_mode = pose_mode
        # in absolute mode, ramp from the target's rest pose to the control
        # pose over the first N frames (0 = snap to control pose immediately)
        self.blend_frames = int(blend_frames)
        self.frame_idx = 0
        # foreshortening transfer: 0 = rigid bone lengths (rotation toward/away
        # from camera is invisible). 1 = bone length tracks the control bone's
        # per-frame length ratio, so torso/shoulder rotation reads as the limb
        # getting shorter in 2D — exactly as in the source video.
        self.foreshorten = float(np.clip(foreshorten, 0.0, 1.0))

        # rest pose of the target: per-bone (length, angle, valid)
        self.bone_len = np.zeros(18)
        self.bone_rest = np.zeros(18)
        self.bone_valid = np.zeros(18, dtype=bool)
        for c in ORDER:
            p = PARENTS[c]
            if target_conf[c] >= 1.0 and target_conf[p] >= 1.0:
                v = target_kpts[c] - target_kpts[p]
                self.bone_len[c] = np.linalg.norm(v)
                self.bone_rest[c] = math.atan2(v[1], v[0])
                self.bone_valid[c] = True
        # repair bones missing on the VTON image from their mirror counterpart
        for c in ORDER:
            if not self.bone_valid[c] and c in MIRROR and self.bone_valid[MIRROR[c]]:
                m = MIRROR[c]
                self.bone_len[c] = self.bone_len[m]
                # reflect direction about the vertical axis: (dx,dy)->(-dx,dy)
                self.bone_rest[c] = math.atan2(math.sin(self.bone_rest[m]),
                                               -math.cos(self.bone_rest[m]))
                self.bone_valid[c] = True

        self.tgt_scale = body_scale(target_kpts, target_conf)

        # head_lock: 0 = head follows the control motion; 1 = head keeps the
        # reference image's orientation and only translates with the neck
        # (so the torso can rotate while the face stays toward the camera).
        self.head_lock = float(np.clip(head_lock, 0.0, 1.0))
        # rest joint positions reconstructed from the (mirror-repaired) bone
        # table — the head_lock anchor, consistent even for repaired joints.
        self.rest_pos = np.zeros((18, 2), dtype=np.float32)
        self.rest_pos[ROOT] = self.tgt_root
        for c in ORDER:
            p = PARENTS[c]
            if self.bone_valid[c]:
                a = self.bone_rest[c]
                self.rest_pos[c] = self.rest_pos[p] + self.bone_len[c] * np.array(
                    [math.cos(a), math.sin(a)])
            else:
                self.rest_pos[c] = self.rest_pos[p]

        # per-bone reference angles. relative mode: lazily set to the control
        # stream's first sighting (delta = motion since then). absolute mode:
        # set to the target's rest angle, so rest + (src - rest) = src — the
        # output follows the control's absolute angles.
        if self.pose_mode == "absolute":
            self.ref_angle = [self.bone_rest[c] if self.bone_valid[c] else None
                              for c in range(18)]
        else:
            self.ref_angle = [None] * 18
        self.ref_root = None
        self.src_scale = None
        # temporal state
        self.last_delta = np.zeros(18)      # continuous (unwrapped) raw deltas
        self.smooth_delta = np.zeros(18)
        self.smooth_trans = np.zeros(2)
        # per-bone control reference length (first sighting) + smoothed ratio
        self.ref_len = [None] * 18
        self.smooth_len_ratio = np.ones(18)
        self._first = True

    def step(self, src_kpts, src_conf):
        """Retarget one control frame. Returns (kpts (18,2), conf (18,))."""
        out = np.zeros((18, 2), dtype=np.float32)
        out_conf = np.zeros(18, dtype=np.float32)

        # --- root translation -------------------------------------------
        if src_conf[ROOT] >= 1.0:
            if self.ref_root is None:
                self.ref_root = src_kpts[ROOT].copy()
            if self.src_scale is None:
                self.src_scale = body_scale(src_kpts, src_conf)
            trans = src_kpts[ROOT] - self.ref_root
        else:
            trans = self.smooth_trans  # hold last motion if neck dropped out

        ratio = 1.0
        if self.tgt_scale and self.src_scale:
            ratio = self.tgt_scale / self.src_scale
        trans = trans * ratio * self.root_motion

        if self._first or self.alpha == 0.0:
            self.smooth_trans = np.asarray(trans, dtype=np.float64)
        else:
            self.smooth_trans = self.alpha * self.smooth_trans + (1 - self.alpha) * trans

        out[ROOT] = self.tgt_root + self.smooth_trans
        out_conf[ROOT] = 1.0

        # --- per-bone rotation deltas ------------------------------------
        for c in ORDER:
            p = PARENTS[c]
            if not self.bone_valid[c]:
                out[c] = out[p]          # park on parent, marked invisible
                out_conf[c] = 0.0
                continue

            src_ok = src_conf[c] >= 1.0 and src_conf[p] >= 1.0
            if src_ok:
                v = src_kpts[c] - src_kpts[p]
                ang = math.atan2(v[1], v[0])
                if self.ref_angle[c] is None:
                    self.ref_angle[c] = ang   # first sighting defines the reference
                raw = wrap_angle(ang - self.ref_angle[c])
                # unwrap against the running value so EMA never crosses +-pi
                delta = self.last_delta[c] + wrap_angle(raw - self.last_delta[c])
                self.last_delta[c] = delta
                # per-bone foreshortening: how short is this control bone now
                # vs its reference, normalized out of any global zoom by the
                # control body scale ratio.
                slen = math.hypot(v[0], v[1])
                if self.ref_len[c] is None and slen > 1e-3:
                    self.ref_len[c] = slen
                if self.ref_len[c]:
                    len_ratio = float(np.clip(slen / self.ref_len[c], 0.15, 1.3))
                else:
                    len_ratio = 1.0
            else:
                delta = self.last_delta[c]            # bone occluded: hold last rotation
                len_ratio = self.smooth_len_ratio[c]  # ...and last foreshortening

            if self._first or self.alpha == 0.0:
                self.smooth_delta[c] = delta
                self.smooth_len_ratio[c] = len_ratio
            else:
                self.smooth_delta[c] = (self.alpha * self.smooth_delta[c]
                                        + (1 - self.alpha) * delta)
                self.smooth_len_ratio[c] = (self.alpha * self.smooth_len_ratio[c]
                                            + (1 - self.alpha) * len_ratio)

            w = 1.0
            if self.pose_mode == "absolute" and self.blend_frames > 0:
                w = min(1.0, self.frame_idx / float(self.blend_frames))

            a = self.bone_rest[c] + w * self.smooth_delta[c]
            # blend foreshortening in by weight too, so it ramps with the pose
            fs = 1.0 + w * self.foreshorten * (self.smooth_len_ratio[c] - 1.0)
            out[c] = out[p] + self.bone_len[c] * fs * np.array([math.cos(a), math.sin(a)])
            out_conf[c] = 1.0

        # head lock: pull head joints toward their rest pose translated
        # rigidly with the neck, instead of following the control rotation.
        if self.head_lock > 0.0:
            shift = out[ROOT] - self.tgt_root
            for c in HEAD:
                if out_conf[c] >= 1.0:
                    locked = self.rest_pos[c] + shift
                    out[c] = (1.0 - self.head_lock) * out[c] + self.head_lock * locked

        self._first = False
        self.frame_idx += 1
        return out, out_conf


# --------------------------------------------------------------------------
# Video frame loading (VHS-style skip / cap / every-nth)
# --------------------------------------------------------------------------
def iter_video_frames(path, skip_frames=0, frame_load_cap=0, select_every_nth=1):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Cannot open control video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    idx, loaded = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= skip_frames and (idx - skip_frames) % select_every_nth == 0:
                yield fps, frame
                loaded += 1
                if frame_load_cap and loaded >= frame_load_cap:
                    break
            idx += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def frame_to_openpose_dict(kpts, conf, width, height):
    flat = []
    for i in range(18):
        if conf[i] >= 1.0:
            flat += [float(kpts[i][0]), float(kpts[i][1]), 1.0]
        else:
            flat += [0.0, 0.0, 0.0]
    return {
        "people": [{
            "pose_keypoints_2d": flat,
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
        }],
        "canvas_width": int(width),
        "canvas_height": int(height),
    }


def draw_pose(canvas, kpts, conf):
    """OpenPose body render, faithful to comfyui_controlnet_aux's
    draw_bodypose (ellipse limbs at 0.6 intensity, radius-4 joints) —
    the same style VACE sees from the DWPose preprocessor."""
    stickwidth = 4
    for (a, b), color in zip(LIMB_SEQ, LIMB_COLORS):
        if conf[a] < 1.0 or conf[b] < 1.0:
            continue
        x1, y1 = kpts[a]
        x2, y2 = kpts[b]
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        poly = cv2.ellipse2Poly((int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in color[::-1]])
    for i in range(18):
        if conf[i] >= 1.0:
            bgr = [int(c) for c in LIMB_COLORS[i][::-1]]
            cv2.circle(canvas, (int(kpts[i][0]), int(kpts[i][1])), 4, bgr, thickness=-1)
    return canvas


# --------------------------------------------------------------------------
# Reusable orchestration (shared by the CLI and pose_pipeline.ControlVideoPipeline)
# --------------------------------------------------------------------------
def estimate_video_poses(estimator, control_video, skip_frames=0, frame_load_cap=0,
                         select_every_nth=1, progress=True):
    """Run pose estimation over a control video.

    Returns (poses, fps, n_missing) where ``poses`` is a list of
    (kpts (18,2) float32, conf (18,) float32) — one entry per loaded frame,
    zero-filled for frames where no person was detected.
    """
    poses, fps, n_miss = [], 16.0, 0
    for fps, frame in iter_video_frames(control_video, skip_frames,
                                        frame_load_cap, select_every_nth):
        kpts, conf = estimator.estimate(frame)
        if kpts is None:
            n_miss += 1
            kpts = np.zeros((18, 2), dtype=np.float32)
            conf = np.zeros(18, dtype=np.float32)
        poses.append((kpts, conf))
        if progress and len(poses) % 25 == 0:
            print(f"      {len(poses)} frames ...")
    return poses, fps, n_miss


def retarget_poses(target_kpts, target_conf, control_poses, *,
                   root_motion=1.0, smoothing=0.4, pose_mode="relative",
                   blend_frames=0, foreshorten=0.0, mirror=False,
                   head_lock=0.0):
    """Apply a stream of control poses onto a fixed target skeleton.

    ``mirror`` flips the control motion left<->right (skeleton-level) and
    ``head_lock`` (0..1) pins the head to the target's rest orientation.

    Returns a list of (kpts (18,2), conf (18,)) — one retargeted frame per
    control pose, all on the target's bone lengths / framing.
    """
    if mirror:
        control_poses = mirror_control_poses(control_poses)
    rt = Retargeter(target_kpts, target_conf, root_motion=root_motion,
                    smoothing=smoothing, pose_mode=pose_mode,
                    blend_frames=blend_frames, foreshorten=foreshorten,
                    head_lock=head_lock)
    return [rt.step(src_kpts, src_conf) for src_kpts, src_conf in control_poses]


def render_pose_video(frames, path, width, height, fps):
    """Encode retargeted pose frames as an mp4 skeleton video."""
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                         max(1.0, float(fps)), (int(width), int(height)))
    for kpts, conf in frames:
        vw.write(draw_pose(np.zeros((int(height), int(width), 3), dtype=np.uint8),
                           kpts, conf))
    vw.release()
    return path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control_video", required=True, help="RGB video of the motion to extract")
    ap.add_argument("--vton_image", required=True, help="VTON result image (defines target skeleton + canvas)")
    ap.add_argument("--output", required=True, help="Output OpenPose JSON path (array of frames)")
    ap.add_argument("--skip_frames", type=int, default=0)
    ap.add_argument("--frame_load_cap", type=int, default=0, help="0 = all frames")
    ap.add_argument("--select_every_nth", type=int, default=1)
    ap.add_argument("--root_motion", type=float, default=1.0,
                    help="Scale of transferred root translation (0 = pin person in place)")
    ap.add_argument("--pose_mode", default="relative", choices=["relative", "absolute"],
                    help="relative: keep the user's rest pose, add control motion on top. "
                         "absolute: follow the control video's actual joint angles "
                         "(user's bone lengths/framing still preserved)")
    ap.add_argument("--blend_frames", type=int, default=0,
                    help="absolute mode: ramp from the user's rest pose to the control pose "
                         "over the first N frames (0 = immediate)")
    ap.add_argument("--foreshorten", type=float, default=0.0,
                    help="0..1: transfer per-bone foreshortening from the control. "
                         "Makes torso/shoulder rotation toward/away from camera visible "
                         "(the limb shortens in 2D as in the source). Try 1.0; 0 = rigid lengths")
    ap.add_argument("--smoothing", type=float, default=0.4,
                    help="Temporal EMA on bone rotations, 0..0.95 (0 = off)")
    ap.add_argument("--per_frame_dir", default=None,
                    help="Also write one JSON per frame (frame_0000.json, ...) into this dir")
    ap.add_argument("--preview", default=None, help="Optional mp4 path to render a skeleton preview")
    ap.add_argument("--render_dir", default=None,
                    help="Render pose frames as PNGs (pose_0000.png, ...) into this dir — "
                         "load with VHS 'Load Images (Path)' and feed straight to VACE, "
                         "bypassing the Load/Render JSON nodes (which only handle 1 frame)")
    ap.add_argument("--mode", default="dwpose",
                    choices=["dwpose", "lightweight", "balanced", "performance"],
                    help="dwpose = HF-hosted DWPose models (default); others = rtmlib presets from openmmlab")
    ap.add_argument("--model_dir", default="models", help="Where to cache the DWPose ONNX models")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--kpt_thr", type=float, default=0.3, help="Detector confidence threshold")
    args = ap.parse_args()

    vton = cv2.imread(args.vton_image)
    if vton is None:
        sys.exit(f"Cannot read VTON image: {args.vton_image}")
    H, W = vton.shape[:2]

    est = PoseEstimator(mode=args.mode, device=args.device, kpt_thr=args.kpt_thr,
                        model_dir=args.model_dir)

    print(f"[1/3] Estimating target skeleton from {args.vton_image} ({W}x{H}) ...")
    tgt_kpts, tgt_conf = est.estimate(vton)
    if tgt_kpts is None:
        sys.exit("No person detected in the VTON image.")
    print(f"      target keypoints detected: {int(tgt_conf.sum())}/18")

    print(f"[2/3] Processing control video {args.control_video} "
          f"(skip={args.skip_frames}, cap={args.frame_load_cap or 'all'}, "
          f"every_nth={args.select_every_nth}) ...")
    control_poses, fps, n_miss = estimate_video_poses(
        est, args.control_video, args.skip_frames,
        args.frame_load_cap, args.select_every_nth)
    n_in = len(control_poses)

    if not control_poses:
        sys.exit("No frames were loaded from the control video.")
    if n_miss:
        print(f"      warning: no person detected in {n_miss}/{n_in} frames (pose held)")

    preview_frames = retarget_poses(
        tgt_kpts, tgt_conf, control_poses,
        root_motion=args.root_motion, smoothing=args.smoothing,
        pose_mode=args.pose_mode, blend_frames=args.blend_frames,
        foreshorten=args.foreshorten)
    frames_json = [frame_to_openpose_dict(k, c, W, H) for k, c in preview_frames]

    print(f"[3/3] Writing {len(frames_json)} frames -> {args.output}")
    with open(args.output, "w") as f:
        json.dump(frames_json, f)

    if args.per_frame_dir:
        os.makedirs(args.per_frame_dir, exist_ok=True)
        for i, fr in enumerate(frames_json):
            with open(os.path.join(args.per_frame_dir, f"frame_{i:04d}.json"), "w") as f:
                json.dump(fr, f)
        print(f"      per-frame JSONs -> {args.per_frame_dir}/")

    if args.render_dir:
        os.makedirs(args.render_dir, exist_ok=True)
        # remove leftovers from a previous (longer) run so VHS doesn't load them
        import glob
        for old in glob.glob(os.path.join(args.render_dir, "pose_*.png")):
            os.remove(old)
        for i, (kpts, conf) in enumerate(preview_frames):
            img = draw_pose(np.zeros((H, W, 3), dtype=np.uint8), kpts, conf)
            cv2.imwrite(os.path.join(args.render_dir, f"pose_{i:04d}.png"), img)
        print(f"      rendered pose PNGs -> {args.render_dir}/")

    if args.preview:
        eff_fps = max(1.0, fps / max(1, args.select_every_nth))
        vw = cv2.VideoWriter(args.preview, cv2.VideoWriter_fourcc(*"mp4v"), eff_fps, (W, H))
        for kpts, conf in preview_frames:
            vw.write(draw_pose(np.zeros((H, W, 3), dtype=np.uint8), kpts, conf))
        vw.release()
        print(f"      preview video -> {args.preview}")

    if args.render_dir:
        print("Done. Load the PNGs with VHS 'Load Images (Path)' -> VACE control_video.")
    else:
        print("Done. Note: 'Render Pose JSON (Human)' renders only the FIRST frame of a "
              "multi-frame JSON — use --render_dir to render all frames as PNGs instead.")


if __name__ == "__main__":
    main()
