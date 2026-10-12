"""Reads the Stage 1 pose files and hands them to the shared analysis in poselab.

The Stage 1 files (written by extract_pose.py) keep MediaPipe's own layout: lm
[T, P, 33, 5] with x and y as fractions of width and height. This module reads one,
picks the participant when two poses per frame were stored, converts to the joints
every model shares, and calls poselab.analysis, which holds the geometry, the frame
quality checks and the repetition logic for all models.

The Stage 1 scripts use it as "pk"; the names imported below are part of that.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poselab import analysis
from poselab.analysis import FLAGS, FS, arm_elevation, frame_flags, gaps, robust_range  # noqa: F401  (used as pk.<name>)
from poselab.skeleton import MEDIAPIPE33, to_canonical

VISIBILITY_MIN = 0.5
MP_L_HIP, MP_R_HIP = 23, 24      # MediaPipe's own numbering, used only before conversion


def select_subject(lm, width, height):
    """Pick one pose per frame when several were requested.

    The exerciser stays in place while a passer-by moves, so the subject is the pose whose
    hip centre is nearest the median hip centre of all detections in the clip.
    """
    n_frames, n_poses = lm.shape[:2]
    if n_poses == 1:
        return np.zeros(n_frames, int)
    hips = lm[:, :, [MP_L_HIP, MP_R_HIP], :2].mean(axis=2) * [width, height]      # [T, P, 2]
    anchor = np.nanmedian(hips.reshape(-1, 2), axis=0)
    dist = np.linalg.norm(hips - anchor, axis=-1)                                 # [T, P]
    return np.where(np.isfinite(dist), dist, np.inf).argmin(axis=1)


def load_clip(path, front_view):
    """The participant's joints as the clip dict described in poselab/analysis.py."""
    z = np.load(path)
    width, height, fps = int(z["width"]), int(z["height"]), float(z["fps"])
    lm_all, world_all = z["lm"], z["world"]
    pick = select_subject(lm_all, width, height)
    rows = np.arange(len(pick))
    lm, world = lm_all[rows, pick], world_all[rows, pick]
    t = z["t_ms"] / 1000.0
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        t = rows / fps
    clip = {"t": t, "px": to_canonical(lm[..., :2] * [width, height], MEDIAPIPE33),
            "score": to_canonical(lm[..., 3:4], MEDIAPIPE33)[..., 0], "min_score": VISIBILITY_MIN,
            "world": to_canonical(world, MEDIAPIPE33).astype(np.float32),      # float32 as stored, so results match the first Stage 1 run exactly
            "has_pose": np.isfinite(lm[:, 0, 0]), "swapped": np.zeros(len(rows), bool),
            "width": width, "height": height, "fps": fps, "n_poses": np.isfinite(lm_all[:, :, 0, 0]).sum(axis=1)}
    return analysis.relabel_front_view(clip) if front_view else clip


def analyse_clip(path, front_view):
    """Frame flags, clip quality summary, movement signal, repetitions and the 15 minimum features."""
    return analysis.analyse(load_clip(path, front_view))
