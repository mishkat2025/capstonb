"""Pose files: what one model produced for one video, as one .npz.

    t_ms      [T]         timestamp of each frame from the video file
    kp        [T, K, 3]   x, y in full-frame pixels and the model's score, in the model's own
                          point order; NaN on frames where the participant was not found
    names     [K]         the model's point names (see skeleton.py)
    box       [T, 4]      x1, y1, x2, y2 around the participant; NaN when not found
    n_people  [T]         people the model reported in the frame
    ms        [T]         time the model took on the frame, in milliseconds
    world     [T, K, 3]   metres; only for models that estimate 3D
    width, height, fps, model, label, min_score, runs_on

The points are stored as the model gave them. Conversion to the shared joints, and the
left/right repair for front views, happen when the file is read with to_clip().
"""
import numpy as np

from . import analysis
from .skeleton import to_canonical


def save(path, track, t_ms, width, height, fps):
    """track: a run.Track."""
    extra = {"world": track.world} if track.world is not None else {}
    np.savez_compressed(path, t_ms=np.asarray(t_ms, np.float64), kp=track.kp, names=np.array(track.names), box=track.box,
                        n_people=track.n_people, ms=track.ms, width=width, height=height, fps=fps,
                        model=track.key, label=track.label, min_score=track.min_score, runs_on=track.runs_on, **extra)


def to_clip(record, front_view):
    """Turn a pose file (or any mapping with the same fields) into the clip dict that analysis.py works on."""
    kp = np.asarray(record["kp"], float)
    names = [str(n) for n in record["names"]]
    t = np.asarray(record["t_ms"], float) / 1000.0
    fps = float(record["fps"])
    if len(t) > 1 and not np.all(np.diff(t) > 0):       # a few phone videos have unusable timestamps
        t = np.arange(len(t)) / fps
    world = record["world"] if "world" in record else None
    clip = {"t": t, "px": to_canonical(kp[..., :2], names), "score": to_canonical(kp[..., 2:], names)[..., 0],
            "min_score": float(record["min_score"]), "has_pose": np.isfinite(kp[..., 0]).any(axis=1),
            "swapped": np.zeros(len(t), bool), "width": int(record["width"]), "height": int(record["height"]), "fps": fps,
            "world": to_canonical(world, names) if world is not None else None}
    return analysis.relabel_front_view(clip) if front_view else clip


def load(path, front_view):
    with np.load(path) as record:
        return to_clip(record, front_view)
