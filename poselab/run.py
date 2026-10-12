"""Run one or more pose models over a video, every frame, in a single pass.

Each frame is decoded once and handed to every model in turn. The models do not see each
other's results.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from .estimators import LABELS, create
from .video import Video


@dataclass
class Track:
    """Everything one model produced for one video (arrays are per frame)."""
    key: str
    label: str
    names: list
    min_score: float
    runs_on: str
    kind: str
    licence: str
    kp: np.ndarray = None           # [T, K, 3]
    box: np.ndarray = None          # [T, 4]
    n_people: np.ndarray = None     # [T]
    ms: np.ndarray = None           # [T] total time per frame; for top-down models this includes the person detector
    parts_ms: dict = field(default_factory=dict)    # e.g. {"detector": [T], "pose": [T]}
    world: np.ndarray = None        # [T, K, 3] or None


@dataclass
class HandTrack:
    """What the hand model found (see hands.py). Index 0 is the left hand, 1 the right."""
    points: np.ndarray              # [T, 2, 21, 2] pixels of the full frame; NaN where the hand was not found
    extension: np.ndarray           # [T, 2] about 1 for a flat hand, low for a fist; NaN where not found
    ms: np.ndarray                  # [T] time for both hands


@dataclass
class Run:
    video: str
    width: int
    height: int
    fps: float
    t_ms: np.ndarray
    tracks: dict                    # model key -> Track
    stopped_early: bool = False
    elapsed_s: float = 0.0
    hands: HandTrack = None         # None when the hand model was not used


def load_models(keys, announce=print):
    models = {}
    for key in keys:
        announce(f"Loading {LABELS[key]} ...")
        started = time.perf_counter()
        models[key] = create(key, announce)
        announce(f"Loaded {LABELS[key]} in {time.perf_counter() - started:.1f} s ({models[key].runs_on})")
    return models


def process(video_path, models, on_frame=None, max_frames=None, hand_reader=None, front_view=False):
    """Run the models over the video.

    on_frame(index, t_ms, frame, poses, frame_ms, hands) is called after each frame with
    the poses {key: Pose}, each model's time, and the hands {"left": Hand or None, ...}
    (None when no hand_reader was given); returning False stops the run.

    hand_reader: a hands.HandReader. It looks for each hand around the wrist position the
    pose models agree on; front_view tells it whether left/right labels need the front-view repair.
    """
    if hand_reader is not None:
        from .hands import SIDES, consensus_joints
    with Video(video_path) as probe:       # one frame through every model first: the first call of a GPU model is slow
        first = next(iter(probe), None)
        if first is None:
            raise OSError(f"no frames in {video_path}")
        for model in models.values():
            model.start(probe.width, probe.height, probe.fps)
            model.estimate(first[2], first[1])

    rows = {key: {"kp": [], "box": [], "n": [], "ms": [], "parts": {}, "world": []} for key in models}
    hand_rows = {"points": [], "extension": [], "ms": []}
    t_all, stopped = [], False
    started = time.perf_counter()
    with Video(video_path) as video:
        for index, t_ms, frame in video:
            if index == 0:
                for model in models.values():
                    model.start(video.width, video.height, video.fps)      # fresh tracking state after the warm-up frame
                if hand_reader is not None:
                    hand_reader.start()
            if max_frames is not None and index >= max_frames:
                stopped = True
                break
            poses, frame_ms = {}, {}
            for key, model in models.items():
                tick = time.perf_counter()
                pose = model.estimate(frame, t_ms)
                wall = 1000 * (time.perf_counter() - tick)
                # A top-down model reports its parts; the shared detector's time is counted for each model that uses it.
                frame_ms[key] = sum(pose.parts_ms.values()) if pose.parts_ms else wall
                poses[key] = pose
                r = rows[key]
                k = len(model.names)
                r["kp"].append(pose.kp if pose.kp is not None else np.full((k, 3), np.nan, np.float32))
                r["box"].append(pose.box if pose.box is not None else np.full(4, np.nan, np.float32))
                r["world"].append(pose.world if pose.world is not None else np.full((k, 3), np.nan, np.float32))
                r["n"].append(pose.n_people)
                r["ms"].append(frame_ms[key])
                for part, value in pose.parts_ms.items():
                    r["parts"].setdefault(part, []).append(value)
            hands = None
            if hand_reader is not None:
                hands, hand_ms = hand_reader.read(frame, consensus_joints(poses, models, front_view))
                hand_rows["points"].append([hands[s].points if hands[s] else np.full((21, 2), np.nan, np.float32) for s in SIDES])
                hand_rows["extension"].append([hands[s].extension if hands[s] else np.nan for s in SIDES])
                hand_rows["ms"].append(hand_ms)
            t_all.append(t_ms)
            if on_frame is not None and on_frame(index, t_ms, frame, poses, frame_ms, hands) is False:
                stopped = True
                break
        width, height, fps = video.width, video.height, video.fps
    hand_track = None
    if hand_reader is not None:
        hand_track = HandTrack(points=np.asarray(hand_rows["points"], np.float32).reshape(-1, 2, 21, 2),
                               extension=np.asarray(hand_rows["extension"], np.float32).reshape(-1, 2),
                               ms=np.asarray(hand_rows["ms"], np.float32))

    tracks = {}
    for key, model in models.items():
        r = rows[key]
        world = np.asarray(r["world"], np.float32)
        tracks[key] = Track(key=key, label=model.label, names=list(model.names), min_score=model.min_score, runs_on=model.runs_on,
                            kind=model.kind, licence=model.licence,
                            kp=np.asarray(r["kp"], np.float32), box=np.asarray(r["box"], np.float32),
                            n_people=np.asarray(r["n"], np.int16), ms=np.asarray(r["ms"], np.float32),
                            parts_ms={p: np.asarray(v, np.float32) for p, v in r["parts"].items()},
                            world=world if np.isfinite(world).any() else None)
    return Run(video=str(video_path), width=width, height=height, fps=fps, t_ms=np.asarray(t_all, np.float64), tracks=tracks,
               stopped_early=stopped, elapsed_s=time.perf_counter() - started, hands=hand_track)
