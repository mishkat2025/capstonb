"""Hands: 21 points per hand, and whether the hand is open or closed.

Body pose models stop at the wrist. Fingers need a hand model, and a hand in these videos
is small, so the hand model is not run on the whole frame. For each hand, a square around
the wrist is cut out of the frame (the pose models say where the wrist and elbow are),
enlarged, and given to MediaPipe Hand Landmarker (Google, Apache-2.0, runs on the CPU).

The hand model is independent of the pose models. It is run once per frame and hand, on
the wrist position that the selected pose models agree on (their median), so its result
is the same whichever pose models are ticked.

Open or closed is a single number, the "extension": for each of the four fingers, the
straight-line distance from the wrist to the fingertip divided by the length of the path
wrist - knuckle - joints - fingertip, averaged over the fingers. A flat hand gives about
1; a fist gives a low value because the fingertips come back towards the palm. It is
computed on the hand model's 3D points, so it does not depend on which way the hand faces.

The two thresholds are geometric conventions, not validated against labelled hands:
bending every finger joint by about 25 degrees gives roughly 0.92, by 45 degrees roughly
0.72, and a fist roughly 0.4.

What a scan of 108 development clips showed on 2026-10-12 (docs/pose_models.md, Section 7):
- The hand is found in most frames only in expert front-view clips. It is lost when it
  moves fast enough to blur, in low light, in side views and in small or low-quality video.
- Low extension values do go with visibly curled fingers (a resting hand hangs in a loose
  curl), but blurred frames also give low values.
- Two things were tried to remove doubtful readings and dropped because they only lowered
  the share of frames with a hand and left the doubtful ones in: the model's default
  detection threshold of 0.5, and rejecting hands that look very small in the picture.
"""
import time
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from . import MODELS, weights
from .skeleton import J, L_HIP, L_SHOULDER, MIRROR, R_HIP, R_SHOULDER, to_canonical

WEIGHTS = MODELS / "mediapipe" / "hand_landmarker.task"
SIDES = ("left", "right")
OPEN_FROM, CLOSED_BELOW = 0.85, 0.60
STATES = ("open", "partly closed", "closed")

CUTOUT = 1.3        # side of the square cut-out, in torso lengths
SHIFT = 0.25        # the cut-out is centred this far beyond the wrist along the forearm, in torso lengths
INPUT = 320         # the cut-out is resized to this before the hand model
NEAR = 0.35         # the hand model's wrist must lie this close to the pose wrist (torso lengths); else it is another hand or a picture
DETECT_MIN = 0.3    # lower than the default 0.5: these hands are small
STEADY_FRAMES = 7   # open/closed is judged on the median over this many frames (about a quarter of a second)

WRIST = 0
FINGERS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]     # index, middle, ring, little: knuckle ... tip


@dataclass
class Hand:
    points: np.ndarray          # [21, 2] pixels of the full frame
    extension: float            # about 1 flat hand, low for a fist; this frame alone
    state: str                  # one of STATES, judged on the median of the last few frames
    cutout: np.ndarray          # the enlarged cut-out the hand model saw (BGR), for display
    cutout_points: np.ndarray   # [21, 2] the same points in cut-out pixels


def extension(world):
    """See the module text. world: [21, 3] points of one hand in metres."""
    values = []
    for knuckle, middle, far, tip in FINGERS:
        path = sum(np.linalg.norm(world[a] - world[b]) for a, b in ((WRIST, knuckle), (knuckle, middle), (middle, far), (far, tip)))
        values.append(np.linalg.norm(world[tip] - world[WRIST]) / path)
    return float(np.mean(values))


def state_of(value):
    return "open" if value >= OPEN_FROM else "closed" if value < CLOSED_BELOW else "partly closed"


def steady(values, window=STEADY_FRAMES):
    """Median of each value and its neighbours in time. A single blurred frame can give a wild value; a real change lasts longer.

    values: [T] with NaN where the hand was not found; those stay NaN.
    """
    values = np.asarray(values, float)
    out = np.full(len(values), np.nan)
    half = window // 2
    for i in np.flatnonzero(np.isfinite(values)):
        near = values[max(0, i - half):i + half + 1]
        out[i] = np.median(near[np.isfinite(near)])
    return out


def consensus_joints(poses, models, front_view):
    """Where the selected pose models put the participant's joints in this frame: their median, [J, 2]; None if nobody found them.

    In a front view a model's left/right labels are first put right, as analysis.relabel_front_view does.
    """
    found = []
    for key, pose in poses.items():
        if pose.kp is None:
            continue
        joints = to_canonical(pose.kp[:, :2], models[key].names)
        if front_view and joints[L_SHOULDER, 0] < joints[R_SHOULDER, 0]:
            joints = joints[MIRROR]
        found.append(joints)
    if not found:
        return None
    stack = np.array(found)                     # [models, J, 2]
    out = np.full(stack.shape[1:], np.nan)
    for j in range(stack.shape[1]):             # each joint: median over the models that have it
        have = stack[:, j][np.isfinite(stack[:, j]).all(axis=1)]
        if len(have):
            out[j] = np.median(have, axis=0)
    return out


class HandReader:
    def __init__(self, detect_min=DETECT_MIN, announce=print):
        weights.ensure("hands", announce)       # downloads the hand model the first time
        self.landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(WEIGHTS)), running_mode=vision.RunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=detect_min, min_hand_presence_confidence=detect_min))
        self.start()

    def start(self):
        """Called before the first frame of each video."""
        self.torso = None       # last known torso length in pixels
        self.recent = {side: deque(maxlen=STEADY_FRAMES // 2 + 1) for side in SIDES}       # latest extension values, for a steady label

    def read(self, frame_bgr, joints):
        """Hands of the participant: ({"left": Hand or None, "right": Hand or None}, milliseconds spent)."""
        started = time.perf_counter()
        hands = {side: None for side in SIDES}
        if joints is not None:
            torso = np.linalg.norm(joints[[L_SHOULDER, R_SHOULDER]].mean(axis=0) - joints[[L_HIP, R_HIP]].mean(axis=0))
            if np.isfinite(torso) and torso > 1:
                self.torso = float(torso)
            if self.torso is not None:
                for side in SIDES:
                    hands[side] = self._one(frame_bgr, joints[J[f"{side}_wrist"]], joints[J[f"{side}_elbow"]])
        for side, one in hands.items():
            self.recent[side].append(one.extension if one else np.nan)
            if one:
                seen = [v for v in self.recent[side] if np.isfinite(v)]
                one.state = state_of(float(np.median(seen)))
        return hands, 1000 * (time.perf_counter() - started)

    def _one(self, frame, wrist, elbow):
        if not np.isfinite(wrist).all():
            return None
        along = wrist - elbow if np.isfinite(elbow).all() else np.zeros(2)
        length = np.linalg.norm(along)
        centre = wrist + SHIFT * self.torso * (along / length if length > 1e-6 else along)
        size = max(8, int(round(CUTOUT * self.torso)))
        x0, y0 = int(round(centre[0] - size / 2)), int(round(centre[1] - size / 2))
        # copy the square, leaving black where it reaches beyond the picture
        height, width = frame.shape[:2]
        cut = np.zeros((size, size, 3), np.uint8)
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + size, width), min(y0 + size, height)
        if xa >= xb or ya >= yb:
            return None
        cut[ya - y0:yb - y0, xa - x0:xb - x0] = frame[ya:yb, xa:xb]
        big = cv2.resize(cut, (INPUT, INPUT), interpolation=cv2.INTER_CUBIC)
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(big, cv2.COLOR_BGR2RGB)))
        best = None
        for image_points, world_points in zip(result.hand_landmarks, result.hand_world_landmarks):
            in_cut = np.array([(q.x * INPUT, q.y * INPUT) for q in image_points])
            points = in_cut * (size / INPUT) + [x0, y0]
            off = np.linalg.norm(points[WRIST] - wrist)
            if off <= NEAR * self.torso and (best is None or off < best[0]):
                value = extension(np.array([(q.x, q.y, q.z) for q in world_points]))
                best = (off, Hand(points.astype(np.float32), value, state_of(value), big, in_cut.astype(np.float32)))
        return None if best is None else best[1]

    def close(self):
        self.landmarker.close()
