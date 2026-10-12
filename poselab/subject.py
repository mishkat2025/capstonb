"""Decide which detected person is the participant.

Some clips have a second person walking through the picture. Every model is given the same
rule, so that differences between models come from the pose model and not from this choice.
The rule rests on one fact about these recordings: the participant exercises in one place.

1. First frame with anybody in it: take the largest person, favouring the middle of the
   picture (the camera is aimed at the participant).
2. Afterwards: compare every person's box with the participant's usual box, which is the
   median of the boxes accepted over the last five seconds. Take the person who overlaps
   it most; if nobody overlaps it enough, the participant counts as not found in that
   frame. A median does not move when a passer-by is accepted for a few frames, so a
   passer-by cannot drag the usual box away with them.
3. If the participant is not found for two seconds, start again from rule 1.

The rule cannot tell a passer-by who stands exactly in front of the participant from the
participant. Those frames are caught later by the quality checks in analysis.py (body
size and position jumps), as in Stage 1.

History, 2026-10-12, tried on one occlusion clip (E01_P02_AF_VO_GM): a first version
compared with a box that followed the chosen person from frame to frame. It followed the
passer-by out of the picture and lost one to three seconds after each crossing.
"""
from collections import deque

import numpy as np

MIN_OVERLAP = 0.3      # intersection over union with the usual box; low, because raising an arm sideways widens the box a lot
MEMORY_S = 5.0         # the usual box is the median over this much recent history
RESTART_AFTER_S = 2.0


def box_of(points, scores=None, min_score=0.0):
    """Bounding box x1, y1, x2, y2 of the valid points [K, 2]; None when there are fewer than two."""
    ok = np.isfinite(points).all(axis=1)
    if scores is not None:
        ok &= np.nan_to_num(scores, nan=0.0) >= min_score
    if ok.sum() < 2:
        return None
    return np.r_[points[ok].min(axis=0), points[ok].max(axis=0)]


def overlap(a, b):
    """Intersection over union of two boxes."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class SubjectTracker:
    def __init__(self, width, height, fps):
        self.width, self.height = width, height
        self.patience = max(1, round(RESTART_AFTER_S * fps))
        self.accepted = deque(maxlen=max(1, round(MEMORY_S * fps)))     # recent boxes of the participant
        self.missed = 0

    def pick(self, boxes):
        """Index of the participant among boxes [[x1, y1, x2, y2], ...], or None."""
        boxes = [np.asarray(b, float) for b in boxes]
        if self.accepted and self.missed >= self.patience:
            self.accepted.clear()
        if not boxes:
            self.missed += 1
            return None
        if not self.accepted:
            def prominence(b):
                area = (b[2] - b[0]) * (b[3] - b[1])
                off_centre = abs((b[0] + b[2]) / 2 - self.width / 2) / (self.width / 2)
                return area * max(0.0, 1.0 - off_centre)
            best = int(np.argmax([prominence(b) for b in boxes]))
        else:
            usual = np.median(self.accepted, axis=0)
            overlaps = [overlap(usual, b) for b in boxes]
            best = int(np.argmax(overlaps))
            if overlaps[best] < MIN_OVERLAP:
                self.missed += 1
                return None
        self.accepted.append(boxes[best])
        self.missed = 0
        return best
