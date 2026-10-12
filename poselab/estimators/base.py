"""What every pose model in this project looks like from the outside.

    model = create("vitpose")          # loads the weights (slow, once)
    model.start(width, height, fps)    # a new video begins
    pose = model.estimate(frame, t_ms) # one frame -> the participant's points
    model.close()

estimate() returns the participant only. Models that see several people apply the shared
rule in subject.py to choose; the number of people they saw is kept in Pose.n_people.
"""
from dataclasses import dataclass, field

import numpy as np

from ..subject import SubjectTracker, box_of


@dataclass
class Pose:
    kp: np.ndarray | None             # [K, 3]: x and y in pixels of the full frame, and the model's score. None: participant not found
    box: np.ndarray | None = None     # [4]: x1, y1, x2, y2 around the participant
    n_people: int = 0                 # how many people the model reported in this frame
    world: np.ndarray | None = None   # [K, 3] metres, only from models that estimate 3D
    parts_ms: dict = field(default_factory=dict)   # time spent in named parts of the model, when it has parts


class Estimator:
    key = ""            # short name used in file and folder names
    label = ""          # name shown to people
    names = []          # the model's own points, in its own order (see skeleton.py)
    min_score = 0.5     # score below which a point counts as not confident; each tool's customary value
    runs_on = "CPU"
    kind = ""           # how the model works, one line
    licence = ""
    has_3d = False      # True when estimate() also fills Pose.world

    def start(self, width, height, fps):
        """Called before the first frame of each video."""
        self.subject = SubjectTracker(width, height, fps)

    def estimate(self, frame_bgr, t_ms) -> Pose:
        raise NotImplementedError

    def close(self):
        pass

    def choose(self, people, boxes=None):
        """Apply the shared participant rule to people [N, K, 3]; boxes default to the extent of each person's points."""
        if boxes is None:
            boxes = [box_of(p[:, :2], p[:, 2], self.min_score) for p in people]
        valid = [i for i, b in enumerate(boxes) if b is not None]
        pick = self.subject.pick([boxes[i] for i in valid])
        if pick is None:
            return Pose(None, n_people=len(people))
        i = valid[pick]
        return Pose(np.asarray(people[i], np.float32), np.asarray(boxes[i], np.float32), n_people=len(people))
