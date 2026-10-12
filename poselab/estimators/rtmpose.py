"""RTMPose-m, 26-point version with feet (OpenMMLab, 2023), run through rtmlib and ONNX Runtime.

Top-down: the shared person detector finds the people, the participant's box is cut out
and resized to 256 x 192, and the model predicts each point as two 1D distributions (one
along x, one along y) instead of a 2D heatmap. That design is what makes it fast.
"""
import time

import numpy as np

from .. import MODELS
from ..skeleton import HALPE26
from . import person_detector
from .base import Estimator, Pose


class RTMPose(Estimator):
    key, label = "rtmpose", "RTMPose-m"
    names = HALPE26
    min_score = 0.3     # the usual drawing threshold in MMPose; scores are not probabilities and can exceed 1
    kind = "Top-down: person detector, then 26 points on the cut-out person"
    licence = "Apache-2.0"
    weights = MODELS / "rtmpose" / "rtmpose-m_halpe26_256x192.onnx"

    def __init__(self):
        from rtmlib import RTMPose as Model
        self.detector = person_detector.shared()
        self.model, self.runs_on = person_detector.load_onnx(Model, self.weights, model_input_size=(192, 256))

    def estimate(self, frame_bgr, t_ms):
        boxes, detector_ms = self.detector(frame_bgr)
        pick = self.subject.pick(boxes)
        if pick is None:
            return Pose(None, n_people=len(boxes), parts_ms={"detector": detector_ms, "pose": 0.0})
        started = time.perf_counter()
        points, scores = self.model(frame_bgr, bboxes=[boxes[pick]])
        kp = np.concatenate([points[0], scores[0][:, None]], axis=1).astype(np.float32)
        return Pose(kp, boxes[pick], len(boxes), parts_ms={"detector": detector_ms, "pose": 1000 * (time.perf_counter() - started)})
