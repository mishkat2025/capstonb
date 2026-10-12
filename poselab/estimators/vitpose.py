"""ViTPose Base, whole-body version (Xu et al., 2022), on the GPU: 17 body points and 6 foot points from one network.

Top-down like RTMPose, and given the same person boxes. The cut-out person (256 x 192)
goes through a plain vision transformer and a small decoder that outputs one heatmap per
point; the point is the heatmap's peak.

This network was trained on COCO-WholeBody and outputs 133 points: 17 body, 6 feet, 68
face and 42 hand points. The first 23 are kept. (The hand points are not used: hands.py
reads the hands for every pose model alike.)

The file is an ONNX conversion of the authors' checkpoint by the easy_ViTPose project (a
third party, Apache-2.0). There is no official checksum to compare it with, so it was
checked against the officially hosted ViTPose++ Base instead (2026-10-12, three clips):
its 17 body points lie 1.4% to 2.1% of a torso length from that model's, as close as the
closest pair among the five models.

How the heatmap positions map back to the picture is not documented for this
conversion. Three standard conventions were tried in that check. Dividing the position
by the heatmap size gave the agreement above; the other two gave 6% to 7%.

Until 2026-10-12 the body points came from ViTPose++ Base through Hugging Face
transformers, which has no foot points. That version and its weights are in
archive/2026-10-12_vitpose_plus_base/.
"""
import time

import cv2
import numpy as np

from .. import MODELS
from ..skeleton import COCO_WHOLEBODY_BODY_FEET
from . import person_detector
from .base import Estimator, Pose

KEPT = len(COCO_WHOLEBODY_BODY_FEET)        # the first 23 of the 133 points: body, then feet


class ViTPose(Estimator):
    key, label = "vitpose", "ViTPose Base"
    names = COCO_WHOLEBODY_BODY_FEET
    min_score = 0.3     # the usual drawing threshold for ViTPose; the score is the height of the heatmap peak
    kind = "Top-down: person detector, then a vision transformer predicts one heatmap per point"
    licence = "Apache-2.0"
    weights = MODELS / "vitpose" / "vitpose-b-wholebody.onnx"

    def __init__(self):
        from rtmlib import ViTPose as Model
        from rtmlib.tools.pose_estimation.post_processings import post_dark_udp
        self.detector = person_detector.shared()
        self.model, self.runs_on = person_detector.load_onnx(Model, self.weights, model_input_size=(192, 256))
        self.refine = post_dark_udp         # moves a peak to sub-cell precision from the shape of the heatmap around it

    def estimate(self, frame_bgr, t_ms):
        boxes, detector_ms = self.detector(frame_bgr)
        pick = self.subject.pick(boxes)
        if pick is None:
            return Pose(None, n_people=len(boxes), parts_ms={"detector": detector_ms, "pose": 0.0})
        started = time.perf_counter()
        image, center, scale = self.model.preprocess(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), boxes[pick])      # this model wants RGB
        heatmaps = self.model.inference(image)[0][:, :KEPT].copy()             # [1, 23, 64, 48]
        _, count, height, width = heatmaps.shape
        flat = heatmaps.reshape(1, count, -1)
        scores = flat.max(axis=2)[0]
        cells = np.stack([flat.argmax(axis=2) % width, flat.argmax(axis=2) // width], axis=-1).astype(np.float32)
        cells = self.refine(cells, heatmaps, kernel=11)[0]
        points = cells / [width, height] * scale + center - scale / 2          # the convention found to fit (see the module text)
        kp = np.concatenate([points, scores[:, None]], axis=1).astype(np.float32)
        return Pose(kp, boxes[pick], len(boxes), parts_ms={"detector": detector_ms, "pose": 1000 * (time.perf_counter() - started)})
