"""YOLO26x-pose (Ultralytics, January 2026), on the GPU.

One-stage: a single network looks at the whole frame once and outputs, for every person
it finds, a box and 17 body points together. No separate person detector and no cut-outs.

Licence: AGPL-3.0. Fine for this comparison; an app that ships this model must itself be
released under AGPL-3.0 or buy Ultralytics' enterprise licence.
"""
from .. import MODELS
from ..skeleton import COCO17
from .base import Estimator


class YoloPose(Estimator):
    key, label = "yolo", "YOLO26x-pose"
    names = COCO17
    min_score = 0.5     # the visibility cut Ultralytics itself uses when drawing
    kind = "One-stage: one network outputs boxes and 17 points for everyone at once"
    licence = "AGPL-3.0"
    weights = MODELS / "yolo" / "yolo26x-pose.pt"

    def __init__(self):
        import torch
        from ultralytics import YOLO, settings
        if settings.get("sync"):
            settings.update({"sync": False})        # do not send usage statistics to Ultralytics
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.runs_on = "GPU" if self.device == 0 else "CPU"
        self.model = YOLO(str(self.weights))

    def estimate(self, frame_bgr, t_ms):
        result = self.model.predict(frame_bgr, verbose=False, device=self.device, imgsz=640)[0]
        if result.keypoints is None or len(result.boxes) == 0:
            return self.choose([])
        people = result.keypoints.data.cpu().numpy()        # [N, 17, 3]: x, y, confidence
        return self.choose(list(people), list(result.boxes.xyxy.cpu().numpy()))
