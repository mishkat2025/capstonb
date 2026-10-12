"""Person detector shared by the two top-down models (ViTPose and RTMPose).

A top-down pose model only looks at one cut-out person at a time, so something has to
find the people first. Both models are given the same boxes from the same detector
(YOLOX-m trained on COCO and HumanArt, published with RTMPose), so that a difference
between them comes from the pose model and not from the detector.

The detector result for a frame is kept until the next frame, so it runs once per frame
however many models ask for it.
"""
import contextlib
import io
import time

import numpy as np

from .. import MODELS

WEIGHTS = MODELS / "person_detector" / "yolox-m_humanart_640.onnx"
_shared = None


def load_onnx(tool, weights, **kwargs):
    """Create an rtmlib tool on the GPU when ONNX Runtime can use it, else on the CPU. Returns (tool, "GPU" or "CPU")."""
    import onnxruntime as ort
    ort.set_default_logger_severity(3)      # hide the note that a few small operations stay on the CPU
    with contextlib.suppress(Exception):
        ort.preload_dlls()                  # the CUDA libraries that come with PyTorch
    device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    with contextlib.redirect_stdout(io.StringIO()):
        model = tool(str(weights), backend="onnxruntime", device=device, **kwargs)
    on_gpu = "CUDAExecutionProvider" in model.session.get_providers()
    return model, "GPU" if on_gpu else "CPU"


class PersonDetector:
    def __init__(self):
        from rtmlib import YOLOX
        self.model, self.runs_on = load_onnx(YOLOX, WEIGHTS, model_input_size=(640, 640))
        self._frame, self._boxes, self._ms = None, None, 0.0

    def __call__(self, frame_bgr):
        """Boxes [N, 4] (x1, y1, x2, y2) of the people in the frame, and the time the detection took in ms."""
        if frame_bgr is not self._frame:
            started = time.perf_counter()
            boxes = np.asarray(self.model(frame_bgr), np.float32).reshape(-1, 4)
            self._frame, self._boxes, self._ms = frame_bgr, boxes, 1000 * (time.perf_counter() - started)
        return self._boxes, self._ms


def shared():
    global _shared
    if _shared is None:
        _shared = PersonDetector()
    return _shared
