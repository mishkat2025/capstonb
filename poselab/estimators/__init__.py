"""The pose models, one file each. Nothing heavy is imported until a model is created."""
from importlib import import_module

# key: (module, class). The order is the display order and fixes each model's chart colour.
REGISTRY = {
    "mediapipe": ("mediapipe_full", "MediaPipeFull"),
    "vitpose": ("vitpose", "ViTPose"),
    "openpose": ("openpose", "OpenPose"),
    "rtmpose": ("rtmpose", "RTMPose"),
    "yolo": ("yolo_pose", "YoloPose"),
}
LABELS = {"mediapipe": "MediaPipe Full", "vitpose": "ViTPose Base", "openpose": "OpenPose",
          "rtmpose": "RTMPose-m", "yolo": "YOLO26x-pose"}
NOTES = {"mediapipe": "33 points with feet, also in 3D, runs on the CPU", "vitpose": "23 points with feet, top-down transformer",
         "openpose": "25 points with feet, the classic bottom-up method", "rtmpose": "26 points with feet, fast top-down",
         "yolo": "17 body points, one-stage"}


def create(key, announce=print):
    """Load one model. Slow the first time: weights are read and, for GPU models, moved to the graphics card.

    Weight files that are not on disk yet are downloaded first (see weights.py).
    """
    from .. import weights
    weights.ensure(key, announce)
    module, name = REGISTRY[key]
    return getattr(import_module(f"{__name__}.{module}"), name)()
