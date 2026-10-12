"""MediaPipe Pose Landmarker, Full variant (Google). 33 points including heels and toe tips.

It finds the person once, then follows them from frame to frame with its own tracker, so
each video needs a fresh landmarker. Settings are the ones validated in Stage 1: video
mode, one pose per frame, default confidence thresholds. The Windows build has no GPU
inference, so it runs on the CPU.
"""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .. import MODELS
from ..skeleton import MEDIAPIPE33
from .base import Estimator, Pose


class MediaPipeFull(Estimator):
    key, label = "mediapipe", "MediaPipe Full"
    names = MEDIAPIPE33
    min_score = 0.5     # "visibility": the model's own estimate that the point is in view and not hidden
    runs_on = "CPU"
    kind = "Detects the person once, then tracks; predicts 33 points with depth"
    licence = "Apache-2.0"
    has_3d = True       # its "world landmarks": the same 33 points in metres around the middle of the hips
    weights = MODELS / "mediapipe" / "pose_landmarker_full.task"

    def __init__(self):
        self.landmarker = None

    def start(self, width, height, fps):
        super().start(width, height, fps)
        self.close()
        self.landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(self.weights)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
        self.last_ts = -1

    def estimate(self, frame_bgr, t_ms):
        height, width = frame_bgr.shape[:2]
        ts = max(self.last_ts + 1, int(round(t_ms)))      # video mode needs strictly increasing integer timestamps
        self.last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(image, ts)
        if not result.pose_landmarks:
            return self.choose([])
        people = [np.array([(q.x * width, q.y * height, q.visibility) for q in pose], np.float32) for pose in result.pose_landmarks]
        pose = self.choose(people)
        if pose.kp is not None:
            pose.world = np.array([(q.x, q.y, q.z) for q in result.pose_world_landmarks[0]], np.float32)
        return pose

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
