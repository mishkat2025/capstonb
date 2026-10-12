"""Read a video frame by frame, with the timestamp the file gives each frame."""
from pathlib import Path

import cv2


class Video:
    """Iterating yields (frame index, timestamp in ms, BGR image). Phone rotation tags are applied by OpenCV."""

    def __init__(self, path):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise OSError(f"cannot open video: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))      # the container's own count; can be off by a few
        self.width = self.height = 0

    def __iter__(self):
        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.height, self.width = frame.shape[:2]
            yield index, self.cap.get(cv2.CAP_PROP_POS_MSEC), frame
            index += 1

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
