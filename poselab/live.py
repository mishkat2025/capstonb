"""The window shown while a video is processed, and the recording of it.

Used as the on_frame callback of run.process(). Keys in the window: Space pauses, Q or Esc
stops (a report is still written for the frames done so far). Closing the window stops too.
"""
import time

import cv2

from .draw import LiveCanvas

WINDOW = "Pose comparison   |   Space: pause   Q: stop"
SHEET_HEIGHT = 480      # frames kept for the report's overlay sheet are shrunk to this height
# Windows' H.264 encoder, as OpenCV drives it, spends one bit per pixel per frame whatever the content
# (measured: 40 MB for 10 s at full size and rate). Recording every second frame at 3/4 size brings
# that to roughly 1 MB per second of video.
RECORD_EVERY, RECORD_SCALE = 2, 0.75


class LiveView:
    def __init__(self, models, video, out_dir, show=True, record=True, screen=(1500, 860), with_hands=False):
        """models: {key: estimator}; video: an open poselab.video.Video (for size, rate and length)."""
        self.models, self.show, self.record, self.out_dir = models, show, record, out_dir
        self.with_hands = with_hands
        self.hand_samples = []  # (frame index, {"left": Hand or None, "right": ...}) for the report's hand sheet
        self.info = [(key, m.label, m.names, m.min_score) for key, m in models.items()]
        self.fps = video.fps
        self.total = max(video.frame_count, 1)
        self.screen = screen
        self.canvas = None
        self.writer, self.video_file, self.codec = None, None, None
        self.keep_every = max(1, self.total // 48)
        self.samples = []       # (frame index, shrunk frame) for the report's overlay sheet
        self.started = time.perf_counter()
        self.window_open = False

    def _open(self, frame):
        height, width = frame.shape[:2]
        space_key = next((key for key, m in self.models.items() if m.has_3d), None)       # a model with 3D points gets a 3D panel
        self.canvas = LiveCanvas(self.info, width, height, self.total / self.fps, *self.screen, with_hands=self.with_hands, space_key=space_key)
        size = (self.canvas.canvas_w, self.canvas.canvas_h)
        self.record_size = tuple(int(v * RECORD_SCALE) // 2 * 2 for v in size)      # encoders want even sizes
        # H.264 through Windows' own encoder plays in a browser; MPEG-4 is the fallback and needs a video player.
        for api, fourcc, codec in ((cv2.CAP_MSMF, "H264", "H.264"), (cv2.CAP_FFMPEG, "mp4v", "MPEG-4")) if self.record else ():
            target = self.out_dir / "live_view.mp4"
            writer = cv2.VideoWriter(str(target), api, cv2.VideoWriter_fourcc(*fourcc), self.fps / RECORD_EVERY, self.record_size)
            if writer.isOpened():
                self.writer, self.video_file, self.codec = writer, target, codec
                break
        if self.show:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(WINDOW, *size)
            self.window_open = True

    def __call__(self, index, t_ms, frame, poses, frame_ms, hands=None):
        if self.canvas is None:
            self._open(frame)
        done = index + 1
        rate = done / max(time.perf_counter() - self.started, 1e-6)
        status = f"frame {done} of about {self.total}   {rate:.1f} frames/s"
        image = self.canvas.frame(t_ms / 1000.0, frame, poses, frame_ms, status, hands)
        if self.writer is not None and index % RECORD_EVERY == 0:
            self.writer.write(cv2.resize(image, self.record_size, interpolation=cv2.INTER_AREA))
        if index % self.keep_every == 0:
            scale = SHEET_HEIGHT / frame.shape[0]
            self.samples.append((index, cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)))
            if hands is not None:
                self.hand_samples.append((index, hands))
        if not self.show:
            return True
        cv2.imshow(WINDOW, image)
        paused_at = None
        while True:
            key = cv2.waitKey(30 if paused_at else 1) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                self.window_open = False
                return False
            if key in (ord("q"), ord("Q"), 27):
                return False
            if key == ord(" "):
                if paused_at is None:
                    paused_at = time.perf_counter()
                else:
                    self.started += time.perf_counter() - paused_at     # keep the pause out of the frames/s figure
                    paused_at = None
            if paused_at is None:
                return True

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.window_open:
            cv2.destroyWindow(WINDOW)
            cv2.waitKey(1)
            self.window_open = False
