"""Skeleton overlays and the live comparison view (OpenCV drawing; colours are BGR).

The participant's right side is orange and the left side blue, as in the Stage 1 sheets.
A joint the model is confident about is a white dot; one below the model's own confidence
threshold is a small grey dot, and the bones touching it are drawn thin.
"""
import cv2
import numpy as np

from . import analysis
from .skeleton import BONES, DRAWN_JOINTS, J, to_canonical

SIDE_COLOUR = {"right": (52, 104, 235), "left": (214, 120, 42), "mid": (240, 240, 240)}
SURFACE, INK, INK2, MUTED, GRID = (251, 252, 252), (11, 11, 11), (78, 81, 82), (129, 135, 137), (217, 224, 225)
# one colour per model, in registry order (the validated categorical palette, as BGR)
SERIES = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (164, 123, 232)]
SERIES_HEX = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(image, string, origin, size=0.5, colour=INK, thickness=1):
    cv2.putText(image, string, origin, FONT, size, colour, thickness, cv2.LINE_AA)


def skeleton(image, px, score, min_score, scale=1.0, weight=2):
    """Draw the skeleton px [J, 2] (shared joints, full-frame pixels) on image, which is the frame resized by scale."""
    pts = px * scale
    ok = np.isfinite(pts).all(axis=1)
    sure = ok & (np.nan_to_num(score, nan=0.0) >= min_score)
    for a, b, side in BONES:
        if ok[a] and ok[b]:
            cv2.line(image, tuple(map(int, pts[a])), tuple(map(int, pts[b])), SIDE_COLOUR[side],
                     weight if sure[a] and sure[b] else 1, cv2.LINE_AA)
    for j in DRAWN_JOINTS:
        if sure[j]:
            cv2.circle(image, tuple(map(int, pts[j])), weight + 1, (255, 255, 255), -1, cv2.LINE_AA)
        elif ok[j]:
            cv2.circle(image, tuple(map(int, pts[j])), weight, (150, 150, 150), -1, cv2.LINE_AA)
    return image


# The hand model's 21 points: 0 wrist, 1-4 thumb, then four points per finger from knuckle to tip.
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 9), (9, 13), (13, 17), (0, 5), (0, 17)] + \
             [(k + i, k + i + 1) for k in (5, 9, 13, 17) for i in range(3)]
HAND_TIPS = [4, 8, 12, 16, 20]
HAND_COLOUR = (122, 175, 27)


def hand(image, points, scale=1.0, weight=1):
    """Draw one hand's 21 points [21, 2] on image; points are in pixels of the picture that image is a scaled copy of."""
    pts = (np.asarray(points) * scale).round().astype(int)
    for a, b in HAND_LINKS:
        cv2.line(image, tuple(map(int, pts[a])), tuple(map(int, pts[b])), HAND_COLOUR, weight, cv2.LINE_AA)
    for tip in HAND_TIPS:
        cv2.circle(image, tuple(map(int, pts[tip])), weight + 1, (255, 255, 255), -1, cv2.LINE_AA)
    return image


def hand_tile(one_hand, size, label):
    """A square picture of one hand's cut-out with its points and state; a grey square when the hand was not found."""
    if one_hand is None:
        tile = np.full((size, size, 3), (236, 239, 240), np.uint8)
        text(tile, label, (5, 14), 0.4, INK2)
        text(tile, "not found", (5, size // 2), 0.4, MUTED)
        return tile
    tile = cv2.resize(one_hand.cutout, (size, size), interpolation=cv2.INTER_AREA)
    hand(tile, one_hand.cutout_points, size / one_hand.cutout.shape[0])
    note = f"{label}: {one_hand.state} {one_hand.extension:.2f}"
    (w, h), _ = cv2.getTextSize(note, FONT, 0.4, 1)
    cv2.rectangle(tile, (0, 0), (min(size, w + 8), h + 9), (0, 0, 0), -1)
    text(tile, note, (4, h + 4), 0.4, (255, 255, 255))
    return tile


# ---- the 3D skeleton, drawn from two directions the camera does not have ----
SPACE_X, SPACE_Z = 0.85, 0.60           # metres shown either side of the hip centre: across the picture, and in depth
SPACE_UP, SPACE_DOWN = 1.15, 0.95       # metres shown above and below the hip centre
SPACE_GROUND, SPACE_LINE, SPACE_TEXT = (40, 38, 36), (96, 94, 92), (206, 204, 200)
HEAD_M = 0.10                           # radius of the circle that stands for the head
LEGS = [J[f"{side}_{part}"] for side in ("left", "right") for part in ("knee", "ankle", "heel", "toe")]


def space_tile(world, score, min_score, width, height):
    """A picture of the 3D points [J, 3] (shared joints, metres around the hip centre) from the side and from above. world None: not found.

    Both views keep the camera's own directions, so nothing is assumed about how the person
    stands. Side view: the camera is on the left and up is up. Top view: the camera is at
    the bottom and left and right are as in the video; the legs are left out so that the
    arms stay readable.
    """
    tile = np.full((height, width, 3), SPACE_GROUND, np.uint8)
    cap = 18                                                    # room for a view's caption
    side_w, side_h, top_w, top_h = 2 * SPACE_Z, SPACE_UP + SPACE_DOWN, 2 * SPACE_X, 2 * SPACE_Z
    beside = min(width / (side_w + top_w), (height - cap) / side_h)
    stacked = min(width / max(side_w, top_w), (height - 2 * cap) / (side_h + top_h))
    if beside >= stacked:                                       # whichever arrangement shows the body larger
        scale, split = beside, int(width * side_w / (side_w + top_w))
        (sx, sy, sw, sh), (tx, ty, tw, th) = (0, 0, split, height), (split, 0, width - split, height)
        cv2.line(tile, (split, 0), (split, height), SPACE_LINE, 1)
    else:
        scale, split = stacked, int(cap + (height - 2 * cap) * side_h / (side_h + top_h))
        (sx, sy, sw, sh), (tx, ty, tw, th) = (0, 0, width, split), (0, split, width, height - split)
        cv2.line(tile, (0, split), (width, split), SPACE_LINE, 1)
    side_origin = np.array([sx + sw / 2, sy + cap + (sh - cap - scale * side_h) / 2 + scale * SPACE_UP])
    top_origin = np.array([tx + tw / 2, ty + cap + (th - cap) / 2])

    # reference lines through the hip centre: straight up in the side view, parallel to the picture in the top view
    cv2.line(tile, (int(side_origin[0]), sy + cap + 4), (int(side_origin[0]), sy + sh - 4), SPACE_LINE, 1)
    cv2.line(tile, (tx + 6, int(top_origin[1])), (tx + tw - 6, int(top_origin[1])), SPACE_LINE, 1)
    text(tile, "From the side", (sx + 6, sy + 13), 0.4, SPACE_TEXT)
    text(tile, "From above", (tx + 6, ty + 13), 0.4, SPACE_TEXT)
    y = sy + sh - 12                                            # where the camera is: a small arrow pointing into the scene
    cv2.fillConvexPoly(tile, np.array([(sx + 5, y - 5), (sx + 5, y + 5), (sx + 13, y)], np.int32), SPACE_TEXT, cv2.LINE_AA)
    text(tile, "camera", (sx + 17, y + 4), 0.35, SPACE_TEXT)
    x, y = int(top_origin[0]) - 26, ty + th - 5
    cv2.fillConvexPoly(tile, np.array([(x - 5, y), (x + 5, y), (x, y - 8)], np.int32), SPACE_TEXT, cv2.LINE_AA)
    text(tile, "camera", (x + 9, y), 0.35, SPACE_TEXT)
    bar, note = int(0.5 * scale), "0.5 m, estimated"            # the model's own metres: a guess, like the depth
    cv2.line(tile, (width - 8 - bar, height - 7), (width - 8, height - 7), SPACE_TEXT, 1)
    text(tile, note, (width - 8 - cv2.getTextSize(note, FONT, 0.35, 1)[0][0], height - 14), 0.35, SPACE_TEXT)

    if world is None or not np.isfinite(world).any():
        text(tile, "no 3D points in this frame", (sx + 6, sy + 34), 0.45, (90, 90, 235))
        return tile
    views = ((side_origin, world[:, [2, 1]], ()), (top_origin, world[:, [0, 2]] * [1, -1], LEGS))
    for origin, flat, hidden in views:
        px = origin + scale * flat
        px[list(hidden)] = np.nan
        skeleton(tile, px, score, min_score)
        ears, nose = px[[J["left_ear"], J["right_ear"]]], px[J["nose"]]
        if np.isfinite(ears).all():
            cv2.circle(tile, tuple(map(int, ears.mean(axis=0))), max(3, int(HEAD_M * scale)), SPACE_TEXT, 1, cv2.LINE_AA)
        if np.isfinite(nose).all():
            cv2.circle(tile, tuple(map(int, nose)), 2, SPACE_TEXT, -1, cv2.LINE_AA)
    return tile


class LiveCanvas:
    """Builds the picture shown while a video is processed: one panel per model, and a growing chart underneath."""
    HEADER, STRIP, PAD, HAND = 30, 190, 8, 150

    def __init__(self, tracks_info, width, height, duration_s, max_width=1500, max_height=860, with_hands=False, space_key=None):
        """tracks_info: list of (key, label, names, min_score), in display order. space_key: the model whose 3D points get a panel of their own."""
        self.info = tracks_info
        self.with_hands = with_hands
        self.space_key = space_key
        n, best = len(tracks_info) + (space_key is not None), None
        for cols in range(1, n + 1):       # the arrangement that gives the largest panels within the size limit
            rows = -(-n // cols)
            panel_h = (max_height - self.STRIP) / rows - self.HEADER - self.PAD
            panel_w = min((max_width - self.PAD) / cols - self.PAD, panel_h * width / height)
            if best is None or panel_w > best[1]:
                best = (cols, panel_w)
        self.cols, self.pw = best[0], int(best[1])
        self.rows = -(-n // self.cols)
        self.ph = int(self.pw * height / width)
        self.scale = self.pw / width
        self.canvas_w = max(self.cols * (self.pw + self.PAD) + self.PAD, 640)
        self.canvas_h = self.rows * (self.ph + self.HEADER + self.PAD) + self.STRIP
        self.canvas_w += self.canvas_w % 2      # video encoders want even sizes
        self.canvas_h += self.canvas_h % 2
        self.duration = max(duration_s, 1e-3)
        self.history = {key: {"t": [], "left": [], "right": []} for key, *_ in tracks_info}

    def frame(self, t_s, frame_bgr, poses, frame_ms, status="", hands=None):
        """hands: {"left": Hand or None, "right": ...} from hands.HandReader, drawn on every panel since it is the same for all models."""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), SURFACE, np.uint8)
        small = cv2.resize(frame_bgr, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
        corner = lambda n: (self.PAD + (n % self.cols) * (self.pw + self.PAD), (n // self.cols) * (self.ph + self.HEADER + self.PAD))
        for n, (key, label, names, min_score) in enumerate(self.info):
            x, y = corner(n)
            tile = small.copy()
            pose = poses.get(key)
            elevation = {"left": np.nan, "right": np.nan}
            if pose is not None and pose.kp is not None:
                joints = to_canonical(pose.kp, names)
                skeleton(tile, joints[:, :2], joints[:, 2], min_score, self.scale)
                for side in elevation:
                    elevation[side] = float(analysis.arm_elevation(joints[None, :, :2], side)[0])
            else:
                text(tile, "participant not found", (8, 22), 0.5, (40, 40, 220))
            for one_hand in (hands or {}).values():
                if one_hand is not None:
                    hand(tile, one_hand.points, self.scale)
            h = self.history[key]
            h["t"].append(t_s)
            h["left"].append(elevation["left"])
            h["right"].append(elevation["right"])
            canvas[y + self.HEADER:y + self.HEADER + self.ph, x:x + self.pw] = tile
            cv2.circle(canvas, (x + 7, y + 15), 5, SERIES[n % len(SERIES)], -1, cv2.LINE_AA)
            text(canvas, label, (x + 18, y + 20), 0.5, INK, 1)
            ms = frame_ms.get(key)
            if ms is not None:
                note = f"{ms:.0f} ms"
                (w, _), _ = cv2.getTextSize(note, FONT, 0.45, 1)
                text(canvas, note, (x + self.pw - w - 2, y + 20), 0.45, INK2)
            if key == self.space_key:       # the same model's 3D points, in the panel after the last model
                found = pose is not None and pose.kp is not None and pose.world is not None
                x, y = corner(len(self.info))
                canvas[y + self.HEADER:y + self.HEADER + self.ph, x:x + self.pw] = space_tile(
                    to_canonical(pose.world, names) if found else None, joints[:, 2] if found else None, min_score, self.pw, self.ph)
                cv2.circle(canvas, (x + 7, y + 15), 5, SERIES[n % len(SERIES)], -1, cv2.LINE_AA)
                text(canvas, f"{label} in 3D", (x + 18, y + 20), 0.5, INK, 1)
        self._strip(canvas, t_s, status, hands)
        return canvas

    def _strip(self, canvas, t_s, status, hands):
        """Arm elevation over time for every model: the arm that has moved more so far. With hands on, both hands enlarged at the right."""
        top = self.canvas_h - self.STRIP + 26
        left, right, bottom = 52, self.canvas_w - 14, self.canvas_h - 30
        if self.with_hands:
            y = self.canvas_h - self.STRIP + (self.STRIP - self.HAND) // 2
            for n, side in enumerate(("right", "left")):       # the participant's right hand is on the picture's left in a front view
                x = self.canvas_w - 14 - (2 - n) * self.HAND - (1 - n) * 6
                canvas[y:y + self.HAND, x:x + self.HAND] = hand_tile((hands or {}).get(side), self.HAND, f"{side} hand")
            right -= 2 * self.HAND + 26
        text(canvas, "Arm elevation (degrees), arm that has moved more so far", (left, top - 10), 0.45, INK2)
        if status:
            (w, _), _ = cv2.getTextSize(status, FONT, 0.45, 1)
            text(canvas, status, (right - w - 6, top - 14), 0.45, INK2)
        for value in (0, 90, 180):
            y = int(bottom - value / 180 * (bottom - top))
            cv2.line(canvas, (left, y), (right, y), GRID, 1)
            text(canvas, str(value), (left - 34, y + 4), 0.4, MUTED)
        for seconds in range(0, int(self.duration) + 1, max(1, int(self.duration) // 8)):
            x = int(left + seconds / self.duration * (right - left))
            text(canvas, f"{seconds}s", (x - 8, bottom + 18), 0.4, MUTED)
        for n, (key, *_) in enumerate(self.info):
            h = self.history[key]
            side = max(("left", "right"), key=lambda s: np.ptp(np.asarray(h[s])[np.isfinite(h[s])]) if np.isfinite(h[s]).sum() > 1 else -1)
            values, times = np.asarray(h[side]), np.asarray(h["t"])
            xs = left + times / self.duration * (right - left)
            ys = bottom - values / 180 * (bottom - top)
            ok = np.isfinite(ys)
            run_edges = np.flatnonzero(np.diff(np.r_[False, ok, False]))
            for start, stop in zip(run_edges[::2], run_edges[1::2]):       # a gap in the line where the angle is unknown
                if stop - start > 1:
                    line = np.stack([xs[start:stop], ys[start:stop]], axis=1).round().astype(np.int32)
                    cv2.polylines(canvas, [line], False, SERIES[n % len(SERIES)], 2, cv2.LINE_AA)
        x = int(left + min(t_s, self.duration) / self.duration * (right - left))
        cv2.line(canvas, (x, top), (x, bottom), MUTED, 1)
