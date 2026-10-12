"""OpenPose, 25-point body model with feet (Cao et al., CMU, 2019): the classic bottom-up method, on the GPU.

Bottom-up: the network looks at the whole frame once and outputs two kinds of maps. A
"confidence map" per body point lights up wherever that point appears, for anybody in
the picture. A pair of "part affinity field" maps per limb holds little arrows pointing
along that limb. Peaks in the confidence maps are the candidate points; the arrows say
which shoulder belongs with which elbow, and so the candidates are joined into people.

The 25 points are the 18 of the older model plus the middle of the hips and three points
per foot (big toe, small toe, heel).

These are CMU's original trained weights and network definition (BODY_25,
pose_iter_584000.caffemodel; its MD5 matches the one in CMU's build file). They are in
Caffe format, which is read by caffe.py. It is still not CMU's program: their program
needs Caffe and an old GPU toolkit that does not run on current graphics cards. The
steps from maps to people below are written to follow CMU's code (its default
thresholds and its way of scoring a limb), so small differences from their program in
resizing and peak finding remain possible.

Licence of the weights: OpenPose licence, non-commercial research use only.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .. import MODELS
from ..skeleton import OPENPOSE25
from .base import Estimator
from .caffe import CaffeNet

NET_HEIGHT = 368        # OpenPose's default: the frame is resized to this height before the network
STRIDE = 8              # the maps are 8 times smaller than the network input
MULTIPLE = 16           # OpenPose pads the network input to a multiple of 16
# CMU's defaults (poseParameters.cpp):
PEAK_MIN = 0.05         # a confidence-map peak lower than this is ignored
ARROW_MIN = 0.05        # a sample along a limb agrees with the field when the dot product exceeds this
AGREE_SHARE = 0.95      # this share of the samples along a limb must agree
MIN_POINTS, MIN_MEAN_SCORE = 3, 0.4     # a person needs this many points and this average score

N_POINTS = 25
# Limbs as pairs of points (indices into OPENPOSE25) and, for each, its two field channels (x, y). From CMU's source.
LIMBS = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
         (1, 0), (0, 15), (15, 17), (0, 16), (16, 18), (2, 17), (5, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)]
FIELDS = [(0, 1), (14, 15), (22, 23), (16, 17), (18, 19), (24, 25), (26, 27), (6, 7), (2, 3), (4, 5), (8, 9), (10, 11), (12, 13),
          (30, 31), (32, 33), (36, 37), (34, 35), (38, 39), (20, 21), (28, 29), (40, 41), (42, 43), (44, 45), (46, 47), (48, 49), (50, 51)]
FILL_ONLY = {18, 19}    # shoulder-to-ear limbs: they may add a missing ear to a person, never start or merge people


def find_peaks(maps):
    """Candidates per body point: list of arrays [n, 3] (x, y, score) in network-input pixels."""
    is_peak = (maps == F.max_pool2d(maps, 3, 1, 1)) & (maps > PEAK_MIN)
    maps = maps.cpu().numpy()
    height, width = maps.shape[1:]
    out = []
    for k, (ys, xs) in enumerate(np.nonzero(c) for c in is_peak.cpu().numpy()):
        found = []
        for y, x in zip(ys, xs):        # refine to a fraction of a pixel: score-weighted centre of the 7 x 7 neighbourhood
            y0, y1, x0, x1 = max(y - 3, 0), min(y + 4, height), max(x - 3, 0), min(x + 4, width)
            patch = np.clip(maps[k, y0:y1, x0:x1], 0, None)
            gy, gx = np.mgrid[y0:y1, x0:x1]
            found.append((float((gx * patch).sum() / patch.sum()), float((gy * patch).sum() / patch.sum()), float(maps[k, y, x])))
        out.append(np.array(found, np.float32).reshape(-1, 3))
    return out


def join_limbs(peaks, fields):
    """For each limb, the accepted connections [(index in peaks[a], index in peaks[b], score), ...]."""
    height, width = fields.shape[1:]
    connections = []
    for (a, b), (fx, fy) in zip(LIMBS, FIELDS):
        candidates = []
        for i, pa in enumerate(peaks[a]):
            for j, pb in enumerate(peaks[b]):
                vector = pb[:2] - pa[:2]
                length = float(np.hypot(*vector))
                if length < 1e-6:
                    continue
                samples = int(np.clip(round(np.sqrt(5 * np.abs(vector).max())), 5, 25))     # more samples for longer limbs
                steps = np.arange(samples)[:, None] * (vector / samples)
                xs = np.round(pa[0] + steps[:, 0]).astype(int).clip(0, width - 1)
                ys = np.round(pa[1] + steps[:, 1]).astype(int).clip(0, height - 1)
                agreement = (fields[fx, ys, xs] * vector[0] + fields[fy, ys, xs] * vector[1]) / length
                agreeing = agreement > ARROW_MIN
                if agreeing.sum() / samples > AGREE_SHARE:
                    candidates.append((i, j, float(agreement[agreeing].mean())))
        chosen, used_a, used_b = [], set(), set()
        for i, j, score in sorted(candidates, key=lambda c: -c[2]):     # best first; each point joins one limb of a kind
            if i not in used_a and j not in used_b:
                chosen.append((i, j, score))
                used_a.add(i)
                used_b.add(j)
        connections.append(chosen)
    return connections


def assemble(peaks, connections):
    """Chain the limbs into people: array [N, 25, 3] with NaN where a person has no such point."""
    people = []     # each: [index of the peak used for each point or -1, ..., total score, number of points]
    for k, ((a, b), chosen) in enumerate(zip(LIMBS, connections)):
        for i, j, score in chosen:
            owners = [p for p in people if p[a] == i or p[b] == j]
            if k in FILL_ONLY:
                for person in owners:
                    for slot, index in ((a, i), (b, j)):
                        if person[slot] < 0:
                            person[slot] = index
                            person[-1] += 1
                            person[-2] += peaks[slot][index][2] + score
            elif len(owners) == 1:
                person = owners[0]
                if person[b] != j:
                    person[b] = j
                    person[-1] += 1
                    person[-2] += peaks[b][j][2] + score
            elif len(owners) == 2:
                first, second = owners
                if not any(first[n] >= 0 and second[n] >= 0 for n in range(N_POINTS)):      # two halves of one person: merge
                    for n in range(N_POINTS):
                        if second[n] >= 0:
                            first[n] = second[n]
                    first[-2] += second[-2] + score
                    first[-1] += second[-1]
                    people[:] = [p for p in people if p is not second]
                else:
                    first[b] = j
                    first[-1] += 1
                    first[-2] += peaks[b][j][2] + score
            elif not owners:
                person = [-1] * N_POINTS + [peaks[a][i][2] + peaks[b][j][2] + score, 2]
                person[a], person[b] = i, j
                people.append(person)
    out = []
    for person in people:
        if person[-1] >= MIN_POINTS and person[-2] / person[-1] >= MIN_MEAN_SCORE:      # drop fragments, as the original does
            kp = np.full((N_POINTS, 3), np.nan, np.float32)
            for n in range(N_POINTS):
                if person[n] >= 0:
                    kp[n] = peaks[n][person[n]]
            out.append(kp)
    return out


class OpenPose(Estimator):
    key, label = "openpose", "OpenPose"
    names = OPENPOSE25
    min_score = PEAK_MIN        # a point that was found at all passed the peak threshold; a missing one is NaN
    kind = "Bottom-up: finds all body points in the frame, then joins them into people"
    licence = "OpenPose licence (non-commercial research only)"
    weights = MODELS / "openpose" / "body_25"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.runs_on = "GPU" if self.device == "cuda" else "CPU"
        self.net = CaffeNet(self.weights / "pose_deploy.prototxt", self.weights / "pose_iter_584000.caffemodel").to(self.device).eval()

    def estimate(self, frame_bgr, t_ms):
        scale = NET_HEIGHT / frame_bgr.shape[0]
        # shrinking needs INTER_AREA, which averages the pixels it merges; a plain cubic resize skips pixels and makes the maps flicker
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        height, width = small.shape[:2]
        padded = cv2.copyMakeBorder(small, 0, -height % MULTIPLE, 0, -width % MULTIPLE, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        image = torch.from_numpy(padded).to(self.device).permute(2, 0, 1)[None].float() / 256 - 0.5     # stays in BGR order
        with torch.inference_mode():
            output = self.net(image)        # channels: 25 confidence maps, 1 background map, then 52 field maps
            # back to network-input size, as the original does before looking for peaks
            output = F.interpolate(output, scale_factor=STRIDE, mode="bicubic", align_corners=False)[0, :, :height, :width]
            peaks = find_peaks(output[:N_POINTS])
            fields = output[N_POINTS + 1:].cpu().numpy()
        people = assemble(peaks, join_limbs(peaks, fields))
        for kp in people:
            kp[:, :2] = (kp[:, :2] + 0.5) / scale - 0.5      # pixel centres of the small image -> full-frame pixels
        return self.choose(people)
