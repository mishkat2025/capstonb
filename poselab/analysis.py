"""Geometry, frame quality flags, movement signal and clip features, for any pose model.

Everything here works on a "clip": a dict holding one person's joints over time in the
shared joint vocabulary of skeleton.py (J joints).

    t          [T]         seconds
    px         [T, J, 2]   pixels of the full frame; NaN where the model gave no such point
    score      [T, J]      the model's own confidence for each point
    min_score              below this a point counts as not confident (differs per model)
    has_pose   [T]         the model found the participant in this frame
    swapped    [T]         left/right labels were exchanged back (see relabel_front_view)
    width, height, fps
    world      [T, J, 3]   metres, only for models that estimate 3D; otherwise None

All 2D angles are measured in pixels, never in coordinates normalised by width and
height: those stretch x and y differently in any frame that is not square.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from .skeleton import ARM, L_HIP, L_SHOULDER, MIRROR, R_HIP, R_SHOULDER

FS = 30.0                 # common sampling rate for signals, Hz
LOWPASS_HZ = 3.0
SCALE_JUMP = 0.20         # torso length vs clip median
TELEPORT = 0.5            # hip-centre step between frames, in torso lengths
BONE_JUMP = 0.25          # 3D limb length vs clip median
MAX_FILL_S = 0.3          # gaps up to this long are interpolated
FLAGS = ["no_pose", "low_confidence", "out_of_frame", "scale_jump", "teleport", "bone_length"]


def angle_deg(u, v):
    """Angle between row vectors of u and v, in degrees."""
    dot = np.sum(u * v, axis=-1)
    norm = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))


def relabel_front_view(clip):
    """Undo left/right label exchanges in a front view; the frames changed are marked in clip["swapped"].

    A model sometimes labels the body back to front while the points themselves stay on
    the right joints. A person facing the camera has their left shoulder on the image
    right, so frames that say otherwise are relabelled.
    """
    px = clip["px"]
    with np.errstate(invalid="ignore"):
        swapped = px[:, L_SHOULDER, 0] < px[:, R_SHOULDER, 0]
    for key in ("px", "score", "world"):
        if clip.get(key) is not None:
            clip[key][swapped] = clip[key][swapped][:, MIRROR]
    clip["swapped"] = swapped
    return clip


def trunk(points):
    """Shoulder midpoint, hip midpoint and torso length for 2D or 3D points [T, J, D]."""
    shoulders = points[:, [L_SHOULDER, R_SHOULDER]].mean(axis=1)
    hips = points[:, [L_HIP, R_HIP]].mean(axis=1)
    return shoulders, hips, np.linalg.norm(shoulders - hips, axis=-1)


def arm_elevation(points, side):
    """Angle between the upper arm and the trunk line: 0 hanging, 90 horizontal, 180 overhead."""
    shoulder, elbow, _ = ARM[side]
    shoulders, hips, _ = trunk(points)
    return angle_deg(points[:, elbow] - points[:, shoulder], hips - shoulders)


def elbow_angle(points, side):
    shoulder, elbow, wrist = ARM[side]
    return angle_deg(points[:, shoulder] - points[:, elbow], points[:, wrist] - points[:, elbow])


def trunk_inclination(px):
    """Angle between the trunk and image vertical: about 0 standing, about 90 lying."""
    shoulders, hips, _ = trunk(px)
    up = shoulders - hips
    return angle_deg(up, np.broadcast_to([0.0, -1.0], up.shape))


def robust_range(x):
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 95) - np.percentile(x, 5)) if len(x) >= 5 else np.nan


def frame_flags(clip, side, use_3d=True):
    """Boolean array [T, len(FLAGS)] for the given moving arm; a frame is usable when no flag is set.

    The limb-length check needs 3D points. It is skipped when the clip has none or when
    use_3d is False (used to judge every model by the same five checks).
    """
    px, score, has = clip["px"], clip["score"], clip["has_pose"]
    flags = np.zeros((len(has), len(FLAGS)), bool)
    flags[:, 0] = ~has
    if not has.any():
        return flags
    shoulder, elbow, wrist = ARM[side]
    required = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, elbow, wrist]
    with np.errstate(invalid="ignore"):
        points = px[:, required]
        missing = ~np.isfinite(points).all(axis=(1, 2))
        flags[:, 1] = has & (missing | (np.nan_to_num(score[:, required], nan=1.0).min(axis=1) < clip["min_score"]))
        outside = (points < 0) | (points > [clip["width"], clip["height"]])
        flags[:, 2] = has & outside.any(axis=(1, 2))

        # The exerciser stays at one distance for the whole clip, so body size is compared with the
        # clip median. A running median would adapt to a passer-by who holds the track for a second.
        _, hips, torso = trunk(px)
        typical = np.nanmedian(torso)
        flags[:, 3] = has & (np.abs(torso - typical) > SCALE_JUMP * typical)
        step = np.linalg.norm(np.diff(hips, axis=0, prepend=hips[:1]), axis=1)
        flags[:, 4] = has & (step > TELEPORT * np.nanmedian(torso))

        world = clip.get("world")
        if use_3d and world is not None:
            for a, b in ((shoulder, elbow), (elbow, wrist)):
                length = np.linalg.norm(world[:, a] - world[:, b], axis=1)
                flags[:, 5] |= has & (np.abs(length - np.nanmedian(length)) > BONE_JUMP * np.nanmedian(length))
    return flags


def moving_side(clip, use_3d=True):
    """The arm whose elevation varies more, judged on the frames that are usable for that arm."""
    ranges = {}
    for s in ARM:
        usable = ~frame_flags(clip, s, use_3d).any(axis=1)
        ranges[s] = robust_range(arm_elevation(clip["px"], s)[usable])
    if not np.isfinite(list(ranges.values())).any():
        return "right", ranges
    return max(ranges, key=lambda s: -1 if np.isnan(ranges[s]) else ranges[s]), ranges


def gaps(t, usable):
    """Durations in seconds of each run of unusable frames."""
    out, start = [], None
    for i, ok in enumerate(usable):
        if not ok and start is None:
            start = i
        if ok and start is not None:
            out.append(t[i] - t[start])
            start = None
    if start is not None:
        out.append(t[-1] - t[start] + (t[-1] - t[0]) / max(len(t) - 1, 1))
    return out


def resample(t, values, usable):
    """Put a per-frame signal on a uniform FS grid; samples inside gaps longer than MAX_FILL_S are NaN."""
    ok = usable & np.isfinite(values)
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / FS)
    if ok.sum() < 2:
        return grid, np.full(len(grid), np.nan)
    tk, vk = t[ok], values[ok]
    out = np.interp(grid, tk, vk)
    right = np.searchsorted(tk, grid, side="left").clip(1, len(tk) - 1)
    span = tk[right] - tk[right - 1]
    outside = (grid < tk[0] - MAX_FILL_S) | (grid > tk[-1] + MAX_FILL_S)
    out[(span > MAX_FILL_S) & (grid > tk[right - 1]) & (grid < tk[right]) | outside] = np.nan
    return grid, out


def smooth(x):
    """Zero-phase low-pass on each run of finite samples."""
    b, a = butter(4, LOWPASS_HZ / (FS / 2))
    out = np.full(len(x), np.nan)
    finite = np.isfinite(x)
    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    for start, stop in zip(edges[::2], edges[1::2]):
        out[start:stop] = filtfilt(b, a, x[start:stop]) if stop - start > 3 * max(len(a), len(b)) else x[start:stop]
    return out


def crossing_time(grid, y, i_from, i_to, level):
    """Time at which y first crosses level going from index i_from towards i_to."""
    step = 1 if i_to >= i_from else -1
    for i in range(i_from, i_to, step):
        lo, hi = y[i], y[i + step]
        if np.isfinite(lo) and np.isfinite(hi) and (lo - level) * (hi - level) <= 0 and lo != hi:
            return grid[i] + (level - lo) / (hi - lo) * (grid[i + step] - grid[i])
    return np.nan


def repetitions(grid, y):
    """Peaks of the arm-elevation signal, with base levels, amplitude and raise / lower times.

    A raise that the clip cuts off (arm already up at the first frame, or still up at the
    last) has no peak with two flanks. It is returned with "partial" set and no timings.
    """
    reps = []
    finite = np.flatnonzero(np.isfinite(y))
    if len(finite) < FS / 2:
        return reps
    span = robust_range(y)
    filled = np.where(np.isfinite(y), y, np.nanmin(y))
    peaks, _ = find_peaks(filled, prominence=max(20.0, 0.3 * span), distance=int(1.5 * FS))
    bounds = np.r_[0, peaks, len(y) - 1]
    for k, p in enumerate(peaks):
        left = bounds[k] + int(np.nanargmin(filled[bounds[k]:p + 1]))
        right = p + int(np.nanargmin(filled[p:bounds[k + 2] + 1]))
        rise, fall = y[p] - filled[left], y[p] - filled[right]
        up = [crossing_time(grid, y, left, p, filled[left] + f * rise) for f in (0.1, 0.9)]
        down = [crossing_time(grid, y, p, right, filled[right] + f * fall) for f in (0.9, 0.1)]
        reps.append({"t_peak": float(grid[p]), "peak_deg": float(y[p]), "partial": False,
                     "base_left_deg": float(filled[left]), "base_right_deg": float(filled[right]),
                     "amplitude_deg": float(min(rise, fall)),
                     "raise_s": float(up[1] - up[0]), "lower_s": float(down[1] - down[0])})

    low, high = np.nanpercentile(y, 5), np.nanpercentile(y, 95)
    if high - low >= 45.0:
        half = low + 0.5 * (high - low)
        first, last = finite[0], finite[-1]
        head = slice(first, peaks[0] if len(peaks) else last + 1)
        tail = slice(peaks[-1] + 1 if len(peaks) else first, last + 1)
        nan = float("nan")
        for edge, part in ((first, head), (last, tail)):
            if y[edge] > half and (filled[part] < half).any():
                reps.append({"t_peak": float(grid[edge]), "peak_deg": float(np.nanmax(y[part])), "partial": True,
                             "base_left_deg": nan, "base_right_deg": nan, "amplitude_deg": nan, "raise_s": nan, "lower_s": nan})
    return sorted(reps, key=lambda r: r["t_peak"])


def analyse(clip, use_3d=True, side=None):
    """Frame flags, clip quality summary, movement signal, repetitions and the 15 minimum features.

    The moving arm is found from the data unless side ("left" or "right") is given, which
    is how several models are made to describe the same arm.
    """
    t, px = clip["t"], clip["px"]
    own_side, ranges = moving_side(clip, use_3d)
    side = side or own_side
    resting = "left" if side == "right" else "right"
    flags = frame_flags(clip, side, use_3d)
    usable = ~flags.any(axis=1)
    gap_list = gaps(t, usable) if len(t) else []

    elevation = arm_elevation(px, side)
    grid, raw = resample(t, elevation, usable)
    signal = smooth(raw)
    reps = repetitions(grid, signal)
    speed = np.abs(np.gradient(signal, 1.0 / FS))

    _, _, torso = trunk(px)
    torso_len = np.nanmedian(torso[usable]) if usable.any() else np.nan
    shoulder, elbow, wrist = ARM[side]
    rel_wrist = (px[:, wrist] - px[:, shoulder]) / torso_len
    arm_len = np.linalg.norm(rel_wrist, axis=1)

    def pct(x, q):
        x = x[usable & np.isfinite(x)]
        return float(np.percentile(x, q)) if len(x) >= 5 else np.nan

    resting_elev = arm_elevation(px, resting)
    elbow_deg = elbow_angle(px, side)
    incline = trunk_inclination(px)
    median_or_nan = lambda v: float(np.nanmedian(v)) if len(v) and np.isfinite(v).any() else np.nan

    features = {
        "move_elev_p5": pct(elevation, 5), "move_elev_p95": pct(elevation, 95),
        "move_elev_range": pct(elevation, 95) - pct(elevation, 5),
        "rest_elev_p95": pct(resting_elev, 95), "rest_elev_range": pct(resting_elev, 95) - pct(resting_elev, 5),
        "elbow_median": pct(elbow_deg, 50), "elbow_p5": pct(elbow_deg, 5),
        "trunk_incl_median": pct(incline, 50), "trunk_incl_range": pct(incline, 95) - pct(incline, 5),
        "wrist_travel_x": pct(rel_wrist[:, 0], 95) - pct(rel_wrist[:, 0], 5),
        "wrist_travel_y": pct(rel_wrist[:, 1], 95) - pct(rel_wrist[:, 1], 5),
        "arm_len_ratio": pct(arm_len, 5) / pct(arm_len, 95) if pct(arm_len, 95) else np.nan,
        "raise_s_median": median_or_nan(np.array([r["raise_s"] for r in reps])),
        "lower_s_median": median_or_nan(np.array([r["lower_s"] for r in reps])),
        "speed_p95": float(np.nanpercentile(speed, 95)) if np.isfinite(speed).any() else np.nan,
    }
    quality = {
        "frames": len(t), "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "pose_frac": float(clip["has_pose"].mean()) if len(t) else 0.0,
        "usable_frac": float(usable.mean()) if len(t) else 0.0,
        "longest_gap_s": float(max(gap_list)) if gap_list else 0.0,
        "gaps_over_fill": int(sum(g > MAX_FILL_S for g in gap_list)),
        **{f"flag_{name}": float(flags[:, i].mean()) if len(t) else 0.0 for i, name in enumerate(FLAGS)},
        "swap_repaired_frac": float(clip["swapped"].mean()) if len(t) else 0.0,
        "moving_side": side, "range_left": ranges["left"], "range_right": ranges["right"],
        "jitter_rms_deg": float(np.sqrt(np.nanmean((raw - signal) ** 2))) if np.isfinite(signal).any() else np.nan,
        "reps": len(reps), "reps_partial": int(sum(r["partial"] for r in reps)), "torso_px": float(torso_len),
    }
    world = clip.get("world")
    return {"clip": clip, "side": side, "own_side": own_side, "flags": flags, "usable": usable, "elevation": elevation,
            "elevation_3d": arm_elevation(world, side) if world is not None else None,
            "grid": grid, "raw": raw, "signal": signal,
            "reps": reps, "features": features, "quality": quality, "rel_wrist": rel_wrist}
