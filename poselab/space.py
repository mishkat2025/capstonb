"""Measures from a model's 3D points (of the five models, only MediaPipe gives them).

MediaPipe returns its 33 points a second time as "world landmarks": positions in metres
with the middle of the hips as the origin, x towards the image right, y down, and z away
from the camera (a smaller z is nearer the camera). The depth is the network's guess from
one picture. Nothing in these videos measured it, so the numbers below show what the 3D
skeleton says and how steady it is, not whether it is right.

Stage 1 already found two weaknesses (docs/stage1_report.md, Sections 8.4 and 10): the 3D
arm bones change length with the arm's position, and the 3D range of an abduction comes
out smaller from a side view than from the front.
"""
import numpy as np

from . import analysis
from .skeleton import ARM, L_SHOULDER, R_SHOULDER

RAISED = (45.0, 135.0)      # the arm's direction seen from above is only read while its elevation is in this band
ARM_DOWN, ARM_UP = 30.0, 150.0


def unit(v):
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def body_axes(world):
    """The trunk's own directions for points [T, J, 3]: up (hips to shoulders), across (right shoulder to left), forward (where the chest faces)."""
    shoulders, hips, _ = analysis.trunk(world)
    up = unit(shoulders - hips)
    across = world[:, L_SHOULDER] - world[:, R_SHOULDER]
    across = unit(across - (across * up).sum(axis=-1, keepdims=True) * up)
    return up, across, np.cross(across, up)


def forward_angle(world, side):
    """Direction of the upper arm seen from above, relative to the trunk: 0 straight out to the side, 90 straight ahead, negative behind the body.

    Biomechanics calls this the plane of elevation. It separates raising the arm sideways
    (abduction) from raising it forwards (flexion), which the elevation angle cannot. It
    means little while the arm hangs or points overhead, where the arm has almost no
    sideways or forward extent to take a direction from.
    """
    shoulder, elbow, _ = ARM[side]
    _, across, forward = body_axes(world)
    arm = world[:, elbow] - world[:, shoulder]
    outward = (arm * across).sum(axis=-1) * (1 if side == "left" else -1)
    return np.degrees(np.arctan2((arm * forward).sum(axis=-1), outward))


def limb_lengths(world, side):
    """Upper-arm and forearm length in metres, per frame."""
    shoulder, elbow, wrist = ARM[side]
    return np.linalg.norm(world[:, elbow] - world[:, shoulder], axis=-1), np.linalg.norm(world[:, wrist] - world[:, elbow], axis=-1)


def measure(a):
    """3D measures for one analysed clip (the dict analysis.analyse returns) whose model gave 3D points.

    Uses the same arm and the same usable frames as the 2D analysis, so that the two can
    be compared frame by frame.
    """
    clip, side, usable = a["clip"], a["side"], a["usable"]
    world, flat, deep = clip["world"], a["elevation"], a["elevation_3d"]
    grid, raw = analysis.resample(clip["t"], deep, usable)
    signal = analysis.smooth(raw)
    reps = analysis.repetitions(grid, signal)
    direction = forward_angle(world, side)
    with np.errstate(invalid="ignore"):
        raised = usable & (deep >= RAISED[0]) & (deep <= RAISED[1])
        down, up = usable & (deep < ARM_DOWN), usable & (deep > ARM_UP)
    upper, fore = limb_lengths(world, side)

    def pct(x, q, mask=usable):
        x = x[mask & np.isfinite(x)]
        return float(np.percentile(x, q)) if len(x) >= 5 else np.nan

    difference = np.abs(flat - deep)
    row = {"arm": side, "frames_usable": int(usable.sum()),
           "elevation_2d_low_deg": pct(flat, 5), "elevation_2d_high_deg": pct(flat, 95), "elevation_2d_range_deg": pct(flat, 95) - pct(flat, 5),
           "elevation_3d_low_deg": pct(deep, 5), "elevation_3d_high_deg": pct(deep, 95), "elevation_3d_range_deg": pct(deep, 95) - pct(deep, 5),
           "difference_median_deg": pct(difference, 50), "difference_p95_deg": pct(difference, 95),
           "repetitions_2d": a["quality"]["reps"] - a["quality"]["reps_partial"], "repetitions_2d_cut_off": a["quality"]["reps_partial"],
           "repetitions_3d": sum(not r["partial"] for r in reps), "repetitions_3d_cut_off": sum(r["partial"] for r in reps),
           "jitter_2d_rms_deg": a["quality"]["jitter_rms_deg"],
           "jitter_3d_rms_deg": float(np.sqrt(np.nanmean((raw - signal) ** 2))) if np.isfinite(signal).any() else np.nan,
           "frames_arm_raised": int(raised.sum()),
           "forward_angle_median_deg": pct(direction, 50, raised), "forward_angle_p25_deg": pct(direction, 25, raised),
           "forward_angle_p75_deg": pct(direction, 75, raised)}
    for name, length in (("upper_arm", upper), ("forearm", fore)):
        row[f"{name}_cm_median"] = 100 * pct(length, 50)
        row[f"{name}_cm_arm_down"] = 100 * pct(length, 50, down)
        row[f"{name}_cm_arm_up"] = 100 * pct(length, 50, up)
        row[f"{name}_spread_pct"] = 100 * (pct(length, 95) - pct(length, 5)) / pct(length, 50)
    return {"row": row, "grid": grid, "signal": signal, "reps": reps, "direction": np.where(raised, direction, np.nan),
            "upper_cm": np.where(usable, 100 * upper, np.nan), "fore_cm": np.where(usable, 100 * fore, np.nan)}
