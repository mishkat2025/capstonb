"""Figures, tables and the HTML report for one video run through several pose models.

build() writes into one folder:
    report.html            open this
    figures/*.png, frames.jpg
    summary.csv            one row per model
    agreement.csv          one row per pair of models
    signals.csv            the arm-elevation signal of every model on a common 30 Hz time grid
    space.csv, space_signals.csv, space.jpg      the 3D measures, when a selected model gives 3D points (see space.py)
    pose/<model>.npz       the raw points (format in store.py)

There is no ground truth in the dataset, so the report never says which model is right.
It shows how the models differ from each other, how often each one gives usable frames,
and how fast each one ran.
"""
import csv
import html
from datetime import datetime
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from . import analysis, space, store
from .draw import SERIES_HEX, SPACE_LINE, skeleton, space_tile, text
from .estimators import REGISTRY
from .skeleton import BODY, J

SURFACE, INK, INK2, MUTED, GRID, AXIS, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, BLUE_LIGHT = "#2a78d6", "#9ec5f4"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SEQUENTIAL.set_bad(NEUTRAL)
COLOUR = {key: SERIES_HEX[i] for i, key in enumerate(REGISTRY)}       # colour follows the model, whichever models are selected

FLAG_TEXT = {
    "no_pose": ("Not found", "The model did not find the participant in the frame."),
    "low_confidence": ("Low confidence", "A shoulder, a hip, or the moving arm's elbow or wrist is missing or scored below the model's own threshold."),
    "out_of_frame": ("Out of frame", "One of those points lies outside the picture."),
    "scale_jump": ("Body size jump", "The torso is more than 20% longer or shorter than its usual length in this video: usually another person."),
    "teleport": ("Position jump", "The hips moved more than half a torso length since the previous frame."),
}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 150,
    })


def save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- numbers

def analyse_run(run, view):
    """Per model: the clip in shared joints and its analysis, all describing the same arm."""
    clips = {}
    for key, track in run.tracks.items():
        record = {"kp": track.kp, "names": track.names, "t_ms": run.t_ms, "fps": run.fps, "min_score": track.min_score,
                  "width": run.width, "height": run.height}
        if track.world is not None:
            record["world"] = track.world       # kept for the 3D section; the checks below do not use it
        clips[key] = store.to_clip(record, front_view=view == "front")
    votes = [analysis.moving_side(clip, use_3d=False)[0] for clip in clips.values() if clip["has_pose"].any()]
    side = max(("right", "left"), key=votes.count) if votes else "right"
    # use_3d=False: only MediaPipe has 3D points, so the 3D limb-length check is left out for every model
    return side, {key: analysis.analyse(clip, use_3d=False, side=side) for key, clip in clips.items()}


def summary_rows(run, analyses):
    rows = []
    for key, track in run.tracks.items():
        a = analyses[key]
        q, f = a["quality"], a["features"]
        ms = track.ms
        row = {
            "model": key, "label": track.label, "runs_on": track.runs_on, "points": len(track.names),
            "foot_points": "yes" if np.isfinite(a["clip"]["px"][:, J["left_heel"]]).any() else "no",
            "frames": q["frames"], "found_frac": q["pose_frac"], "usable_frac": q["usable_frac"], "longest_gap_s": q["longest_gap_s"],
            **{f"flag_{name}": q[f"flag_{name}"] for name in FLAG_TEXT},
            "relabelled_frac": q["swap_repaired_frac"], "max_people": int(track.n_people.max()) if len(track.n_people) else 0,
            "arm": a["side"], "own_choice_of_arm": a["own_side"],
            "repetitions": q["reps"] - q["reps_partial"], "repetitions_cut_off": q["reps_partial"],
            "elevation_low_deg": f["move_elev_p5"], "elevation_high_deg": f["move_elev_p95"], "elevation_range_deg": f["move_elev_range"],
            "raise_s_median": f["raise_s_median"], "lower_s_median": f["lower_s_median"],
            "elbow_median_deg": f["elbow_median"], "trunk_inclination_median_deg": f["trunk_incl_median"],
            "jitter_rms_deg": q["jitter_rms_deg"], "torso_px": q["torso_px"],
            "ms_median": float(np.median(ms)) if len(ms) else np.nan, "ms_p95": float(np.percentile(ms, 95)) if len(ms) else np.nan,
            **{f"ms_{part}_median": float(np.median(v)) for part, v in track.parts_ms.items()},
        }
        row["frames_per_s"] = 1000.0 / row["ms_median"] if row["ms_median"] else np.nan
        rows.append(row)
    return rows


def agreement(analyses, side):
    """For each pair of models: difference in arm elevation and distance between the same joints."""
    torso = np.nanmedian([a["quality"]["torso_px"] for a in analyses.values()])
    rows = []
    for a_key, b_key in combinations(analyses, 2):
        a, b = analyses[a_key], analyses[b_key]
        both = a["usable"] & b["usable"]
        diff = np.abs(a["elevation"] - b["elevation"])[both]
        diff = diff[np.isfinite(diff)]
        distance = np.linalg.norm(a["clip"]["px"][:, BODY] - b["clip"]["px"][:, BODY], axis=-1)[both]
        distance = distance[np.isfinite(distance)]
        rows.append({"model_a": a_key, "model_b": b_key, "frames_both_usable": int(both.sum()),
                     "elevation_diff_median_deg": float(np.median(diff)) if len(diff) else np.nan,
                     "elevation_diff_p95_deg": float(np.percentile(diff, 95)) if len(diff) else np.nan,
                     "joint_distance_median_px": float(np.median(distance)) if len(distance) else np.nan,
                     "joint_distance_median_pct_torso": float(100 * np.median(distance) / torso) if len(distance) else np.nan})
    return rows


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------- figures

def figure_elevation(analyses, labels, side, t0, path):
    keys = list(analyses)
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 1.5 * len(keys) + 1.0), sharex=True, sharey=True, squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        a = analyses[key]
        t = a["clip"]["t"] - t0
        ax.fill_between(t, 0, 185, where=~a["usable"], color=NEUTRAL, step="mid", linewidth=0)
        for other in keys:
            if other != key:
                ax.plot(analyses[other]["grid"] - t0, analyses[other]["signal"], color=AXIS, linewidth=1)
        ax.plot(a["grid"] - t0, a["signal"], color=COLOUR[key])
        ax.plot([r["t_peak"] - t0 for r in a["reps"]], [r["peak_deg"] for r in a["reps"]], "o", color=INK, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        q = a["quality"]
        full, cut = q["reps"] - q["reps_partial"], q["reps_partial"]
        title = f"{labels[key]}   ·   {full} repetition{'s' if full != 1 else ''}"
        if cut:
            title += f" + {cut} cut off by the video"
        if np.isfinite(a["features"]["move_elev_range"]):
            title += f"   ·   range {a['features']['move_elev_range']:.0f}°"
        ax.set_title(title)
        ax.set_ylim(0, 185)
        ax.set_yticks([0, 90, 180])
    axes[-1, 0].set_xlabel("Time in video (s)")
    fig.supylabel(f"Elevation of the {side} arm (°)", fontsize=9, color=INK2)
    handles = [plt.Line2D([], [], color=INK2, label="This model (in its own colour)"),
               plt.Line2D([], [], color=AXIS, linewidth=1, label="The other models"),
               plt.Line2D([], [], color=INK, marker="o", markersize=5, linestyle="", label="Top of a repetition"),
               Patch(facecolor=NEUTRAL, label="Frame not usable for this model")]
    fig.legend(handles=handles, loc="upper right", ncols=4, fontsize=8, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0.01, 0, 1, 0.97))
    save(fig, path)


def figure_agreement(pairs, keys, labels, path):
    index = {key: i for i, key in enumerate(keys)}
    panels = [("elevation_diff_median_deg", "Arm elevation: median difference (°)", "{:.1f}"),
              ("joint_distance_median_pct_torso", "Same joints: median distance (% of torso length)", "{:.1f}")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.62 * len(keys) + 1.9))
    for ax, (field, title, fmt) in zip(axes, panels):
        table = np.full((len(keys), len(keys)), np.nan)
        for row in pairs:
            i, j = index[row["model_a"]], index[row["model_b"]]
            table[i, j] = table[j, i] = row[field]
        top = np.nanmax(table) if np.isfinite(table).any() else 1.0
        ax.imshow(np.ma.masked_invalid(table), cmap=SEQUENTIAL, vmin=0, vmax=max(top, 1e-6), aspect="auto")
        for i in range(len(keys)):
            for j in range(len(keys)):
                if np.isfinite(table[i, j]):
                    ax.text(j, i, fmt.format(table[i, j]), ha="center", va="center", fontsize=9,
                            color="white" if table[i, j] > 0.55 * top else INK)
        short = [labels[k].split()[0] for k in keys]
        ax.set_xticks(range(len(keys)), short)
        ax.set_yticks(range(len(keys)), [labels[k] for k in keys])
        ax.set_title(title)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(keys)), minor=True)       # thin gaps between cells
        ax.set_yticks(np.arange(-0.5, len(keys)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    save(fig, path)


def figure_quality(analyses, labels, t0, path):
    keys = list(analyses)
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(keys) + 1.2))
    to_rgb = matplotlib.colors.to_rgb
    for i, key in enumerate(keys):
        a = analyses[key]
        t = a["clip"]["t"] - t0
        state = np.where(a["usable"], 0, np.where(a["clip"]["has_pose"], 1, 2))
        row = np.array([to_rgb(COLOUR[key]), to_rgb(MUTED), to_rgb(GRID)])[state][None]
        ax.imshow(row, extent=(t[0], t[-1] if len(t) > 1 else 1, i + 0.34, i - 0.34), aspect="auto", interpolation="nearest")
        ax.text(1.01, i, f"{a['quality']['usable_frac']:.0%} usable", transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=INK2)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_yticks(range(len(keys)), [labels[k] for k in keys])
    ax.set_xlabel("Time in video (s)")
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [Patch(facecolor=INK2, label="Usable frame (shown in the model's colour)"),
               Patch(facecolor=MUTED, label="Participant found, but a quality check failed"),
               Patch(facecolor=GRID, label="Participant not found")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02), ncols=3, fontsize=8)
    fig.tight_layout()
    save(fig, path)


def figure_speed(rows, path):
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.2))
    ax.set_axisbelow(True)
    longest = max(r["ms_median"] for r in rows)
    for i, r in enumerate(rows):
        detector, pose = r.get("ms_detector_median"), r.get("ms_pose_median")
        if detector is not None:
            ax.barh(i, pose, height=0.44, color=BLUE)
            ax.barh(i, detector - 0.004 * longest, left=pose + 0.004 * longest, height=0.44, color=BLUE_LIGHT)
            total = pose + detector
        else:
            total = r["ms_median"]
            ax.barh(i, total, height=0.44, color=BLUE)
        ax.text(total + 0.012 * longest, i, f"{total:.0f} ms   ·   {1000 / total:.0f} frames/s", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), [f"{r['label']} ({r['runs_on']})" for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, longest * 1.28)
    ax.set_xlabel("Median time per frame (ms)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=BLUE, label="Pose model"), Patch(facecolor=BLUE_LIGHT, label="Person detector (run once per frame, shared by the top-down models)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02), ncols=2, fontsize=8)
    fig.tight_layout()
    save(fig, path)


def figure_space(a, m, colour, t0, path):
    """For the model with 3D points: elevation from the picture and from 3D, the raised arm's direction, and the 3D bone lengths."""
    t = a["clip"]["t"] - t0
    fig, (top, mid, low) = plt.subplots(3, 1, figsize=(10, 6.8), sharex=True, gridspec_kw={"height_ratios": [1.5, 1, 1]})
    top.fill_between(t, 0, 185, where=~a["usable"], color=NEUTRAL, step="mid", linewidth=0)
    top.plot(a["grid"] - t0, a["signal"], color=colour, label="Measured in the picture (2D)")
    top.plot(m["grid"] - t0, m["signal"], color=INK, linewidth=1.5, label="From the 3D points")
    top.set_ylim(0, 185)
    top.set_yticks([0, 90, 180])
    top.set_ylabel("Arm elevation (°)")
    top.set_title(f"Elevation of the {a['side']} arm, measured two ways")
    top.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncols=2, fontsize=8)

    for level in (0, 90):
        mid.axhline(level, color=AXIS, linewidth=1)
    mid.plot(t, m["direction"], color=INK, linewidth=1.5)
    mid.set_ylim(-60, 120)
    mid.set_yticks([-45, 0, 45, 90], ["−45°", "to the side  0°", "45°", "to the front  90°"])
    mid.set_title(f"Which way the raised arm points, seen from above (read only while it is between {space.RAISED[0]:.0f}° and {space.RAISED[1]:.0f}°)")

    low.plot(t, m["upper_cm"], color=INK, linewidth=1.5, label="Upper arm")
    low.plot(t, m["fore_cm"], color=MUTED, linewidth=1.5, label="Forearm")
    longest = np.nanmax(np.r_[m["upper_cm"], m["fore_cm"], 1.0])
    low.set_ylim(0, 1.2 * longest)
    low.set_ylabel("Length (cm)")
    low.set_title("Length of the arm bones in the 3D skeleton (a real bone keeps its length)")
    low.legend(loc="lower right", ncols=2, fontsize=8)
    low.set_xlabel("Time in video (s)")
    fig.tight_layout()
    save(fig, path)


def space_sheet(samples, a, height, path, n_frames=6):
    """Top row: frames spread over the video with the model's skeleton. Bottom row: its 3D points in the same frames, from the side and from above."""
    if not samples:
        return False
    chosen = [samples[int((k + 0.5) * len(samples) / n_frames)] for k in range(min(n_frames, len(samples)))]
    clip = a["clip"]
    top, bottom = [], []
    for index, small in chosen:
        tile = small.copy()
        skeleton(tile, clip["px"][index], clip["score"][index], clip["min_score"], tile.shape[0] / height)
        top.append(tile)
        tile = space_tile(clip["world"][index], clip["score"][index], clip["min_score"], tile.shape[1], tile.shape[0])
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), SPACE_LINE, 1)
        bottom.append(tile)
    cv2.imwrite(str(path), np.vstack([np.hstack(top), np.hstack(bottom)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


HAND_STATES = ("open", "partly closed", "closed")
HAND_STATE_COLOUR = {"open": "#9ec5f4", "partly closed": "#3987e5", "closed": "#104281"}     # one hue, light to dark
HAND_COLOUR = {"left": "#2a78d6", "right": "#eb6834"}                                        # as in the skeleton: left blue, right orange


def hand_rows(run):
    """Per hand: how often it was found, and how the found frames split into open, partly closed and closed."""
    from .hands import CLOSED_BELOW, OPEN_FROM, steady
    rows = []
    for n, side in enumerate(("left", "right")):
        value = steady(run.hands.extension[:, n])
        found = np.isfinite(value)
        v = value[found]
        share = lambda mask: float(mask.mean()) if len(v) else np.nan
        rows.append({"hand": side, "frames": len(value), "found_frac": float(found.mean()) if len(value) else 0.0,
                     "open_frac": share(v >= OPEN_FROM), "partly_closed_frac": share((v < OPEN_FROM) & (v >= CLOSED_BELOW)),
                     "closed_frac": share(v < CLOSED_BELOW),
                     "extension_median": float(np.median(v)) if len(v) else np.nan,
                     "extension_lowest_tenth": float(np.percentile(v, 10)) if len(v) else np.nan,
                     "ms_both_hands_median": float(np.median(run.hands.ms)) if len(run.hands.ms) else np.nan})
    return rows


def figure_hands(run, t0, path):
    from .hands import CLOSED_BELOW, OPEN_FROM, steady
    t = run.t_ms / 1000.0 - t0
    values = [steady(run.hands.extension[:, n]) for n in range(2)]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True, gridspec_kw={"height_ratios": [2.6, 1]})
    for n, side in enumerate(("left", "right")):
        top.plot(t, values[n], color=HAND_COLOUR[side], linewidth=1.5, label=f"{side.capitalize()} hand")
    for level in (OPEN_FROM, CLOSED_BELOW):
        top.axhline(level, color=AXIS, linewidth=1)
    top.set_ylim(0.2, 1.04)
    top.set_yticks([0.2, 0.4, CLOSED_BELOW, OPEN_FROM, 1.0], ["0.2", "0.4", f"closed below {CLOSED_BELOW:.2f}", f"open from {OPEN_FROM:.2f}", "1.0"])
    top.set_ylabel("Finger extension")
    top.set_title("How straight the fingers are: about 1 for a flat hand, low for a fist. Gaps: hand not found")
    top.legend(loc="lower left", ncols=2, fontsize=8)

    to_rgb = matplotlib.colors.to_rgb
    palette = np.array([to_rgb(HAND_STATE_COLOUR[s]) for s in HAND_STATES] + [to_rgb(GRID)])
    for n, side in enumerate(("left", "right")):
        value = values[n]
        state = np.where(~np.isfinite(value), 3, np.where(value >= OPEN_FROM, 0, np.where(value >= CLOSED_BELOW, 1, 2)))
        bottom.imshow(palette[state][None], extent=(t[0], t[-1] if len(t) > 1 else 1, n + 0.34, n - 0.34), aspect="auto", interpolation="nearest")
    bottom.set_ylim(1.5, -0.5)
    bottom.set_yticks([0, 1], ["Left hand", "Right hand"])
    bottom.set_xlabel("Time in video (s)")
    bottom.grid(False)
    bottom.tick_params(axis="y", length=0)
    bottom.spines["left"].set_visible(False)
    handles = [Patch(facecolor=HAND_STATE_COLOUR[s], label=s.capitalize()) for s in HAND_STATES] + [Patch(facecolor=GRID, label="Hand not found")]
    bottom.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.55), ncols=4, fontsize=8)
    fig.tight_layout()
    save(fig, path)


def hand_sheet(hand_samples, path, n_frames=6, size=240):
    """Two rows (right hand, left hand) of enlarged cut-outs from frames spread over the video."""
    from .draw import hand_tile
    if not hand_samples:
        return False
    chosen = [hand_samples[int((k + 0.5) * len(hand_samples) / n_frames)] for k in range(min(n_frames, len(hand_samples)))]
    rows = [np.hstack([hand_tile(hands.get(side), size, f"{side} hand") for _, hands in chosen]) for side in ("right", "left")]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return True


def overlay_sheet(samples, analyses, labels, height, path, n_frames=6):
    """Rows: models. Columns: frames spread over the video, with each model's skeleton drawn on them."""
    if not samples:
        return False
    chosen = [samples[int((k + 0.5) * len(samples) / n_frames)] for k in range(min(n_frames, len(samples)))]
    rows = []
    for key, a in analyses.items():
        clip = a["clip"]
        tiles = []
        for index, small in chosen:
            tile = small.copy()
            scale = tile.shape[0] / height
            skeleton(tile, clip["px"][index], clip["score"][index], clip["min_score"], scale)
            fired = [FLAG_TEXT[name][0] for name, on in zip(analysis.FLAGS, a["flags"][index]) if on and name in FLAG_TEXT]
            if fired:
                cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (40, 40, 220), 4)
                text(tile, ", ".join(fired)[:38], (6, tile.shape[0] - 10), 0.45, (40, 40, 220))
            if clip["swapped"][index]:
                text(tile, "left/right relabelled", (6, tile.shape[0] - 28), 0.45, (160, 90, 20))
            tiles.append(tile)
        row = np.hstack(tiles)
        cv2.rectangle(row, (0, 0), (14 + 10 * len(labels[key]), 26), (0, 0, 0), -1)
        text(row, labels[key], (6, 18), 0.55, (255, 255, 255))
        rows.append(row)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


# ---------------------------------------------------------------- HTML

CSS = """
:root { --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1080px; margin:0 auto; padding:32px 20px 64px; }
h1 { font-size:26px; margin:0 0 4px; } h2 { font-size:19px; margin:44px 0 8px; } h3 { font-size:15px; margin:20px 0 4px; }
p { margin:8px 0; max-width:76ch; } .lede { color:var(--ink2); }
.card { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin:14px 0; }
.note { border-left:4px solid #2a78d6; }
img, video { max-width:100%; height:auto; display:block; border-radius:6px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; } th { color:var(--ink2); font-weight:600; vertical-align:bottom; }
td.wrap, th.wrap { white-space:normal; text-align:left; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:7px; }
.facts { display:flex; flex-wrap:wrap; gap:8px 28px; color:var(--ink2); font-size:14px; margin:10px 0 0; }
.facts b { color:var(--ink); font-weight:600; }
code { background:#f0efec; padding:1px 5px; border-radius:4px; font-size:13px; }
ul { margin:8px 0; padding-left:22px; max-width:76ch; } li { margin:4px 0; }
"""


def num(value, fmt="{:.0f}", missing="–"):
    return fmt.format(value) if value is not None and np.isfinite(value) else missing


def table(headers, rows, wrap=()):
    head = "".join(f"<th{' class=wrap' if i in wrap else ''}>{h}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td{' class=wrap' if i in wrap else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def name_cell(row):
    return f'<span class="dot" style="background:{COLOUR[row["model"]]}"></span>{html.escape(row["label"])}'


def space_html(label, s, has_sheet):
    """The report section for the model with 3D points. s: the row from space.measure()."""
    deg = lambda v, fmt="{:.0f}°": num(v, fmt)
    cm = lambda v: num(v, "{:.1f} cm")
    count = lambda full, cut: f"{full}" + (f" + {cut} cut off" if cut else "")
    name = html.escape(label)
    parts = [f"""<h2>{name} in 3D</h2>
<p>{name} returns its points a second time as positions in space: metres to the left or right of, above or below, and nearer or farther than the middle of the hips. Of the five models in this program it is the only one that does. In the live view the 3D skeleton has a panel of its own.</p>
<div class="card note"><b>The depth is an estimate, not a measurement.</b> The video comes from one ordinary camera, which records no distance. The model fills in the depth, and the body's size in metres, from what bodies usually look like. Nothing in the dataset can check it. Stage 1 found these 3D points less steady than the 2D ones; the bottom chart below shows how steady they are in this video.</div>"""]
    if has_sheet:
        parts.append("""<div class="card"><img src="space.jpg" alt="Sample frames and the 3D skeleton of each, from the side and from above"></div>
<p><b>How to read it.</b> Top row: six frames with the skeleton as it is drawn elsewhere in this report. Bottom row: the 3D points of the same frames, drawn from two directions the camera does not have. <b>From the side</b>: the camera is on the left, so a point further left is nearer the camera. <b>From above</b>: the camera is at the bottom, left and right are as in the video, and the legs are left out so the arms stay readable. The grey lines pass through the middle of the hips. The circle is the head and the dot is the nose. Right side orange, left side blue.</p>""")
    parts.append(f"""<div class="card"><img src="figures/space.png" alt="Arm elevation in 2D and 3D, direction of the raised arm, and 3D bone lengths over time"></div>
<p><b>How to read it.</b></p><ul>
<li><b>Top:</b> the arm's elevation measured in the picture, which is the measure used everywhere else in this report, and the same angle computed from the 3D points. When a sideways raise is filmed from the front, the arm stays flat in the picture and the two should agree. When the arm points towards or away from the camera, the picture shortens it and distorts the angle; the 3D angle would then be the better one, but only if the estimated depth is good.</li>
<li><b>Middle:</b> which way the upper arm points when seen from above, relative to the chest: 0° is straight out to the side (abduction), 90° is straight ahead (flexion). The picture alone cannot give this. It is read only while the arm is between {space.RAISED[0]:.0f}° and {space.RAISED[1]:.0f}°, because an arm that hangs or points overhead has almost no sideways or forward direction.</li>
<li><b>Bottom:</b> the length of the upper arm and the forearm in the 3D skeleton. A real bone keeps its length, so every rise and fall of these lines is error in the 3D estimate.</li></ul>""")
    parts.append(table(["Elevation of the " + s["arm"] + " arm", "Lowest", "Highest", "Range", "Repetitions", "Jitter"],
                       [["Measured in the picture (2D)", deg(s["elevation_2d_low_deg"]), deg(s["elevation_2d_high_deg"]), deg(s["elevation_2d_range_deg"]),
                         count(s["repetitions_2d"], s["repetitions_2d_cut_off"]), deg(s["jitter_2d_rms_deg"], "{:.2f}°")],
                        ["From the 3D points", deg(s["elevation_3d_low_deg"]), deg(s["elevation_3d_high_deg"]), deg(s["elevation_3d_range_deg"]),
                         count(s["repetitions_3d"], s["repetitions_3d_cut_off"]), deg(s["jitter_3d_rms_deg"], "{:.2f}°")]]))
    parts.append(f"""<ul><li>Frame by frame, the two angles differ by a median of {deg(s["difference_median_deg"], "{:.1f}°")} (95th percentile {deg(s["difference_p95_deg"], "{:.1f}°")}) over the {s["frames_usable"]} usable frames.</li>
<li><b>Direction of the raised arm:</b> median {deg(s["forward_angle_median_deg"])}, middle half of the frames {deg(s["forward_angle_p25_deg"])} to {deg(s["forward_angle_p75_deg"])}, over the {s["frames_arm_raised"]} frames with the arm between {space.RAISED[0]:.0f}° and {space.RAISED[1]:.0f}°.</li></ul>""")
    parts.append(table(["Arm bone in the 3D skeleton", "Median length", f"Arm down (below {space.ARM_DOWN:.0f}°)", f"Arm overhead (above {space.ARM_UP:.0f}°)", "Spread"],
                       [[title, cm(s[f"{bone}_cm_median"]), cm(s[f"{bone}_cm_arm_down"]), cm(s[f"{bone}_cm_arm_up"]), num(s[f"{bone}_spread_pct"], "{:.0f}%")]
                        for title, bone in (("Upper arm", "upper_arm"), ("Forearm", "forearm"))]))
    parts.append("""<p><b>Spread</b> is the distance between the 5th and the 95th percentile of the length, as a share of the median. For a rigid bone it should be near zero. The lengths are in the model's own metres, which are a guess like the depth, so they are not the participant's real arm lengths.</p>""")
    return "\n".join(parts)


def write_html(path, run, view, side, rows, pairs, tracks, has_sheet, live_video, codec, hands=None, has_hand_sheet=False, space_section=None):
    labels = {r["model"]: r["label"] for r in rows}
    video = Path(run.video)
    duration = (run.t_ms[-1] - run.t_ms[0]) / 1000 if len(run.t_ms) > 1 else 0.0
    view_text = {"front": "front (left/right labels are checked and repaired)", "side": "side (left/right labels are taken as the model gives them)"}[view]
    relabelled = [r for r in rows if r["relabelled_frac"] > 0]
    other_arm = [r for r in rows if r["own_choice_of_arm"] != r["arm"]]

    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pose comparison: {html.escape(video.stem)}</title><style>{CSS}</style></head><body><main>
<h1>Pose model comparison</h1>
<p class="lede">{html.escape(video.name)} · run on {datetime.now():%d %B %Y, %H:%M}</p>
<div class="facts"><span><b>{len(run.t_ms)}</b> frames</span><span><b>{duration:.1f} s</b></span><span><b>{run.width} × {run.height}</b> pixels</span>
<span><b>{run.fps:.1f}</b> frames/s in the file</span><span>view: <b>{view_text}</b></span><span><b>{len(rows)}</b> model{'s' if len(rows) != 1 else ''}</span></div>"""]
    if run.stopped_early:
        parts.append('<div class="card note"><b>The run was stopped before the end of the video.</b> Everything below covers only the frames that were processed.</div>')

    parts.append("""<div class="card note"><b>Read this first.</b><ul>
<li><b>Nobody measured the true joint positions in these videos.</b> The report shows how far the models differ from each other. When two models agree, both may still be wrong in the same way; when they differ, the report cannot say which one is right.</li>
<li><b>Confidence scores are not comparable between models.</b> Each model's "confident" threshold is its own tool's usual value, so "usable frames" is judged a little differently for each.</li>
<li><b>The movement measure is arm elevation, chosen for E01 Abduction.</b> For other exercises the agreement, quality and speed sections still apply; the repetition count may not mean anything.</li>
<li><b>Nothing here says whether the exercise was done correctly.</b></li></ul></div>""")

    if live_video:
        parts.append("<h2>The live view, recorded</h2>")
        if codec == "H.264":
            parts.append(f'<div class="card"><video controls preload="metadata" src="{live_video}"></video></div>')
        else:
            parts.append(f'<p>The recording is in <a href="{live_video}">{live_video}</a>. Open it in a video player; this format does not play inside a browser.</p>')

    # ---- summary
    parts.append(f"<h2>Summary</h2><p>All models describe the participant's <b>{side} arm</b>, the arm most models saw moving more.</p>")
    parts.append(table(
        ["Model", "Found", "Usable frames", "Longest gap", "Repetitions", "Lowest", "Highest", "Range", "Raise time", "Jitter", "Time per frame"],
        [[name_cell(r), f"{r['found_frac']:.0%}", f"{r['usable_frac']:.0%}", f"{r['longest_gap_s']:.1f} s",
          f"{r['repetitions']}" + (f" + {r['repetitions_cut_off']} cut off" if r["repetitions_cut_off"] else ""),
          num(r["elevation_low_deg"], "{:.0f}°"), num(r["elevation_high_deg"], "{:.0f}°"), num(r["elevation_range_deg"], "{:.0f}°"),
          num(r["raise_s_median"], "{:.2f} s"), num(r["jitter_rms_deg"], "{:.2f}°"), f"{r['ms_median']:.0f} ms"] for r in rows]))
    parts.append("""<ul><li><b>Found</b>: frames in which the model located the participant. <b>Usable</b>: found, and every quality check passed.</li>
<li><b>Lowest / Highest</b>: the 5th and 95th percentile of the arm's elevation over the usable frames (0° hanging, 90° horizontal, 180° overhead). <b>Range</b> is the difference.</li>
<li><b>Raise time</b>: median time to go from 10% to 90% of a repetition's rise.</li>
<li><b>Jitter</b>: how much the frame-by-frame angle wobbles around its smoothed curve (root mean square). Lower means a steadier estimate, not a more accurate one.</li></ul>""")
    if other_arm:
        names = ", ".join(html.escape(r["label"]) for r in other_arm)
        parts.append(f'<p>On its own, {names} would have picked the other arm as the moving one. It is shown for the {side} arm so that all models describe the same thing.</p>')

    # ---- elevation
    parts.append("""<h2>The movement each model sees</h2><div class="card"><img src="figures/elevation.png" alt="Arm elevation over time, one panel per model"></div>
<p><b>How to read it.</b> Each panel is one model. The coloured line is that model's arm elevation after smoothing; the thin grey lines are the other models, so a coloured line that leaves the grey bundle is where this model disagrees with the rest. Black dots are the tops of the repetitions it detected. Grey bands are frames the model could not use.</p>""")

    # ---- agreement
    if pairs:
        parts.append("""<h2>How closely the models agree</h2><div class="card"><img src="figures/agreement.png" alt="Agreement between each pair of models"></div>
<p><b>How to read it.</b> Each cell compares two models on the frames that were usable for both. Left: the typical difference between their arm-elevation angles. Right: the typical distance between the same joint in the two models, over the twelve joints all models report (shoulders, elbows, wrists, hips, knees, ankles), as a share of the participant's torso length so that it does not depend on how large the person is in the picture. Lighter cells mean closer agreement.</p>
<p>A small steady distance is expected even between good models: they were trained on datasets that mark, for example, the hip in slightly different places.</p>""")
        parts.append(table(["Pair", "Frames usable for both", "Elevation difference, median", "95th percentile", "Joint distance, median", "As share of torso"],
                           [[f"{html.escape(labels[p['model_a']])} and {html.escape(labels[p['model_b']])}", p["frames_both_usable"],
                             num(p["elevation_diff_median_deg"], "{:.1f}°"), num(p["elevation_diff_p95_deg"], "{:.1f}°"),
                             num(p["joint_distance_median_px"], "{:.1f} px"), num(p["joint_distance_median_pct_torso"], "{:.1f}%")] for p in pairs]))

    # ---- quality
    parts.append("""<h2>Which frames were usable</h2><div class="card"><img src="figures/quality.png" alt="Usable frames over time for each model"></div>
<p><b>How to read it.</b> One row per model, time running left to right. A frame is usable when the participant was found and none of the checks below fired. The checks are the same for every model.</p>""")
    parts.append(table(["Model"] + [FLAG_TEXT[name][0] for name in FLAG_TEXT] + ["Most people seen in a frame"],
                       [[name_cell(r)] + [f"{r[f'flag_{name}']:.1%}" for name in FLAG_TEXT] + [r["max_people"]] for r in rows]))
    parts.append("<ul>" + "".join(f"<li><b>{title}</b>: {what}</li>" for title, what in FLAG_TEXT.values()) + "</ul>")
    parts.append("<p>MediaPipe also gives 3D points, and Stage 1 used them for a limb-length check. That check is left out here because the other models have no 3D points and every model should face the same checks. What the 3D points show is in their own section further down.</p>")
    if relabelled:
        listing = ", ".join(f"{html.escape(r['label'])} {r['relabelled_frac']:.1%}" for r in relabelled)
        parts.append(f"<p><b>Left/right relabelling.</b> In a front view a model sometimes calls the left side right. Those frames were repaired, not discarded: {listing} of frames.</p>")

    # ---- speed
    parts.append("""<h2>Speed on this computer</h2><div class="card"><img src="figures/speed.png" alt="Median time per frame for each model"></div>
<p><b>How to read it.</b> Median time one frame takes inside each model. A top-down model (ViTPose, RTMPose) needs a person detector first; its time is shown as the lighter segment and counted for each top-down model, although it runs only once per frame. Times were measured during this run, with all selected models loaded together and the live view being drawn, so a model run on its own may be faster. They say nothing about speed on a phone.</p>""")
    parts.append(table(["Model", "Runs on", "Median", "95th percentile", "Frames per second", "Person detector", "Pose model"],
                       [[name_cell(r), r["runs_on"], f"{r['ms_median']:.1f} ms", f"{r['ms_p95']:.1f} ms", f"{r['frames_per_s']:.0f}",
                         num(r.get("ms_detector_median"), "{:.1f} ms"), num(r.get("ms_pose_median"), "{:.1f} ms")] for r in rows]))

    # ---- frames
    if has_sheet:
        parts.append("""<h2>What the skeletons look like</h2><div class="card"><img src="frames.jpg" alt="Sample frames with each model's skeleton"></div>
<p>Six frames spread over the video, one row per model. The participant's right side is orange, the left side blue. White dots are points the model is confident about; small grey dots and thin lines are points below its threshold. A red border marks a frame that failed a check, with the reason written at the bottom.</p>""")

    # ---- 3D
    if space_section:
        parts.append(space_section)

    # ---- hands
    if hands:
        pct = lambda v: num(v, "{:.0%}")
        parts.append("""<h2>Hands: open or closed</h2>
<p>The pose models stop at the wrist. A separate hand model (MediaPipe Hand Landmarker, 21 points per hand) looks at an enlarged cut-out around each wrist. It is run once per frame, on the wrist position the selected pose models agree on, so this section is the same whichever pose models were ticked.</p>
<div class="card"><img src="figures/hands.png" alt="Finger extension and hand state over time"></div>""")
        parts.append(table(["Hand", "Found", "Of the frames where it was found: open", "Partly closed", "Closed", "Extension, median", "Lowest tenth"],
                           [[f"{h['hand'].capitalize()} hand", pct(h["found_frac"]), pct(h["open_frac"]), pct(h["partly_closed_frac"]), pct(h["closed_frac"]),
                             num(h["extension_median"], "{:.2f}"), num(h["extension_lowest_tenth"], "{:.2f}")] for h in hands]))
        parts.append("""<ul><li><b>Extension</b> measures how straight the four fingers are: the distance from wrist to fingertip divided by the length of the finger's path from the wrist. A flat hand is about 1; a fist is low, because the fingertips come back to the palm. It uses the hand model's 3D points, so it does not depend on which way the hand faces.</li>
<li><b>Open</b> means 0.85 or more, <b>closed</b> below 0.60, <b>partly closed</b> in between, judged on the median over seven frames (about a quarter of a second) so that one blurred frame does not flip the answer. The two limits come from geometry, not from labelled examples, and they have not been checked against clips of a clenched fist.</li>
<li><b>Not found is common and expected</b> when the hand moves fast and blurs, in low light, or when the hand is very small in the picture. "Left" and "right" are the participant's own hands.</li>
<li>The hand model took a median of """ + num(hands[0]["ms_both_hands_median"], "{:.0f} ms") + """ per frame for both hands, on the CPU.</li></ul>""")
        if has_hand_sheet:
            parts.append("""<div class="card"><img src="hands.jpg" alt="Enlarged cut-outs of both hands"></div>
<p>The cut-outs the hand model saw, from six frames spread over the video: right hand on the top row, left hand below. Green lines are the fingers it found, white dots the fingertips. Check these by eye before trusting the numbers.</p>""")

    # ---- models
    parts.append("<h2>The models</h2>")
    parts.append(table(["Model", "How it works", "Points", "Foot points", "Confident above", "Licence"],
                       [[name_cell(r), html.escape(tracks[r["model"]].kind), r["points"], r["foot_points"], f"{tracks[r['model']].min_score:g}",
                         html.escape(tracks[r["model"]].licence)] for r in rows], wrap=(1, 5)))
    parts.append("""<p>The same rule decides, for every model, which detected person is the participant: the largest person near the middle at the start, then whoever overlaps the participant's usual position over the last five seconds. Details are in <code>poselab/subject.py</code>.</p>
<h2>Files</h2><ul><li><code>summary.csv</code>: the numbers in this report, one row per model.</li><li><code>agreement.csv</code>: one row per pair of models.</li>
<li><code>signals.csv</code>: each model's smoothed arm elevation on a common time grid, for your own plots.</li>
<li><code>space.csv</code> and <code>space_signals.csv</code>: the 3D numbers above, and the 2D angle, 3D angle, arm direction and bone lengths of every frame, when a model with 3D points was selected.</li>
<li><code>pose/</code>: every point of every frame as each model returned it (the 3D points are the <code>world</code> array in MediaPipe's file); <code>pose/hands.npz</code> and <code>hands.csv</code> when the hand model was on.</li></ul>
<p class="lede">The video frames in this report show a participant's face. Keep the report on this computer.</p></main></body></html>""")
    path.write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------- entry point

def build(run, view, out_dir, samples=(), hand_samples=(), live_video=None, codec=None):
    """Write everything for one run into out_dir and return the path of report.html."""
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "pose").mkdir(exist_ok=True)
    for key, track in run.tracks.items():
        store.save(out_dir / "pose" / f"{key}.npz", track, run.t_ms, run.width, run.height, run.fps)
    if run.hands is not None:       # index 0 = the participant's left hand, 1 = right; 21 points each in full-frame pixels
        np.savez_compressed(out_dir / "pose" / "hands.npz", t_ms=run.t_ms, points=run.hands.points, extension=run.hands.extension,
                            ms=run.hands.ms, sides=np.array(["left", "right"]), width=run.width, height=run.height, fps=run.fps)

    side, analyses = analyse_run(run, view)
    rows = summary_rows(run, analyses)
    pairs = agreement(analyses, side)
    labels = {key: track.label for key, track in run.tracks.items()}
    t0 = run.t_ms[0] / 1000.0 if len(run.t_ms) else 0.0

    write_csv(out_dir / "summary.csv", rows)
    write_csv(out_dir / "agreement.csv", pairs)
    grid = next(iter(analyses.values()))["grid"]
    signals = [{"time_s": round(float(t - t0), 4), **{key: (round(float(a["signal"][i]), 3) if i < len(a["signal"]) and np.isfinite(a["signal"][i]) else "")
                                                       for key, a in analyses.items()}} for i, t in enumerate(grid)]
    write_csv(out_dir / "signals.csv", signals)

    style()
    figure_elevation(analyses, labels, side, t0, out_dir / "figures" / "elevation.png")
    if pairs:
        figure_agreement(pairs, list(analyses), labels, out_dir / "figures" / "agreement.png")
    figure_quality(analyses, labels, t0, out_dir / "figures" / "quality.png")
    figure_speed(rows, out_dir / "figures" / "speed.png")
    has_sheet = overlay_sheet(list(samples), analyses, labels, run.height, out_dir / "frames.jpg")

    space_section = None        # the first selected model that gave 3D points (of the five, only MediaPipe does)
    space_key = next((key for key, a in analyses.items() if a["clip"]["world"] is not None and a["usable"].any()), None)
    if space_key is not None:
        a = analyses[space_key]
        measured = space.measure(a)
        write_csv(out_dir / "space.csv", [{"model": space_key, **measured["row"]}])
        cell = lambda v, digits: round(float(v), digits) if np.isfinite(v) else ""
        write_csv(out_dir / "space_signals.csv", [
            {"time_s": round(float(a["clip"]["t"][i] - t0), 4), "usable": int(a["usable"][i]),
             "elevation_2d_deg": cell(a["elevation"][i], 2), "elevation_3d_deg": cell(a["elevation_3d"][i], 2),
             "forward_angle_deg": cell(measured["direction"][i], 2), "upper_arm_cm": cell(measured["upper_cm"][i], 2),
             "forearm_cm": cell(measured["fore_cm"][i], 2)} for i in range(len(a["usable"]))])
        figure_space(a, measured, COLOUR[space_key], t0, out_dir / "figures" / "space.png")
        space_section = space_html(labels[space_key], measured["row"], space_sheet(list(samples), a, run.height, out_dir / "space.jpg"))

    hands, has_hand_sheet = None, False
    if run.hands is not None and len(run.t_ms):
        hands = hand_rows(run)
        write_csv(out_dir / "hands.csv", hands)
        figure_hands(run, t0, out_dir / "figures" / "hands.png")
        has_hand_sheet = hand_sheet(list(hand_samples), out_dir / "hands.jpg")

    report = out_dir / "report.html"
    write_html(report, run, view, side, rows, pairs, run.tracks, has_sheet, Path(live_video).name if live_video else None, codec,
               hands=hands, has_hand_sheet=has_hand_sheet, space_section=space_section)
    return report
