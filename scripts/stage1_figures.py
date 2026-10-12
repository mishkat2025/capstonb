"""Stage 1 figures for one pose run.

    .venv/Scripts/python scripts/stage1_figures.py --exercise E01

Writes PNGs to outputs/stage1/<run>/figures/. Each figure has a matching table in
outputs/stage1/<run>/ (qc_clips.csv, reps.csv, features.csv) written by stage1_tables.py.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import posekit as pk

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED, GRID, AXIS, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COHORT_COLOR = {"expert": BLUE, "nonexpert": ORANGE}
COHORT_LABEL = {"expert": "Expert", "nonexpert": "Non-expert"}
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SEQUENTIAL.set_bad(NEUTRAL)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "lines.linewidth": 1.5, "figure.dpi": 150,
})


def save(fig, out, name):
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / name)


def traces(clips, analyses, out):
    """Arm elevation over time for a few clips of each cohort."""
    picks = []
    for cohort in ("expert", "nonexpert"):
        front = [c for c in clips if c["cohort"] == cohort and c["view"] == "F"]
        step = max(1, len(front) // 6)
        picks += front[::step][:6]
    fig, axes = plt.subplots(4, 3, figsize=(11, 8.5), sharey=True)
    for ax, c in zip(axes.ravel(), picks):
        a = analyses[c["clip"]]
        t = a["clip"]["t"] - a["clip"]["t"][0]
        resting = "left" if a["side"] == "right" else "right"
        bad = ~a["usable"]
        ax.fill_between(t, 0, 180, where=bad, color=NEUTRAL, step="mid", linewidth=0)
        ax.plot(t, pk.arm_elevation(a["clip"]["px"], resting), color=MUTED, linewidth=1, label="Resting arm")
        ax.plot(a["grid"] - a["grid"][0], a["signal"], color=COHORT_COLOR[c["cohort"]], label="Moving arm")
        ax.plot([r["t_peak"] - a["grid"][0] for r in a["reps"]], [r["peak_deg"] for r in a["reps"]], "o", color=INK, markersize=3.5)
        ax.set_title(f"{c['clip']}  ·  {len(a['reps'])} peaks  ·  judges' repetition item {float(c['pf4']):.1f}", fontsize=7.5)
        ax.set_ylim(0, 185)
        ax.set_yticks([0, 45, 90, 135, 180])
    for ax in axes[-1]:
        ax.set_xlabel("Time in clip (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Arm elevation (°)")
    handles = [plt.Line2D([], [], color=BLUE, label="Moving arm, expert clip"),
               plt.Line2D([], [], color=ORANGE, label="Moving arm, non-expert clip"),
               plt.Line2D([], [], color=MUTED, linewidth=1, label="Resting arm"),
               plt.Line2D([], [], color=INK, marker="o", markersize=3.5, linestyle="", label="Detected peak")]
    fig.legend(handles=handles, loc="upper right", ncols=4, fontsize=8, bbox_to_anchor=(0.995, 0.965))
    fig.suptitle("Arm elevation from pose, front view. Grey bands are unusable frames. Rows 1–2 expert, rows 3–4 non-expert.",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "traces.png")


def heatmap(clips, analyses, out):
    """Every front-view clip as one row: elevation of the moving arm over time."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), gridspec_kw={"width_ratios": [3, 1.2]})
    for ax, cohort, seconds in zip(axes, ("expert", "nonexpert"), (45, 8)):
        rows = [c for c in clips if c["cohort"] == cohort and c["view"] == "F"]
        rows.sort(key=lambda c: (c["participant"], c["variation"]))
        n = int(seconds * pk.FS)
        image = np.full((len(rows), n), np.nan)
        for i, c in enumerate(rows):
            s = analyses[c["clip"]]["signal"][:n]
            image[i, :len(s)] = s
        mesh = ax.imshow(np.ma.masked_invalid(image), aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=180,
                         extent=(0, seconds, len(rows), 0), interpolation="nearest")
        ax.set_title(f"{COHORT_LABEL[cohort]}: {len(rows)} clips, sorted by participant")
        ax.set_xlabel("Time in clip (s)")
        ax.set_ylabel("Clip")
        ax.grid(False)
    bar = fig.colorbar(mesh, ax=axes, shrink=0.6, pad=0.02)
    bar.set_label("Elevation of the moving arm (°)")
    bar.outline.set_visible(False)
    fig.suptitle("Front-view clips. Expert clips are sets of repetitions; non-expert clips hold a single raise. Grey: no usable signal.", x=0.01, ha="left", fontsize=9.5)
    save(fig, out, "heatmap.png")


def rep_overlay(clips, analyses, out):
    """Repetitions on a common 0-100% time axis, per cohort."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    u = np.linspace(0, 1, 101)
    for ax, cohort in zip(axes, ("expert", "nonexpert")):
        curves = []
        for c in clips:
            if c["cohort"] != cohort or c["view"] != "F":
                continue
            a = analyses[c["clip"]]
            g, y = a["grid"], a["signal"]
            for r in a["reps"]:
                lo, hi = r["t_peak"] - 1.2 * r["raise_s"] - 0.6, r["t_peak"] + 1.2 * r["lower_s"] + 0.6
                seg = (g >= lo) & (g <= hi) & np.isfinite(y)
                if seg.sum() > 10 and np.isfinite(lo + hi):
                    curves.append(np.interp(u, (g[seg] - g[seg][0]) / (g[seg][-1] - g[seg][0]), y[seg]))
        curves = np.array(curves)
        for curve in curves[:: max(1, len(curves) // 150)]:
            ax.plot(100 * u, curve, color=COHORT_COLOR[cohort], alpha=0.08, linewidth=0.8)
        if len(curves):
            mean, sd = curves.mean(axis=0), curves.std(axis=0)
            ax.fill_between(100 * u, mean - sd, mean + sd, color=COHORT_COLOR[cohort], alpha=0.18, linewidth=0)
            ax.plot(100 * u, mean, color=COHORT_COLOR[cohort], linewidth=2)
        ax.set_title(f"{COHORT_LABEL[cohort]}: {len(curves)} repetitions (mean, ±1 SD band)")
        ax.set_xlabel("Position within the repetition (%)")
        ax.set_ylim(0, 185)
        ax.set_yticks([0, 45, 90, 135, 180])
    axes[0].set_ylabel("Arm elevation (°)")
    fig.suptitle("Shape of one repetition, front view", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, out, "rep_overlay.png")


def wrist_paths(clips, analyses, out):
    """Wrist position relative to the shoulder, mirrored so the moving arm points to +x."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharex=True, sharey=True)
    for ax, cohort in zip(axes, ("expert", "nonexpert")):
        rows = [c for c in clips if c["cohort"] == cohort and c["view"] == "F"]
        for c in rows[:: max(1, len(rows) // 25)]:
            a = analyses[c["clip"]]
            xy = a["rel_wrist"][a["usable"]]
            sign = -1 if a["side"] == "right" else 1      # the subject's right arm is on the image left
            ax.plot(sign * xy[:, 0], -xy[:, 1], color=COHORT_COLOR[cohort], alpha=0.25, linestyle="", marker=".", markersize=2)
        ax.plot(0, 0, "o", color=INK, markersize=5)
        ax.annotate("shoulder", (0, 0), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK2)
        ax.set_title(f"{COHORT_LABEL[cohort]} (sample of clips)")
        ax.set_xlabel("Sideways from the shoulder (torso lengths)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Above the shoulder (torso lengths)")
    fig.suptitle("Path of the moving wrist, front view", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, out, "wrist_paths.png")


def coverage(clips, analyses, out):
    """Share of usable clips by variation and view, per cohort."""
    variations, views = ["FL", "ML", "LL", "LJ", "HJ", "O", "LR"], ["F", "L", "R"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, cohort in zip(axes, ("expert", "nonexpert")):
        table = np.full((len(variations), len(views)), np.nan)
        counts = np.zeros_like(table)
        for i, var in enumerate(variations):
            for j, view in enumerate(views):
                q = [analyses[c["clip"]]["quality"] for c in clips if c["cohort"] == cohort and c["variation"] == var and c["view"] == view]
                counts[i, j] = len(q)
                if q:
                    table[i, j] = np.mean([x["usable_frac"] >= 0.8 and x["longest_gap_s"] <= 1.0 for x in q])
        ax.imshow(np.ma.masked_invalid(table), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
        for i in range(len(variations)):
            for j in range(len(views)):
                if counts[i, j]:
                    ax.text(j, i, f"{table[i, j]:.0%}\nn={int(counts[i, j])}", ha="center", va="center", fontsize=7.5,
                            color="white" if table[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(views)), ["Front", "Left", "Right"])
        ax.set_yticks(range(len(variations)), ["Full light", "Medium light", "Low light", "Low jitter", "High jitter", "Occlusion", "Low resolution"])
        ax.set_title(COHORT_LABEL[cohort])
        ax.grid(False)
        ax.tick_params(length=0)
    fig.suptitle("Share of clips with ≥80% usable frames and no gap over 1 s", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, out, "coverage.png")


def views_2d_3d(clips, analyses, out):
    """One expert performance seen from three views: 2D against 3D elevation."""
    by_perf = {}
    for c in clips:
        if c["cohort"] == "expert" and c["variation"] == "FL":
            by_perf.setdefault(c["performance"], {})[c["view"]] = c
    perf = next((p for p in sorted(by_perf) if len(by_perf[p]) == 3), None)
    if perf is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, sharey=True)
    for view, color, label in (("F", BLUE, "Front"), ("L", ORANGE, "Left"), ("R", AQUA, "Right")):
        a = analyses[by_perf[perf][view]["clip"]]
        t = a["clip"]["t"] - a["clip"]["t"][0]
        for ax, values in ((axes[0], a["elevation"]), (axes[1], a["elevation_3d"])):
            ax.plot(t, np.where(a["usable"], values, np.nan), color=color, label=label)   # unusable frames leave a break
    axes[0].set_title("2D: angle measured in the image")
    axes[1].set_title("3D: angle from MediaPipe world landmarks")
    axes[1].set_xlabel("Time in clip (s)")
    for ax in axes:
        ax.set_ylabel("Arm elevation (°)")
        ax.set_ylim(0, 185)
        ax.set_yticks([0, 45, 90, 135, 180])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=3, bbox_to_anchor=(0.995, 0.965))
    fig.suptitle(f"The same performance from three views ({perf}). Lines break where frames are unusable; clips are not time-aligned.",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "views_2d_3d.png")


def peak_histogram(clips, analyses, out):
    """Peak arm elevation per front-view clip, by cohort."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(0, 190, 10)
    for cohort in ("expert", "nonexpert"):
        peaks = [analyses[c["clip"]]["features"]["move_elev_p95"] for c in clips if c["cohort"] == cohort and c["view"] == "F"]
        peaks = [p for p in peaks if np.isfinite(p)]
        ax.hist(peaks, bins=bins, histtype="stepfilled", alpha=0.35, color=COHORT_COLOR[cohort], edgecolor=COHORT_COLOR[cohort], linewidth=1.5,
                label=f"{COHORT_LABEL[cohort]} (n={len(peaks)})")
    ax.set_xlabel("95th percentile of moving-arm elevation in the clip (°)")
    ax.set_ylabel("Clips")
    ax.set_xticks(range(0, 181, 30))
    ax.legend(loc="upper left")
    ax.set_title("How high the arm goes, front view")
    save(fig, out, "peak_elevation.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="full_p1")
    ap.add_argument("--exercise", default="E01")
    args = ap.parse_args()

    run_dir = ROOT / "data" / "pose" / args.run
    out = ROOT / "outputs" / "stage1" / args.run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    manifest = [r for r in csv.DictReader(open(ROOT / "data" / "meta" / "manifest.csv", encoding="utf-8"))
                if r["exercise"] == args.exercise and (run_dir / f"{r['clip']}.npz").exists()]
    analyses = {c["clip"]: pk.analyse_clip(run_dir / f"{c['clip']}.npz", front_view=c["view"] == "F") for c in manifest}
    for figure in (traces, heatmap, rep_overlay, wrist_paths, coverage, views_2d_3d, peak_histogram):
        figure(manifest, analyses, out)


if __name__ == "__main__":
    main()
