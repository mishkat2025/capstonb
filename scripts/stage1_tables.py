"""Stage 1 tables: clip quality, repetitions, features, and the pass/fail criteria.

    .venv/Scripts/python scripts/stage1_tables.py --exercise E01

Reads data/pose/<run>/*.npz (default run: full_p1, MediaPipe Full with one pose per frame)
and writes outputs/stage1/<run>/qc_clips.csv, reps.csv and features.csv, plus
outputs/stage1/summary.txt with the criteria from docs/feasibility_plan.md.

The Lite model was dropped on 2026-10-12. The version of this script that also compared
Lite with Full, and one pose with two poses per frame, is kept with its results in
archive/2026-10-12_lite/.
"""
import argparse
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

import posekit as pk

ROOT = Path(__file__).resolve().parents[1]
META = ["clip", "participant", "cohort", "gender", "view", "variation", "score", "pf1_zero_judges", "pf2", "pf4", "duplicate_of"]


def analyse_run(run_dir, manifest):
    quality, reps, features = [], [], []
    for path in sorted(run_dir.glob("*.npz")):
        row = manifest.get(path.stem)
        if row is None:
            continue
        a = pk.analyse_clip(path, front_view=row["view"] == "F")
        meta = {k: row[k] for k in META}
        relaxed = ~a["flags"][:, :pk.FLAGS.index("bone_length")].any(axis=1)     # sensitivity: ignore the limb-length flag
        relaxed_gaps = pk.gaps(a["clip"]["t"], relaxed)
        in_3d = a["elevation_3d"][a["usable"]]
        a["quality"]["range_2d"] = a["features"]["move_elev_range"]
        a["quality"]["range_3d"] = pk.robust_range(in_3d)
        a["quality"]["performance"] = row["performance"]
        a["quality"]["usable_frac_no_limb_flag"] = float(relaxed.mean()) if len(relaxed) else 0.0
        a["quality"]["longest_gap_no_limb_flag_s"] = float(max(relaxed_gaps)) if relaxed_gaps else 0.0
        quality.append({**meta, **a["quality"]})
        features.append({**meta, "moving_side": a["side"], "reps": len(a["reps"]), **a["features"]})
        reps += [{"clip": path.stem, "rep": i + 1, **r} for i, r in enumerate(a["reps"])]
    return pd.DataFrame(quality), pd.DataFrame(reps), pd.DataFrame(features)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exercise", default="E01")
    ap.add_argument("--run", default="full_p1")
    ap.add_argument("--pose", type=Path, default=ROOT / "data" / "pose")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "stage1")
    args = ap.parse_args()

    manifest = {r["clip"]: r for r in csv.DictReader(open(ROOT / "data" / "meta" / "manifest.csv", encoding="utf-8"))
                if r["exercise"] == args.exercise}
    out = io.StringIO()
    say = lambda *a: print(*a, file=out)
    run = args.run
    if not any((args.pose / run).glob("*.npz")):
        raise SystemExit(f"no pose files in {args.pose / run}")
    q, reps, feats = analyse_run(args.pose / run, manifest)
    (args.out / run).mkdir(parents=True, exist_ok=True)
    q.to_csv(args.out / run / "qc_clips.csv", index=False)
    reps.to_csv(args.out / run / "reps.csv", index=False)
    feats.to_csv(args.out / run / "features.csv", index=False)

    q["usable_clip"] = (q["usable_frac"] >= 0.8) & (q["longest_gap_s"] <= 1.0)
    say(f"\n================ {run}: {len(q)} clips ({args.exercise}, development participants) ================")
    say("\nclips per cohort and view:")
    say(q.groupby(["cohort", "view"]).size().unstack(fill_value=0).to_string())

    say("\nshare of usable clips (>= 80% usable frames, no gap > 1 s) / median usable-frame share, by variation and view")
    for cohort in ("expert", "nonexpert"):
        sub = q[q["cohort"] == cohort]
        table = sub.groupby(["variation", "view"]).agg(n=("clip", "size"), usable=("usable_clip", "mean"), median_frames=("usable_frac", "median"))
        say(f"  {cohort}")
        say(table.round(3).to_string())

    say("\nwhich flags fire (mean share of frames), front view; swap_repaired counts relabelled frames and does not make a frame unusable")
    flag_cols = [c for c in q.columns if c.startswith("flag_")] + ["swap_repaired_frac"]
    say(q[q["view"] == "F"].groupby(["cohort", "variation"])[flag_cols].mean().round(3).to_string())

    front = q[q["view"] == "F"].merge(feats[["clip", "move_elev_range", "rest_elev_range"]], on="clip")
    say("\nCRITERION coverage (front view, full light and medium light): share of usable clips, need >= 0.90 per cohort")
    for cohort in ("expert", "nonexpert"):
        sub = front[(front["cohort"] == cohort) & front["variation"].isin(["FL", "ML"])]
        relaxed = (sub["usable_frac_no_limb_flag"] >= 0.8) & (sub["longest_gap_no_limb_flag_s"] <= 1.0)
        say(f"  {cohort}: {sub['usable_clip'].mean():.3f}  ({int(sub['usable_clip'].sum())} of {len(sub)});  "
            f"ignoring the limb-length flag: {relaxed.mean():.3f} ({int(relaxed.sum())} of {len(sub)})")

    say("\nCRITERION signal (front view, usable clips): one arm clearly moves (range >= 45 deg and >= 2x the other arm), need >= 0.90")
    for cohort in ("expert", "nonexpert"):
        sub = front[(front["cohort"] == cohort) & front["usable_clip"]]
        ok = (sub["move_elev_range"] >= 45) & (sub["move_elev_range"] >= 2 * sub["rest_elev_range"])
        say(f"  {cohort}: {ok.mean():.3f}  ({int(ok.sum())} of {len(sub)}); moving arm: {sub['moving_side'].value_counts().to_dict()}; "
            f"median range {sub['move_elev_range'].median():.0f} deg")

    say("\nrepetition count from pose against the judges' repetition item (10 x PF4), usable clips, exploratory")
    for (cohort, view), sub in q[q["usable_clip"]].groupby(["cohort", "view"]):
        judged = (10 * sub["pf4"].astype(float)).round()
        counted = sub["reps"].clip(upper=10)
        diff = counted - judged
        say(f"  {cohort:9s} {view}: n={len(sub):3d}  exact {np.mean(diff == 0):.2f}  within one {np.mean(diff.abs() <= 1):.2f}  "
            f"mean difference {diff.mean():+.2f}  judged median {judged.median():.0f}  counted median {counted.median():.0f}")

    if q["view"].nunique() == 3:
        say("\nside views against the front view (stress test): medians per cohort and view")
        by_view = q.groupby(["cohort", "view"]).agg(
            clips=("clip", "size"), usable_clips=("usable_clip", "mean"), usable_frames=("usable_frac", "median"),
            range_2d=("range_2d", "median"), range_3d=("range_3d", "median"), reps=("reps", "median"),
            right_arm=("moving_side", lambda s: (s == "right").mean()))
        say(by_view.round(2).to_string())
        wide = q[q["cohort"] == "expert"].pivot_table(index="performance", columns="view", values=["reps", "range_2d", "range_3d"], aggfunc="first").dropna()
        say(f"\nexpert performances with all three views: {len(wide)}")
        for view in ("L", "R"):
            d = wide["reps"][view] - wide["reps"]["F"]
            say(f"  repetitions, {view} against front: same {np.mean(d == 0):.2f}  within one {np.mean(d.abs() <= 1):.2f}")
            for kind in ("range_2d", "range_3d"):
                d = wide[kind][view] - wide[kind]["F"]
                say(f"  {kind}, {view} minus front: median {d.median():+.0f} deg  (middle half {d.quantile(0.25):+.0f} to {d.quantile(0.75):+.0f})")

    say("\nmovement in usable front-view clips, medians")
    cols = ["duration_s", "reps", "jitter_rms_deg"]
    say(front[front["usable_clip"]].groupby("cohort")[cols].median().round(2).to_string())
    fcols = ["move_elev_p5", "move_elev_p95", "move_elev_range", "rest_elev_range", "elbow_median", "trunk_incl_median",
             "wrist_travel_x", "wrist_travel_y", "arm_len_ratio", "raise_s_median", "lower_s_median", "speed_p95"]
    usable_front = feats.merge(q[["clip", "usable_clip"]], on="clip")
    usable_front = usable_front[(usable_front["view"] == "F") & usable_front["usable_clip"]]
    say(usable_front.groupby("cohort")[fcols].median().round(2).T.to_string())

    (args.out / "summary.txt").write_text(out.getvalue(), encoding="utf-8")
    print(out.getvalue())


if __name__ == "__main__":
    main()
