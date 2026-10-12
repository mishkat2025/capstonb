"""What MediaPipe's 3D points say over the Stage 1 pose files: one row per clip, and a summary.

    .venv/Scripts/python scripts/stage1_space.py --exercise E01

Reads data/pose/<run>/*.npz (default run: full_p1) and writes
outputs/stage1/<run>/space_clips.csv and space_summary.txt. No model is run: the pose
files already hold the 3D points. The measures are those of poselab/space.py, on the same
arm and the same usable frames as the demo's report (five checks, no limb-length check).

A clip is counted in the summary when at least 90% of its frames are usable and the arm
spends at least 30 frames in the band where its direction can be read.
"""
import argparse
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

import posekit as pk
from poselab import analysis, space

ROOT = Path(__file__).resolve().parents[1]
META = ["clip", "participant", "cohort", "view", "variation", "performance", "pf1_zero_judges"]
MIN_USABLE, MIN_RAISED = 0.9, 30
VIEW = {"F": "front", "L": "left side", "R": "right side"}


def scan(run_dir, manifest):
    rows = []
    for path in sorted(run_dir.glob("*.npz")):
        meta = manifest.get(path.stem)
        if meta is None:
            continue
        a = analysis.analyse(pk.load_clip(path, front_view=meta["view"] == "F"), use_3d=False)
        if a["usable"].any():
            rows.append({**{k: meta[k] for k in META}, "usable_frac": a["quality"]["usable_frac"], **space.measure(a)["row"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exercise", default="E01")
    ap.add_argument("--run", default="full_p1")
    ap.add_argument("--pose", type=Path, default=ROOT / "data" / "pose")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "stage1")
    args = ap.parse_args()

    manifest = {r["clip"]: r for r in csv.DictReader(open(ROOT / "data" / "meta" / "manifest.csv", encoding="utf-8"))
                if r["exercise"] == args.exercise}
    d = scan(args.pose / args.run, manifest)
    if d.empty:
        raise SystemExit(f"no pose files in {args.pose / args.run}")
    d["counted"] = (d["usable_frac"] >= MIN_USABLE) & (d["frames_arm_raised"] >= MIN_RAISED)
    (args.out / args.run).mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out / args.run / "space_clips.csv", index=False)

    out = io.StringIO()
    say = lambda *a: print(*a, file=out)
    u = d[d["counted"]]
    same = (u["repetitions_2d"] == u["repetitions_3d"]) & (u["repetitions_2d_cut_off"] == u["repetitions_3d_cut_off"])
    middle = lambda s: f"{s.median():.1f} ({s.quantile(0.25):.1f} to {s.quantile(0.75):.1f})"
    say(f"{len(d)} clips with at least one usable frame; {len(u)} counted, from {u['participant'].nunique()} participants")
    say(f"counted: at least {MIN_USABLE:.0%} usable frames and at least {MIN_RAISED} frames with the arm between "
        f"{space.RAISED[0]:.0f} and {space.RAISED[1]:.0f} degrees\n")
    say("Medians over the counted clips; in brackets the middle half of the clips")
    for (cohort, view), g in d.groupby(["cohort", "view"]):
        c = g[g["counted"]]
        say(f"\n{cohort}, {VIEW[view]} view: {len(c)} of {len(g)} clips counted")
        if c.empty:
            continue
        say(f"  elevation range, picture / 3D          {c['elevation_2d_range_deg'].median():.1f} / {c['elevation_3d_range_deg'].median():.1f} deg")
        say(f"  highest elevation, picture / 3D        {c['elevation_2d_high_deg'].median():.1f} / {c['elevation_3d_high_deg'].median():.1f} deg")
        say(f"  difference per frame (clip median)     {c['difference_median_deg'].median():.1f} deg")
        say(f"  direction of the raise (clip median)   {middle(c['forward_angle_median_deg'])} deg; "
            f"lowest {c['forward_angle_median_deg'].min():.1f}, highest {c['forward_angle_median_deg'].max():.1f}")
        say(f"  upper arm, arm down / arm overhead     {c['upper_arm_cm_arm_down'].median():.1f} / {c['upper_arm_cm_arm_up'].median():.1f} cm")
        say(f"  spread of length, upper arm / forearm  {c['upper_arm_spread_pct'].median():.1f}% / {c['forearm_spread_pct'].median():.1f}%")
        say(f"  same repetition count from both        {int(same[c.index].sum())} of {len(c)}")
    say(f"\nall counted clips: spread of length, upper arm / forearm {u['upper_arm_spread_pct'].median():.1f}% / {u['forearm_spread_pct'].median():.1f}%; "
        f"same repetition count {int(same.sum())} of {len(u)}")

    front = u[u["view"] == "F"].sort_values("forward_angle_median_deg", key=np.abs, ascending=False)
    say(f"\nfront view, {len(front)} counted clips: direction within 20 degrees of sideways in {int((front['forward_angle_median_deg'].abs() < 20).sum())}")
    say("the five furthest from sideways:")
    for _, r in front.head(5).iterrows():
        say(f"  {r['clip']:22s} {r['forward_angle_median_deg']:6.1f} deg   judges marking it as not the right exercise: {r['pf1_zero_judges']}")

    say("\nThe same expert performance filmed from two sides (second minus first; median, middle half)")
    table = u[u["cohort"] == "expert"].pivot_table(index="performance", columns="view", values=["forward_angle_median_deg", "elevation_3d_range_deg"])
    for a, b in (("F", "L"), ("F", "R")):
        if ("forward_angle_median_deg", a) not in table or ("forward_angle_median_deg", b) not in table:
            continue
        both = table.dropna(subset=[("forward_angle_median_deg", a), ("forward_angle_median_deg", b)])
        turn = both[("forward_angle_median_deg", b)] - both[("forward_angle_median_deg", a)]
        reach = both[("elevation_3d_range_deg", b)] - both[("elevation_3d_range_deg", a)]
        say(f"  {VIEW[a]} and {VIEW[b]}: {len(both)} performances; 3D range {middle(reach)} deg; direction {middle(turn)} deg, "
            f"median size of the difference {turn.abs().median():.1f} deg")

    (args.out / args.run / "space_summary.txt").write_text(out.getvalue(), encoding="utf-8")
    print(out.getvalue())


if __name__ == "__main__":
    main()
