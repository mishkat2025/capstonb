"""Write the list of clips to watch by hand, with empty columns to fill in.

    python scripts/make_watch_log.py --exercise E01

Picks 12 ordinary expert and 12 ordinary non-expert clips of development participants,
spread over views and variations, plus every clip the judges marked "not the right
exercise" and every byte-identical pair. The sheet shows no pose-derived numbers, so the
counts entered are independent of the pipeline. An existing sheet is never overwritten.
"""
import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["clip", "why", "cohort", "view", "variation", "repetitions_seen", "moving_arm", "peak_height",
           "idle_seconds_at_start", "second_person", "notes"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exercise", default="E01")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "stage1" / "watch_log.csv")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; delete it first if you really want a new list")

    dev = {r["participant"] for r in csv.DictReader(open(ROOT / "data" / "meta" / "participant_splits.csv", encoding="utf-8")) if r["split"] == "dev"}
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "meta" / "manifest.csv", encoding="utf-8"))
            if r["exercise"] == args.exercise and r["on_disk"] == "1" and r["participant"] in dev]
    special = [r for r in rows if r["pf1_zero_judges"] == "3" or r["duplicate_of"]]
    ordinary = [r for r in rows if r not in special and r["pf1_zero_judges"] == "0"]

    rng = random.Random(args.seed)
    picked = []
    for cohort in ("expert", "nonexpert"):
        pool = [r for r in ordinary if r["cohort"] == cohort]
        rng.shuffle(pool)
        seen, chosen = set(), []
        for r in pool:                       # first pass: one clip per (view, variation) and participant
            key = (r["view"], r["variation"])
            if key not in seen and r["participant"] not in {c["participant"] for c in chosen}:
                seen.add(key)
                chosen.append(r)
        chosen += [r for r in pool if r not in chosen]
        picked += [(r, "ordinary") for r in chosen[:12]]
    picked += [(r, "judges: not the right exercise" if r["pf1_zero_judges"] == "3" else "byte-identical pair") for r in special]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r, why in picked:
            w.writerow({"clip": r["clip"], "why": why, "cohort": r["cohort"], "view": r["view"], "variation": r["variation"]})
    print(f"wrote {len(picked)} clips to {args.out}")


if __name__ == "__main__":
    main()
