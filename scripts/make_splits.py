"""Assign every participant to the lockbox or to one of five development folds.

The participant is the only unit that is ever split: all clips of a person (every exercise,
view and variation) share one assignment. Assignment is stratified so that each fold holds
experts and non-experts, both genders, and people who did the floor exercises (E07-E09).

Run once, before looking at any pose output, and commit the result:

    python scripts/make_splits.py            # writes data/meta/participant_splits.csv
"""
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOOR = {"E07", "E08", "E09"}


def stratum(cohort, gender, exercises):
    if gender == "F":
        return f"{cohort}-F"
    return f"{cohort}-M-floor" if exercises & FLOOR else f"{cohort}-M-standing"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "meta" / "manifest.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "meta" / "participant_splits.csv")
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lockbox", type=float, default=0.2, help="fraction of each stratum held out")
    args = ap.parse_args()

    clips = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    people = defaultdict(lambda: {"exercises": set(), "n_clips": 0})
    for c in clips:
        p = people[c["participant"]]
        p.update(cohort=c["cohort"], gender=c["gender"])
        p["exercises"].add(c["exercise"])
        p["n_clips"] += 1

    strata = defaultdict(list)
    for pid, p in people.items():
        p["stratum"] = stratum(p["cohort"], p["gender"], p["exercises"])
        strata[p["stratum"]].append(pid)

    rng = random.Random(args.seed)
    next_fold = 0
    for name in sorted(strata):
        ids = sorted(strata[name])
        rng.shuffle(ids)
        n_lock = round(args.lockbox * len(ids))
        for pid in ids[:n_lock]:
            people[pid].update(split="lockbox", fold="")
        for pid in ids[n_lock:]:  # deal round-robin, continuing across strata so fold sizes stay equal
            people[pid].update(split="dev", fold=next_fold)
            next_fold = (next_fold + 1) % args.folds

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["participant", "cohort", "gender", "stratum", "n_clips", "exercises", "split", "fold"])
        for pid in sorted(people):
            p = people[pid]
            w.writerow([pid, p["cohort"], p["gender"], p["stratum"], p["n_clips"], " ".join(sorted(p["exercises"])), p["split"], p["fold"]])

    groups = ["lockbox"] + [f"fold{k}" for k in range(args.folds)]
    group_of = lambda pid: "lockbox" if people[pid]["split"] == "lockbox" else f"fold{people[pid]['fold']}"
    print(f"seed {args.seed}: {len(people)} participants -> " + ", ".join(f"{g}: {sum(group_of(p) == g for p in people)}" for g in groups))
    print("\nparticipants per group (expert + nonexpert), by exercise")
    print("exercise  " + "  ".join(f"{g:>8s}" for g in groups))
    for ex in sorted({c["exercise"] for c in clips}):
        cells = []
        for g in groups:
            ids = [pid for pid, p in people.items() if ex in p["exercises"] and group_of(pid) == g]
            n = Counter(people[pid]["cohort"] for pid in ids)
            cells.append(f"{n['expert']:>3d}+{n['nonexpert']:<3d} ")
        print(f"{ex}       " + "  ".join(cells))
    print("\nmembers")
    for g in groups:
        print(f"  {g}: " + " ".join(sorted(pid for pid in people if group_of(pid) == g)))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
