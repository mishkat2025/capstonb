"""Build the per-clip manifest for MobiPhysio (standard library only).

Joins the Dataverse file listing with the physiotherapist marking CSV and, when --videos
is given, with the clips on disk. Writes:

    data/meta/manifest.csv      one row per clip
    data/meta/scores_long.csv   one row per clip x judge (the seven EAAQ items)

    python scripts/build_manifest.py --videos data

The local copy is the reference for names. On Dataverse (version 3.0) 370 E01 files carry
the bytes of a different E01 clip, so the dv_* columns describe what Dataverse serves under
a name, and dv_name_ok says whether that agrees with the local copy ("yes", "no", or
"unknown" for clips that are not on disk).
"""
import argparse
import csv
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "meta" / "source"
DOI = "doi:10.7910/DVN/XSI0QN"
LISTING_URL = f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={DOI}"

NAME = re.compile(r"^E(\d{2})_P(\d{2,3})_A([FLR])_V([A-Z]+)_G([MF])$")
KNOWN_VARIATIONS = {"FL", "ML", "LL", "LJ", "HJ", "O", "LR"}
EXERCISES = {1: "Abduction", 2: "Adduction", 3: "Lateral Rotation", 4: "Medial Rotation", 5: "Circumduction",
             6: "Wrist Extension", 7: "Hip Joint Flexion", 8: "Lumber Flexion", 9: "Back Extension"}
# Not encoded in the filename or the README. P01-P24 reproduces the paper's expert/non-expert
# counts for all nine exercises and matches the two judge-label styles in the marking CSV.
EXPERT_MAX_ID = 24
ITEMS = ["pf1", "pf2", "pf3", "pf4", "cf1", "cf2", "cf3"]


def load_listing(path, fetch):
    if fetch or not path.exists():
        req = urllib.request.Request(LISTING_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    version = json.loads(path.read_text(encoding="utf-8"))["data"]["latestVersion"]
    files = [f["dataFile"] for f in version["files"] if f["dataFile"]["filename"].lower().endswith(".mp4")]
    label = f"{version.get('versionNumber')}.{version.get('versionMinorNumber')} released {version.get('releaseTime')}"
    return files, label


def load_scores(path):
    """Return {CLIP_NAME: [[7 items] per judge]} and a list of name fixes that were applied."""
    rows = list(csv.reader(open(path, encoding="utf-8-sig", newline="")))[2:]  # two header rows
    scores, fixes, cur = {}, [], None
    for r in rows:
        r = [c.strip() for c in r]
        if r[0]:
            cur = r[0].upper()
            if cur != r[0]:
                fixes.append(f"{r[0]} -> {cur}")
            scores[cur] = []
        if cur and len(r) > 9 and r[1]:
            scores[cur].append([float(x) for x in r[2:9]])
    return scores, fixes


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listing", type=Path, default=SOURCE / "dataverse_listing_v3.0.json")
    ap.add_argument("--fetch-listing", action="store_true", help="re-download the listing from Dataverse")
    ap.add_argument("--scores", type=Path, default=SOURCE / "Physiotherapist Exercise Marking.csv")
    ap.add_argument("--videos", type=Path, help="folder (searched recursively) holding the clips; every clip found is hashed")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "meta")
    args = ap.parse_args()

    files, listing_label = load_listing(args.listing, args.fetch_listing)
    scores, name_fixes = load_scores(args.scores)
    on_disk = {p.name: p for p in args.videos.rglob("*.mp4")} if args.videos else {}
    local_md5 = {name: md5_of(p) for name, p in on_disk.items()}

    dv_names_of = defaultdict(list)
    for f in files:
        dv_names_of[f["md5"]].append(f["filename"][:-4])
    local_names_of = defaultdict(list)
    for name, h in local_md5.items():
        local_names_of[h].append(name[:-4])

    rows, long_rows, problems = [], [], []
    for f in sorted(files, key=lambda f: f["filename"]):
        clip = f["filename"][:-4]
        m = NAME.match(clip)
        if not m:
            problems.append(f"unparseable name: {f['filename']}")
            continue
        ex, pid = int(m[1]), int(m[2])
        judges = scores.get(clip.upper(), [])
        if len(judges) != 3:
            problems.append(f"{clip}: {len(judges)} judge rows in the marking CSV")
        mean = lambda i: round(sum(j[i] for j in judges) / len(judges), 4) if judges else ""
        h = local_md5.get(f["filename"])
        rows.append({
            "clip": clip, "filename": f["filename"],
            "exercise": f"E{ex:02d}", "exercise_name": EXERCISES[ex],
            "participant": f"P{pid:02d}", "cohort": "expert" if pid <= EXPERT_MAX_ID else "nonexpert",
            "gender": m[5], "view": m[3], "variation": m[4],
            "variation_known": int(m[4] in KNOWN_VARIATIONS),
            "performance": f"E{ex:02d}_P{pid:02d}_V{m[4]}",
            "on_disk": int(h is not None),
            "local_size_mb": round(on_disk[f["filename"]].stat().st_size / 1e6, 2) if h else "",
            "local_md5": h or "",
            "duplicate_of": ";".join(n for n in local_names_of[h] if n != clip) if h else "",
            "dv_name_ok": "unknown" if h is None else ("yes" if h == f["md5"] else "no"),
            "dv_name_with_local_bytes": ";".join(dv_names_of.get(h, [])) if h and h != f["md5"] else "",
            "dv_size_mb": round(f["filesize"] / 1e6, 2), "dv_md5": f["md5"], "dv_file_id": f["id"],
            "score": round(sum(sum(j) for j in judges) / len(judges) / 7 * 100, 2) if judges else "",
            "pf1_zero_judges": sum(1 for j in judges if j[0] == 0) if judges else "",
            "pf2": mean(1), "pf4": mean(3), "cf1": mean(4), "cf2": mean(5), "cf3": mean(6),
            "stage2a": int(m[3] == "F" and m[4] == "FL"),
        })
        for k, j in enumerate(judges, 1):
            long_rows.append({"clip": clip, "judge": k, **dict(zip(ITEMS, j)), "total": round(sum(j), 2)})

    views = Counter(r["performance"] for r in rows)
    for r in rows:
        r["views_in_performance"] = views[r["performance"]]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("manifest.csv", rows), ("scores_long.csv", long_rows)):
        with open(args.out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    print(f"listing: Dataverse version {listing_label}")
    print(f"marking CSV md5: {md5_of(args.scores)}")
    print(f"clips in listing: {len(rows)} ({sum(r['dv_size_mb'] for r in rows) / 1000:.1f} GB), participants: {len({r['participant'] for r in rows})}")
    print("\nexercise  expert  nonexpert  on disk  Dataverse name ok / wrong")
    for ex in sorted(EXERCISES):
        sub = [r for r in rows if r["exercise"] == f"E{ex:02d}"]
        ok = Counter(r["dv_name_ok"] for r in sub)
        print(f"E{ex:02d}       {sum(r['cohort'] == 'expert' for r in sub):6d}  {sum(r['cohort'] == 'nonexpert' for r in sub):9d}  "
              f"{sum(r['on_disk'] for r in sub):7d}  {ok['yes']:5d} / {ok['no']}")
    print("\nto check by hand:")
    for p in problems:
        print("  ", p)
    for fix in name_fixes:
        print("   marking CSV name normalised:", fix)
    for name in sorted(set(on_disk) - {r["filename"] for r in rows}):
        print("   on disk but not in the listing:", name)
    for r in rows:
        if not r["variation_known"]:
            print(f"   unknown variation code: {r['clip']}")
        if r["duplicate_of"] and r["clip"] < r["duplicate_of"]:
            print(f"   byte-identical clips on disk: {r['clip']} == {r['duplicate_of']}")
    flagged = [r for r in rows if r["pf1_zero_judges"] == 3]
    print(f"   clips all three judges marked 'not the right exercise' (PF1 = 0): {len(flagged)}; "
          + ", ".join(f"{k}:{v}" for k, v in sorted(Counter(r["exercise"] for r in flagged).items())))
    print(f"\nwrote {args.out / 'manifest.csv'} and {args.out / 'scores_long.csv'}")


if __name__ == "__main__":
    main()
