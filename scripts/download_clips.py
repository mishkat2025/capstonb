"""Download a chosen subset of MobiPhysio clips from Harvard Dataverse (standard library only).

WARNING: Dataverse version 3.0 serves the wrong video under 370 of the 558 E01 filenames
(checked against the copy supplied by the dataset authors). Whether the other exercises are
affected is unknown. Prefer the authors' copy. This script refuses names known to be wrong
and needs --accept-unverified-names for names that have not been checked; anything fetched
that way must be compared with the authors' copy before it is used.

Selects rows of data/meta/manifest.csv with --where filters, shows the count and size, and
downloads only when --go is given. Each clip is written to <name>.part, checked against the
Dataverse MD5, then renamed; clips already present with the right size are skipped.

    python scripts/download_clips.py --out D:/mobiphysio/videos --where stage2a=1 exercise!=E01
"""
import argparse
import csv
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCESS_URL = "https://dataverse.harvard.edu/api/access/datafile/"
RETRY_WAITS = (5, 20, 60)


def select(rows, filters):
    for f in filters:
        negate = "!=" in f
        col, _, vals = f.partition("!=" if negate else "=")
        if not vals or col not in rows[0]:
            sys.exit(f"bad filter {f!r}: use column=value[,value] or column!=value[,value]; columns: {', '.join(rows[0])}")
        wanted = set(vals.split(","))
        rows = [r for r in rows if (r[col] in wanted) != negate]
    return rows


def fetch(url, part, expected_md5):
    h = hashlib.md5()
    req = urllib.request.Request(url, headers={"User-Agent": "mobiphysio-subset-download/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as out:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            out.write(chunk)
            h.update(chunk)
    if h.hexdigest() != expected_md5:
        part.unlink()
        raise ValueError(f"MD5 mismatch (got {h.hexdigest()}, manifest says {expected_md5})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "meta" / "manifest.csv")
    ap.add_argument("--out", type=Path, required=True, help="folder to download into")
    ap.add_argument("--where", nargs="*", default=[], metavar="COL=VAL", help="filters on manifest columns, combined with AND")
    ap.add_argument("--accept-unverified-names", action="store_true", help="also fetch names not yet checked against the authors' copy")
    ap.add_argument("--go", action="store_true", help="actually download (default: only show what would be fetched)")
    ap.add_argument("--base-url", default=ACCESS_URL, help=argparse.SUPPRESS)
    args = ap.parse_args()

    rows = select(list(csv.DictReader(open(args.manifest, encoding="utf-8"))), args.where)
    wrong = [r for r in rows if r["dv_name_ok"] == "no"]
    unverified = [r for r in rows if r["dv_name_ok"] == "unknown"]
    if wrong:
        print(f"skipping {len(wrong)} clips whose Dataverse name is known to carry another clip's video")
    if unverified and not args.accept_unverified_names:
        sys.exit(f"{len(unverified)} selected clips have names not checked against the authors' copy; "
                 "add --accept-unverified-names to fetch them anyway")
    rows = [r for r in rows if r["dv_name_ok"] != "no"]

    size = lambda sub: sum(float(r["dv_size_mb"]) for r in sub) / 1000
    todo = [r for r in rows
            if not ((args.out / r["filename"]).exists() and abs((args.out / r["filename"]).stat().st_size / 1e6 - float(r["dv_size_mb"])) < 0.01)]
    print(f"selected {len(rows)} clips ({size(rows):.1f} GB); {len(rows) - len(todo)} already in {args.out}; "
          f"to download: {len(todo)} ({size(todo):.1f} GB)")
    if not args.go:
        print("dry run - add --go to download")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, r in enumerate(todo, 1):
        target = args.out / r["filename"]
        for wait in (0,) + RETRY_WAITS:
            time.sleep(wait)
            try:
                fetch(args.base_url + r["dv_file_id"], target.with_name(target.name + ".part"), r["dv_md5"])
                target.with_name(target.name + ".part").replace(target)
                print(f"[{i}/{len(todo)}] {r['filename']} ({r['dv_size_mb']} MB)")
                break
            except (urllib.error.URLError, OSError, ValueError) as e:
                error = e
        else:
            failed.append(r["filename"])
            print(f"[{i}/{len(todo)}] FAILED {r['filename']}: {error}")
        time.sleep(0.5)
    print(f"done: {len(todo) - len(failed)} downloaded, {len(failed)} failed" + (": " + " ".join(failed) if failed else ""))


if __name__ == "__main__":
    main()
