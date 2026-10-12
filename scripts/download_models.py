"""Fetch the model weights into models/, one folder per model (about 0.9 GB to download, 0.77 GB kept).

    .venv/Scripts/python scripts/download_models.py            # everything that is missing
    .venv/Scripts/python scripts/download_models.py --only vitpose yolo
    .venv/Scripts/python scripts/download_models.py --list     # what is there and what is missing; downloads nothing

Files that are already present are left alone. The sources, the licences and the
comparison with models/sources.json are in poselab/weights.py. The demo calls the same
code for any model whose files are missing, so running this first is optional.

--record rewrites models/sources.json from the files on disk. It is for whoever changes
which model files the project uses, not for setting up a copy.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poselab import weights


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(weights.SOURCES), help="sources to fetch (default: all)")
    ap.add_argument("--list", action="store_true", help="show what is present and what is missing, and stop")
    ap.add_argument("--record", action="store_true", help="rewrite models/sources.json from the files on disk")
    args = ap.parse_args()

    names = args.only or list(weights.SOURCES)
    known = weights.recorded()
    if not args.list:
        for name in names:
            weights.fetch(name)
    for name in names:
        print(f"{name}  ({weights.SOURCES[name]['licence']})")
        for f in weights.files_of(name):
            path = weights.MODELS / f
            size = path.stat().st_size if path.exists() else known.get(f, {}).get("bytes", 0)
            print(f"  {'present' if path.exists() else 'MISSING'}  {f}  {size / 1e6:.1f} MB")
    if args.record:
        weights.write_record(args.only)
        print("wrote", weights.RECORD)


if __name__ == "__main__":
    main()
