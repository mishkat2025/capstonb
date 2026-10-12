"""Write one contact sheet per clip: six evenly spaced frames side by side.

Frames are decoded exactly as the pose step decodes them, so a person lying sideways
on a sheet means the rotation tag is not being applied.

    .venv/Scripts/python scripts/contact_sheets.py --videos "data/E01(Abduction)" --view F
    .venv/Scripts/python scripts/contact_sheets.py --videos "data/E01(Abduction)" --clips E01_P01_AF_VFL_GM E01_P26_AF_VFL_GF

Lockbox participants are skipped unless a clip is named explicitly.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_FRAMES, TILE_HEIGHT = 6, 320


def sheet(path):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for k in range(N_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / N_FRAMES))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        tiles.append(cv2.resize(frame, (max(1, round(w * TILE_HEIGHT / h)), TILE_HEIGHT), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.hstack(tiles) if tiles else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "stage1" / "contact_sheets")
    ap.add_argument("--splits", type=Path, default=ROOT / "data" / "meta" / "participant_splits.csv")
    ap.add_argument("--view", choices=["F", "L", "R"], help="only clips of this view")
    ap.add_argument("--clips", nargs="*", help="explicit clip names (without .mp4)")
    args = ap.parse_args()

    dev = {r["participant"] for r in csv.DictReader(open(args.splits, encoding="utf-8")) if r["split"] == "dev"}
    paths = sorted(args.videos.rglob("*.mp4"))
    if args.clips:
        paths = [p for p in paths if p.stem in set(args.clips)]
    else:
        paths = [p for p in paths if p.stem.split("_")[1] in dev and (not args.view or p.stem.split("_")[2] == "A" + args.view)]

    args.out.mkdir(parents=True, exist_ok=True)
    done = 0
    for p in paths:
        img = sheet(p)
        if img is not None:
            cv2.imwrite(str(args.out / f"{p.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            done += 1
    print(f"wrote {done} of {len(paths)} contact sheets to {args.out}")


if __name__ == "__main__":
    main()
