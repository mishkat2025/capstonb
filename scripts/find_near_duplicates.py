"""Look for clips that show the same recording under different names.

Byte-identical files are caught by MD5 in the manifest. A re-encoded copy has different
bytes but nearly the same landmark track, so this compares the tracks: two clips with the
same number of frames (within two) whose shoulder, elbow, wrist and hip positions correlate
above 0.999 are reported.

    .venv/Scripts/python scripts/find_near_duplicates.py

Works on the Stage 1 pose files (MediaPipe's own point numbering).
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOINTS = [11, 12, 13, 14, 15, 16, 23, 24]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="full_p1")
    ap.add_argument("--threshold", type=float, default=0.999)
    args = ap.parse_args()

    tracks = {}
    for path in sorted((ROOT / "data" / "pose" / args.run).glob("*.npz")):
        xy = np.load(path)["lm"][:, 0, JOINTS, :2].reshape(-1, 2 * len(JOINTS))
        if len(xy) >= 10 and np.isfinite(xy).mean() > 0.9:
            tracks[path.stem] = np.nan_to_num(xy - np.nanmean(xy, axis=0))
    names = sorted(tracks, key=lambda n: len(tracks[n]))
    found = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if len(tracks[b]) - len(tracks[a]) > 2:
                break
            n = min(len(tracks[a]), len(tracks[b]))
            x, y = tracks[a][:n].ravel(), tracks[b][:n].ravel()
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            if denom and float(x @ y) / denom > args.threshold:
                found += 1
                print(f"{a}  ~  {b}   frames {len(tracks[a])}/{len(tracks[b])}")
    print(f"{len(tracks)} clips compared, {found} near-identical pairs")


if __name__ == "__main__":
    main()
