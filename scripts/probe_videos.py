"""Read container metadata of every clip in a folder (no frames are decoded).

    .venv/Scripts/python scripts/probe_videos.py --videos "data/E01(Abduction)"

Writes outputs/stage1/video_probe.csv with one row per clip: resolution as displayed
(after the rotation tag is applied), nominal frame rate, frame count, duration, codec,
rotation tag and file size.
"""
import argparse
import csv
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def probe(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"filename": path.name, "opened": 0}
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    row = {
        "filename": path.name, "opened": 1,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(fps, 3), "frames": frames, "duration_s": round(frames / fps, 2) if fps else "",
        "codec": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)),
        "rotation_tag": int(cap.get(cv2.CAP_PROP_ORIENTATION_META)),
        "size_bytes": path.stat().st_size,
    }
    cap.release()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "stage1" / "video_probe.csv")
    args = ap.parse_args()

    rows = [probe(p) for p in sorted(args.videos.rglob("*.mp4"))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "opened", "width", "height", "fps", "frames", "duration_s", "codec", "rotation_tag", "size_bytes"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"probed {len(rows)} clips, {sum(1 for r in rows if not r['opened'])} could not be opened -> {args.out}")


if __name__ == "__main__":
    main()
