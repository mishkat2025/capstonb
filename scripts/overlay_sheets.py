"""Contact sheets with the detected skeleton drawn on the frames (Stage 1 pose files).

    .venv/Scripts/python scripts/overlay_sheets.py --videos data --exercise E01
    .venv/Scripts/python scripts/overlay_sheets.py --videos data --clips E01_P02_AF_VO_GM

Without --clips it takes two clips per cohort, view and variation plus the ten clips with
the lowest share of usable frames (read from outputs/stage1/<run>/qc_clips.csv, so run
stage1_tables.py first). --same-as reuses the clips another run already has sheets for.
The subject's right side is drawn orange, the left side blue; frames that fail a quality
check get a red border and the names of the flags, and frames whose left/right labels were
exchanged back are marked "left/right relabelled". The face and finger points are in the
pose files but not drawn: no Stage 1 measurement uses them.
"""
import argparse
import csv
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import posekit as pk
from poselab.draw import skeleton

ROOT = Path(__file__).resolve().parents[1]
N_FRAMES, TILE_HEIGHT = 6, 480


def shrink(frame):
    scale = TILE_HEIGHT / frame.shape[0]
    return cv2.resize(frame, (round(frame.shape[1] * scale), TILE_HEIGHT), interpolation=cv2.INTER_AREA), scale


def grab_tiles(video, t_seconds):
    """Return {frame index: (resized frame, scale)} for six evenly spaced frames, and whether seeking was used.

    t_seconds holds the timestamp of every frame as recorded during pose extraction. The
    fast path jumps straight to each wanted frame and accepts it only if its timestamp is
    the recorded one. Seeking is not exact in every file, so on any mismatch the clip is
    decoded from the start instead, which is slow but always lands on the right frames.
    """
    targets = sorted({int((k + 0.5) * len(t_seconds) / N_FRAMES) for k in range(N_FRAMES)})
    cap = cv2.VideoCapture(str(video))
    tiles = {}
    for index in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or abs(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 - t_seconds[index]) > 0.0005:
            tiles = None
            break
        tiles[index] = shrink(frame)
    cap.release()
    if tiles is not None:
        return tiles, True

    cap = cv2.VideoCapture(str(video))
    tiles, index = {}, 0
    while len(tiles) < len(targets):
        ok, frame = cap.read()
        if not ok:
            break
        if index in targets:
            tiles[index] = shrink(frame)
        index += 1
    cap.release()
    return tiles, False


def draw_row(tiles, analysis):
    """One row of tiles with the skeleton and quality flags of one pose run."""
    clip, flags, swapped = analysis["clip"], analysis["flags"], analysis["clip"]["swapped"]
    row = []
    for index in sorted(tiles):
        tile, scale = tiles[index]
        tile = tile.copy()
        skeleton(tile, clip["px"][index], clip["score"][index], clip["min_score"], scale)
        fired = [name for name, on in zip(pk.FLAGS, flags[index]) if on]
        if fired:
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (40, 40, 220), 4)
            cv2.putText(tile, ",".join(fired)[:40], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 220), 1, cv2.LINE_AA)
        if swapped[index]:
            cv2.putText(tile, "left/right relabelled", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 90, 20), 1, cv2.LINE_AA)
        row.append(tile)
    return np.hstack(row) if row else None


def default_selection(rows, run):
    seen, chosen = {}, []
    for r in rows:
        key = (r["cohort"], r["view"], r["variation"])
        if seen.setdefault(key, 0) < 2:
            seen[key] += 1
            chosen.append(r["clip"])
    qc = ROOT / "outputs" / "stage1" / run / "qc_clips.csv"
    if qc.exists():
        quality = sorted(csv.DictReader(open(qc, encoding="utf-8")), key=lambda q: float(q["usable_frac"]))
        chosen += [q["clip"] for q in quality[:10]]
    return chosen


def render(job):
    """Write the sheet for one clip. Runs in a worker process: decoding a whole clip is the slow part."""
    video, pose_file, front_view, target = job
    analysis = pk.analyse_clip(pose_file, front_view=front_view)
    tiles, seeked = grab_tiles(video, analysis["clip"]["t"])
    img = draw_row(tiles, analysis)
    if img is not None:
        cv2.imwrite(str(target), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return seeked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--run", default="full_p1")
    ap.add_argument("--exercise", default="E01")
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--same-as", help="use the clips that already have overlay sheets for this other run")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) * 3 // 4))
    ap.add_argument("--out", type=Path, help="output folder (default: outputs/stage1/<run>/overlays)")
    args = ap.parse_args()

    run_dir = ROOT / "data" / "pose" / args.run
    out = args.out or ROOT / "outputs" / "stage1" / args.run / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    videos = {p.stem: p for p in args.videos.rglob("*.mp4")}
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "meta" / "manifest.csv", encoding="utf-8"))
            if r["exercise"] == args.exercise and (run_dir / f"{r['clip']}.npz").exists()]
    view_of = {r["clip"]: r["view"] for r in rows}

    if args.clips:
        chosen = args.clips
    elif args.same_as:
        chosen = sorted(p.stem for p in (ROOT / "outputs" / "stage1" / args.same_as / "overlays").glob("*.jpg"))
    else:
        chosen = default_selection(rows, args.run)
    chosen = [c for c in dict.fromkeys(chosen) if c in view_of]

    jobs = [(videos[c], run_dir / f"{c}.npz", view_of[c] == "F", out / f"{c}.jpg") for c in chosen]
    with Pool(args.workers) as pool:
        done = list(pool.imap_unordered(render, jobs))
    print(f"wrote {len(done)} overlay sheets to {out} ({len(done) - sum(done)} needed a full decode)")


if __name__ == "__main__":
    main()
