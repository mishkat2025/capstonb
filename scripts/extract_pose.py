"""Run MediaPipe Pose Landmarker over many clips and store raw landmarks, one .npz per clip.

    .venv/Scripts/python scripts/extract_pose.py --videos data --exercise E01
    .venv/Scripts/python scripts/extract_pose.py --videos data --exercise E01 --view F

This is the Stage 1 batch extractor: MediaPipe only, many clips in parallel on the CPU.
To try any of the five pose models on one video, use demo.py instead. The default model
is Full; the Lite and Heavy files are not in models/mediapipe/ (Lite is in the archive).

Output: data/pose/<model>_p<num_poses>/<clip>.npz with
    t_ms    [T]            container timestamp of each frame
    lm      [T, P, 33, 5]  x, y (fraction of width/height), z, visibility, presence; NaN where no pose
    world   [T, P, 33, 3]  metres, origin at the hip centre
    width, height, fps     as decoded (rotation tag applied)
and run.json describing the model file, library versions and options.

Several models can be given at once. Each clip is then decoded a single time and every
frame is passed to each model, which saves the second decode; the models do not see each
other's results. Video mode keeps tracking state between frames, so every clip gets its
own landmarker per model. Clips of lockbox participants are skipped unless
--include-lockbox is given.

Speed notes for this machine (measured 2026-10-12): the Windows build of MediaPipe has no
GPU inference, and GPU video decoding is slower than CPU decoding here, so everything runs
on the CPU. Throughput stops improving beyond about 12 worker processes.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from contextlib import ExitStack
from multiprocessing import Pool
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]


def process_clip(job):
    path, num_poses, targets = job          # targets: [(model file, output file), ...]
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    t_ms = []
    lm = [[] for _ in targets]
    world = [[] for _ in targets]
    last_ts, width, height = -1, 0, 0
    with ExitStack() as stack:
        landmarkers = [stack.enter_context(vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_poses=num_poses))) for model_path, _ in targets]
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = max(last_ts + 1, int(round(t)))  # video mode needs strictly increasing integer timestamps
            last_ts = ts
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t_ms.append(t)
            for k, landmarker in enumerate(landmarkers):
                result = landmarker.detect_for_video(image, ts)
                f_lm = np.full((num_poses, 33, 5), np.nan, np.float32)
                f_world = np.full((num_poses, 33, 3), np.nan, np.float32)
                for p, (pose, pose_world) in enumerate(zip(result.pose_landmarks, result.pose_world_landmarks)):
                    f_lm[p] = [(q.x, q.y, q.z, q.visibility, q.presence) for q in pose]
                    f_world[p] = [(q.x, q.y, q.z) for q in pose_world]
                lm[k].append(f_lm)
                world[k].append(f_world)
    cap.release()
    n = len(t_ms)
    for k, (_, out_path) in enumerate(targets):
        tmp = out_path.with_name(out_path.stem + ".tmp.npz")
        np.savez_compressed(tmp, t_ms=np.asarray(t_ms, np.float64),
                            lm=np.asarray(lm[k], np.float32).reshape(n, num_poses, 33, 5),
                            world=np.asarray(world[k], np.float32).reshape(n, num_poses, 33, 3),
                            width=width, height=height, fps=fps)
        tmp.replace(out_path)
    return path.stem, n, len(targets), time.perf_counter() - started


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, required=True, help="folder searched recursively for clips")
    ap.add_argument("--model", nargs="+", choices=["lite", "full", "heavy"], default=["full"])
    ap.add_argument("--num-poses", type=int, default=1)
    ap.add_argument("--exercise", help="e.g. E01")
    ap.add_argument("--view", choices=["F", "L", "R"])
    ap.add_argument("--variation", help="e.g. O")
    ap.add_argument("--clips", nargs="*", help="explicit clip names (without .mp4)")
    ap.add_argument("--limit", type=int, help="process at most this many clips")
    ap.add_argument("--include-lockbox", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) * 3 // 4))
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "meta" / "manifest.csv")
    ap.add_argument("--splits", type=Path, default=ROOT / "data" / "meta" / "participant_splits.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "pose")
    args = ap.parse_args()

    models = {m: ROOT / "models" / "mediapipe" / f"pose_landmarker_{m}.task" for m in dict.fromkeys(args.model)}
    for m, path in models.items():
        if not path.exists():
            raise SystemExit(f"model file not found: {path}" + ("  (run scripts/download_models.py first)" if m == "full" else ""))
    out_dirs = {m: args.out / f"{m}_p{args.num_poses}" for m in models}
    for m, out_dir in out_dirs.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run.json").write_text(json.dumps({
            "model": models[m].name, "model_sha256": hashlib.sha256(models[m].read_bytes()).hexdigest(),
            "mediapipe": mp.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
            "running_mode": "VIDEO", "num_poses": args.num_poses,
            "min_pose_detection_confidence": 0.5, "min_pose_presence_confidence": 0.5, "min_tracking_confidence": 0.5,
            "frames": "every frame at native rate", "timestamps": "container PTS via OpenCV", "decode": "CPU (OpenCV FFmpeg)",
        }, indent=2), encoding="utf-8")

    split = {r["participant"]: r["split"] for r in csv.DictReader(open(args.splits, encoding="utf-8"))}
    on_disk = {p.stem: p for p in args.videos.rglob("*.mp4")}
    rows = [r for r in csv.DictReader(open(args.manifest, encoding="utf-8")) if r["clip"] in on_disk]
    if args.clips:
        rows = [r for r in rows if r["clip"] in set(args.clips)]
    else:
        rows = [r for r in rows
                if (not args.exercise or r["exercise"] == args.exercise)
                and (not args.view or r["view"] == args.view)
                and (not args.variation or r["variation"] == args.variation)
                and (args.include_lockbox or split[r["participant"]] == "dev")]

    jobs = []
    for r in rows:      # a clip is run only for the models that do not have it yet
        targets = [(models[m], out_dirs[m] / f"{r['clip']}.npz") for m in models if not (out_dirs[m] / f"{r['clip']}.npz").exists()]
        if targets:
            jobs.append((on_disk[r["clip"]], args.num_poses, targets))
    jobs = jobs[: args.limit]
    print(f"{len(rows)} clips selected, {len(rows) - len(jobs)} already done for {' + '.join(models)}, running {len(jobs)} with {args.workers} workers")

    started, frames = time.perf_counter(), 0
    with Pool(args.workers) as pool:
        for i, (clip, n, n_models, seconds) in enumerate(pool.imap_unordered(process_clip, jobs), 1):
            frames += n
            if i % 20 == 0 or i == len(jobs):
                elapsed = time.perf_counter() - started
                print(f"[{i}/{len(jobs)}] {frames} frames in {elapsed:.0f}s ({frames / elapsed:.0f} frames/s overall); last: {clip}, {n} frames x {n_models} model(s), {seconds:.1f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
