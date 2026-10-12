"""Pick a video, watch the pose models work on it, and get a report.

    run_demo.bat                          double-click: opens the launcher window
    .venv/Scripts/python demo.py          the same, from a terminal
    .venv/Scripts/python demo.py --video "data/E01(Abduction)/E01_P02_AF_VFL_GM.mp4" --models mediapipe rtmpose
    .venv/Scripts/python demo.py --video ... --no-window      no live window, the report is still written

While a video runs: Space pauses, Q or Esc stops early. Each run gets its own folder in
outputs/demo/ holding report.html, the figures, the recorded live view and the raw points.

Clips of the participants held back for the final test (the "lockbox", see
data/meta/participant_splits.csv) are refused, so that nobody looks at them before the
models are frozen. --include-lockbox overrides this on the command line.
"""
import argparse
import csv
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from poselab.estimators import LABELS, NOTES, REGISTRY

OUT = ROOT / "outputs" / "demo"
SPLITS = ROOT / "data" / "meta" / "participant_splits.csv"


def view_from_name(name):
    """MobiPhysio names carry the camera angle: _AF_ front, _AL_ / _AR_ side. None when the name does not say."""
    match = re.search(r"_A([FLR])_", name)
    return None if match is None else ("front" if match.group(1) == "F" else "side")


def lockbox_participant(path):
    """The participant id if this clip belongs to a held-back participant, else None."""
    match = re.match(r"E\d\d_(P\d\d)_", Path(path).name)
    if match is None or not SPLITS.exists():
        return None
    with open(SPLITS, encoding="utf-8") as f:
        split = {r["participant"]: r["split"] for r in csv.DictReader(f)}
    return match.group(1) if split.get(match.group(1)) == "lockbox" else None


def run_once(video_path, keys, view, loaded, announce=print, show=True, record=True, hands=True, max_frames=None, screen=(1500, 860)):
    """Process one video with the chosen models and write its report. loaded caches models (and the hand model) between runs."""
    from poselab import report, run
    from poselab.live import LiveView
    from poselab.video import Video

    video_path = Path(video_path)
    out_dir = OUT / f"{video_path.stem}__{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        if key not in loaded:
            loaded.update(run.load_models([key], announce))
    models = {key: loaded[key] for key in keys}
    if hands and "hands" not in loaded:
        from poselab.hands import HandReader
        loaded["hands"] = HandReader(announce=announce)
    hand_reader = loaded["hands"] if hands else None

    announce("Warming up the models on the first frame ...")
    with Video(video_path) as probe:
        view_window = LiveView(models, probe, out_dir, show=show, record=record, screen=screen, with_hands=hands)
    try:
        result = run.process(video_path, models, on_frame=view_window, max_frames=max_frames, hand_reader=hand_reader,
                             front_view=view == "front")
    finally:
        view_window.close()
    announce(f"Processed {len(result.t_ms)} frames in {result.elapsed_s:.0f} s. Writing the report ...")
    return report.build(result, view, out_dir, samples=view_window.samples, hand_samples=view_window.hand_samples,
                        live_video=view_window.video_file, codec=view_window.codec)


def launcher():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("MobiPhysio pose comparison")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=16)
    frame.grid()
    loaded = {}

    video = tk.StringVar()
    view = tk.StringVar(value="auto")
    status = tk.StringVar(value="Choose a video and press Start.")
    chosen = {key: tk.BooleanVar(value=True) for key in REGISTRY}
    record = tk.BooleanVar(value=True)
    with_hands = tk.BooleanVar(value=True)

    ttk.Label(frame, text="Video", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
    ttk.Entry(frame, textvariable=video, width=70, state="readonly").grid(row=1, column=0, sticky="we", pady=(2, 10))

    def choose():
        start = ROOT / "data"
        path = filedialog.askopenfilename(title="Choose a video", initialdir=start if start.exists() else ROOT,
                                          filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")])
        if path:
            video.set(path)
            guess = view_from_name(Path(path).name)
            status.set(f"Camera view read from the file name: {guess}." if guess else "The file name does not say the camera view: choose it below.")
    ttk.Button(frame, text="Choose...", command=choose).grid(row=1, column=1, padx=(8, 0), pady=(2, 10))

    ttk.Label(frame, text="Models", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w")
    for n, key in enumerate(REGISTRY):
        ttk.Checkbutton(frame, text=f"{LABELS[key]}   ({NOTES[key]})", variable=chosen[key]).grid(row=3 + n, column=0, columnspan=2, sticky="w")
    base = 3 + len(REGISTRY)

    ttk.Label(frame, text="Camera view", font=("Segoe UI", 10, "bold")).grid(row=base, column=0, sticky="w", pady=(10, 0))
    views = ttk.Frame(frame)
    views.grid(row=base + 1, column=0, columnspan=2, sticky="w")
    for text, value in (("From the file name", "auto"), ("Front", "front"), ("Side", "side")):
        ttk.Radiobutton(views, text=text, variable=view, value=value).pack(side="left", padx=(0, 14))
    ttk.Checkbutton(frame, text="Hands: find the fingers and say whether each hand is open or closed (works with any model)",
                    variable=with_hands).grid(row=base + 2, column=0, columnspan=2, sticky="w", pady=(10, 0))
    ttk.Checkbutton(frame, text="Record the live view into the report (about 1 MB per second of video)", variable=record).grid(
        row=base + 3, column=0, columnspan=2, sticky="w")
    base += 1

    def say(message):
        status.set(message)
        root.update()

    def start():
        path = video.get()
        keys = [key for key in REGISTRY if chosen[key].get()]
        if not path:
            return messagebox.showinfo("Pose comparison", "Choose a video first.")
        if not keys:
            return messagebox.showinfo("Pose comparison", "Tick at least one model.")
        held_back = lockbox_participant(path)
        if held_back:
            return messagebox.showwarning("Pose comparison", f"{held_back} is one of the participants held back for the final test.\n\n"
                                          "Their clips stay unseen until the models are frozen, so this video is not opened. Choose another one.")
        which = view.get() if view.get() != "auto" else view_from_name(Path(path).name)
        if which is None:
            return messagebox.showinfo("Pose comparison", "The file name does not say the camera view. Choose Front or Side.")
        button.state(["disabled"])
        try:
            screen = (min(root.winfo_screenwidth() - 60, 1800), min(root.winfo_screenheight() - 140, 980))
            report_path = run_once(path, keys, which, loaded, announce=say, record=record.get(), hands=with_hands.get(), screen=screen)
            os.startfile(report_path)
            say(f"Report opened: {report_path.relative_to(ROOT)}")
        except Exception as error:
            traceback.print_exc()
            messagebox.showerror("Pose comparison", f"The run failed:\n\n{error}")
            say("The run failed. Details are in the terminal window.")
        finally:
            button.state(["!disabled"])

    button = ttk.Button(frame, text="Start", command=start)
    button.grid(row=base + 3, column=0, sticky="w", pady=(14, 6))
    ttk.Button(frame, text="Open reports folder", command=lambda: (OUT.mkdir(parents=True, exist_ok=True), os.startfile(OUT))).grid(
        row=base + 3, column=1, sticky="e", pady=(14, 6))
    ttk.Label(frame, textvariable=status, wraplength=560, foreground="#52514e").grid(row=base + 4, column=0, columnspan=2, sticky="w")
    ttk.Label(frame, text="While a video runs: Space pauses, Q stops early.", foreground="#898781").grid(row=base + 5, column=0, columnspan=2, sticky="w")
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, help="without this, the launcher window opens")
    ap.add_argument("--models", nargs="+", choices=list(REGISTRY), default=list(REGISTRY))
    ap.add_argument("--view", choices=["front", "side"], help="camera view; read from the file name when it has _AF_, _AL_ or _AR_")
    ap.add_argument("--no-window", action="store_true", help="process without showing the live view")
    ap.add_argument("--no-record", action="store_true", help="do not save a recording of the live view")
    ap.add_argument("--no-hands", action="store_true", help="skip the hand model (fingers, open or closed)")
    ap.add_argument("--max-frames", type=int, help="stop after this many frames")
    ap.add_argument("--include-lockbox", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="do not open the report when it is ready")
    args = ap.parse_args()

    if args.video is None:
        return launcher()
    if not args.video.exists():
        sys.exit(f"no such file: {args.video}")
    held_back = lockbox_participant(args.video)
    if held_back and not args.include_lockbox:
        sys.exit(f"{held_back} is a held-back (lockbox) participant; their clips stay unseen until the final test.")
    view = args.view or view_from_name(args.video.name)
    if view is None:
        sys.exit("the file name does not say the camera view: add --view front or --view side")
    started = time.perf_counter()
    report_path = run_once(args.video, list(dict.fromkeys(args.models)), view, {}, show=not args.no_window, record=not args.no_record,
                           hands=not args.no_hands, max_frames=args.max_frames)
    print(f"report: {report_path}  ({time.perf_counter() - started:.0f} s in total)")
    if not args.no_open:
        os.startfile(report_path)


if __name__ == "__main__":
    main()
