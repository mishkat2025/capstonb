"""Shared code for comparing pose estimation models on MobiPhysio clips.

    skeleton     one joint vocabulary for every model
    estimators   one file per model, all with the same interface
    subject      which detected person is the participant
    video        frame-by-frame reading with timestamps
    store        pose files on disk
    analysis     angles, frame quality flags, repetitions, clip features
    draw         skeleton overlays and the live view
    report       figures and the HTML report for one video
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
