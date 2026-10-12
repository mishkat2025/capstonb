"""Where each model's weight files come from, and fetching the ones that are missing.

The weights are not kept in the repository: they are large, and each has its own licence.
models/sources.json records every file this project was developed with: its source, its
licence, its size and its SHA-256.

    weights.ensure("vitpose")              # called when a model is created; does nothing once its files are there
    python scripts/download_models.py      # everything at once (setup.bat runs this)

A file that has just been downloaded is compared with that record. A difference is
reported, not treated as an error: two of the addresses serve "the latest" version, which
the publisher may replace. OpenPose's body model is the exception. CMU publishes its
checksum, and a file that does not match it is not written.
"""
import hashlib
import io
import json
import tarfile
import urllib.request
import zipfile

from . import MODELS

RECORD = MODELS / "sources.json"
GOOGLE = "https://storage.googleapis.com/mediapipe-models/"
OPENMMLAB = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"

# kind "url": one file; "zip": the one .onnx inside a zip; "tar": named files out of a .tar.gz
SOURCES = {
    "mediapipe": {
        "licence": "Apache-2.0",
        "files": [{"kind": "url", "url": GOOGLE + "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
                   "to": "mediapipe/pose_landmarker_full.task"}],
    },
    "hands": {       # MediaPipe Hand Landmarker: 21 points per hand, used on a cut-out around each wrist
        "licence": "Apache-2.0",
        "files": [{"kind": "url", "url": GOOGLE + "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                   "to": "mediapipe/hand_landmarker.task"}],
    },
    "vitpose": {
        "licence": "Apache-2.0",
        # ViTPose Base trained on COCO-WholeBody (133 points, feet included), converted to ONNX by the
        # easy_ViTPose project from the ViTPose authors' checkpoint. A third-party conversion: no official checksum exists.
        "files": [{"kind": "url", "url": "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-b-wholebody.onnx",
                   "to": "vitpose/vitpose-b-wholebody.onnx"}],
    },
    "openpose": {    # CMU's original 25-point body model. CMU's own server is offline; this is a mirror of their files on Zenodo.
        "licence": "OpenPose licence: non-commercial research use only",
        "files": [{"kind": "tar", "url": "https://zenodo.org/api/records/18290485/files/openpose_models_minimal.tar.gz/content",
                   "members": {"openpose_models/pose/body_25/pose_iter_584000.caffemodel": "openpose/body_25/pose_iter_584000.caffemodel",
                               "openpose_models/pose/body_25/pose_deploy.prototxt": "openpose/body_25/pose_deploy.prototxt"},
                   # the checksum CMU lists for this file in the CMakeLists.txt of its OpenPose repository
                   "md5": {"openpose/body_25/pose_iter_584000.caffemodel": "78287b57cf85fa89c03f1393d368e5b7"}}],
    },
    "rtmpose": {
        "licence": "Apache-2.0",
        "files": [{"kind": "zip", "url": OPENMMLAB + "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
                   "to": "rtmpose/rtmpose-m_halpe26_256x192.onnx"}],
    },
    "person_detector": {     # YOLOX-m, shared by the two top-down models
        "licence": "Apache-2.0",
        "files": [{"kind": "zip", "url": OPENMMLAB + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
                   "to": "person_detector/yolox-m_humanart_640.onnx"}],
    },
    "yolo": {
        "licence": "AGPL-3.0",
        "files": [{"kind": "url", "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt",
                   "to": "yolo/yolo26x-pose.pt"}],
    },
}
# what each model of the demo needs (keys as in estimators.REGISTRY, plus "hands")
NEEDS = {"mediapipe": ["mediapipe"], "vitpose": ["vitpose", "person_detector"], "openpose": ["openpose"],
         "rtmpose": ["rtmpose", "person_detector"], "yolo": ["yolo"], "hands": ["hands"]}


def targets(item):
    """The files one entry provides, relative to models/."""
    return list(item["members"].values()) if item["kind"] == "tar" else [item["to"]]


def files_of(name):
    return [t for item in SOURCES[name]["files"] for t in targets(item)]


def missing(name):
    return [f for f in files_of(name) if not (MODELS / f).exists()]


def recorded():
    """models/sources.json as {file: {"bytes": ..., "sha256": ...}}; empty when there is no record."""
    if not RECORD.exists():
        return {}
    return {f: info for source in json.loads(RECORD.read_text(encoding="utf-8")).values() for f, info in source["files"].items()}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, announce):
    request = urllib.request.Request(url, headers={"User-Agent": "poselab-setup"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        buffer, done, shown = io.BytesIO(), 0, -1
        while chunk := response.read(1 << 20):
            buffer.write(chunk)
            done += len(chunk)
            if total and 10 * done // total != shown:
                shown = 10 * done // total
                announce(f"    {10 * shown:3d}%  of {total / 1e6:.0f} MB")
        return buffer.getvalue()


def get(item, announce=print):
    """Download one entry unless its files are already there."""
    if all((MODELS / t).exists() for t in targets(item)):
        return
    announce(f"  {item['url']}")
    data = download(item["url"], announce)
    if item["kind"] == "tar":       # a bundle: only the named files are taken out of it, and checked where a checksum is published
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as bundle:
            for inside, to in item["members"].items():
                content = bundle.extractfile(bundle.getmember(inside)).read()
                expected = item.get("md5", {}).get(to)
                if expected and hashlib.md5(content).hexdigest() != expected:
                    raise SystemExit(f"{inside}: checksum does not match the published one; nothing was written")
                (MODELS / to).parent.mkdir(parents=True, exist_ok=True)
                (MODELS / to).write_bytes(content)
        return
    if item["kind"] == "zip":       # the archive holds one end2end.onnx plus deployment notes; only the model is kept
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            inside = [n for n in archive.namelist() if n.endswith(".onnx")]
            if len(inside) != 1:
                raise SystemExit(f"expected one .onnx in {item['url']}, found {inside}")
            data = archive.read(inside[0])
    target = MODELS / item["to"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def fetch(name, announce=print):
    """Download the missing files of one source and compare the new ones with models/sources.json."""
    wanted = missing(name)
    if not wanted:
        return
    known = recorded()
    size = sum(known.get(f, {}).get("bytes", 0) for f in wanted)
    announce(f"Downloading the {name} weights" + (f" (about {size / 1e6:.0f} MB)" if size else "") + f". Licence: {SOURCES[name]['licence']}")
    for item in SOURCES[name]["files"]:
        get(item, announce)
    for f in wanted:
        if f in known and sha256(MODELS / f) != known[f]["sha256"]:
            announce(f"  note: {f} is not the same file this project was developed with (see models/sources.json); the publisher may have replaced it")


def ensure(key, announce=print):
    """Make sure the weights one model of the demo needs are on disk. key: a model key, or "hands"."""
    for name in NEEDS[key]:
        fetch(name, announce)


def write_record(names=None):
    """Rewrite the entries of models/sources.json for the named sources (default: all) from the files on disk."""
    record = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}
    for name in names or SOURCES:
        record[name] = {"licence": SOURCES[name]["licence"], "from": [item["url"] for item in SOURCES[name]["files"]],
                        "files": {f: {"bytes": (MODELS / f).stat().st_size, "sha256": sha256(MODELS / f)}
                                  for f in files_of(name) if (MODELS / f).exists()}}
    record = {name: record[name] for name in SOURCES if name in record}      # the order of SOURCES; entries of retired sources are dropped
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
