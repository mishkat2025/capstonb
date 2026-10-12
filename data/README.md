# Data

Nothing from the dataset is stored in this repository. The videos show participants'
faces, and the pose tracks, score sheets and per-clip tables are all made from them.
Only `meta/participant_splits.csv` is kept here, because the method depends on it: it
says which participants are development participants and which are held back for the
final test (the "lockbox").

## What goes here

| Path | What it is | Where it comes from |
|---|---|---|
| `E01(Abduction)/*.mp4` | The video clips of one exercise, named like `E01_P02_AF_VFL_GM.mp4` | The MobiPhysio dataset |
| `Physiotherapist Exercise Marking.csv` | The physiotherapists' marks per clip | The MobiPhysio dataset |
| `meta/manifest.csv`, `meta/scores_long.csv` | One row per clip, and one row per clip and judge | Written by `scripts/build_manifest.py` |
| `meta/participant_splits.csv` | Development and lockbox participants | Written by `scripts/make_splits.py`; kept in the repository |
| `pose/full_p1/*.npz` | MediaPipe Full landmarks for every frame of every development clip | Written by `scripts/extract_pose.py` |

## The dataset

MobiPhysio: smartphone videos of nine Active Range of Motion physiotherapy exercises
(E01 to E09), with marks from three physiotherapists.

- Paper: Data in Brief 65 (2026) 112635, https://doi.org/10.1016/j.dib.2026.112635
- Repository: Harvard Dataverse, https://doi.org/10.7910/DVN/XSI0QN

Two things found while preparing the data, both described in `docs/feasibility_plan.md`:

- On Dataverse (version 3.0), 370 of the 558 E01 file names carry a different E01 video
  than in the copy this project used. `scripts/build_manifest.py --videos data` compares
  a local copy with the Dataverse listing, and `scripts/download_clips.py` refuses names
  that have not been checked.
- The marking sheet on Dataverse covers the expert clips only. This project used a fuller
  sheet that also marks the non-expert clips.

## Rules

- Eleven participants are held back for the final test. Their clips are not analysed or
  viewed until then; the scripts refuse them unless told otherwise.
- Videos, and every image or report made from them, stay on the computer that holds the data.
