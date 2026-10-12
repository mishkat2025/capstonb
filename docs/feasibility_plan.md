# MobiPhysio: E01 feasibility plan

Written 2026-10-10. Based on the Data in Brief paper (65, 2026, 112635), the Harvard Dataverse file listing (version 3.0, released 2026-02-11), the repository README, and the marking CSV you supplied (658,975 bytes, MD5 `4eb7e9c9d98b3f91082d76eaa1b06b9f`).

The plan was written before any video was available. No accuracy, speed or clinical claim is made anywhere in it.

## Update, 2026-10-12: Full only, four more pose models, and a pick-a-video demo

- **MediaPipe Lite is dropped.** The project continues with MediaPipe Full. Lite files and the Lite-against-Full comparison are in `archive/2026-10-12_lite/`. The "model size" criterion of Stage 1 (Section 2) is therefore closed as measured, not carried forward.
- **Four more pose models were added for comparison**: ViTPose Base, OpenPose, RTMPose-m and YOLO26x-pose. The plan treated MediaPipe as a hypothesis to test; these give it something to be tested against. `docs/pose_models.md` describes them and what was measured.
- **`demo.py` runs any of the five on one video**, shows them live side by side and writes a report. `README.md` explains how to use it.
- **The flag `low_visibility` (Section 2) is now named `low_confidence`** in code and tables, because the other models report a confidence score, not a visibility. The check is unchanged.
- **The exploratory "plane of elevation" feature of Section 3 is now computed** from MediaPipe's 3D points (`poselab/space.py`) and shown in the demo. On front-view abduction clips it reads "sideways" as it should, and it singled out the one performance the judges rejected. It has not been tested on a movement that should read differently; circumduction (E05) is the first such test. Details: `docs/pose_models.md`, Section 8.
- **Not done yet:** running the four new models over all development clips. Until then, statements about them rest on the few clips tried in the demo.

## Update, 2026-10-11: E01 is on disk, and Dataverse names are wrong for E01

The E01 copy supplied to you (558 clips in `data/E01(Abduction)`) was compared with Dataverse byte by byte.

- **Every local file exists on Dataverse, but 370 of the 558 names point at different videos there.** The pattern is an off-by-one: with the E01 videos sorted by size, each of those 370 names carries on Dataverse the video one place larger than the one it carries in your copy. They are the 370 smallest apart from the very smallest. The other 188 names agree.
- **Your copy has the correct names.** Under your names the three views of an expert performance differ in length by a median of 0.9 s (7.2 s under Dataverse's names), expert and non-expert clips separate cleanly by frame rate and duration, and four clips checked by eye match your names for gender, view, lighting and occlusion and contradict Dataverse's.
- **Consequences.** Use only the authors' copy. Do not fetch E01 from Dataverse. Whether E02–E09 are affected is unknown until your copy of each can be compared; `build_manifest.py --videos data` does that comparison and `download_clips.py` now refuses unchecked names unless told otherwise. The duplicate reported earlier (P02 and P06) exists only on Dataverse. Your copy has a different one: `E01_P60_AF_VLR_GF` and `E01_P60_AF_VML_GF` are byte-identical.

What the clips contain, from container metadata under your names (all 558 clips):

| | Expert (382 clips) | Non-expert (176 clips) |
|---|---|---|
| Duration | 11.5–54.1 s, median 30.9 s | 1.1–7.2 s, median 3.0 s |
| Frame rate | 30 fps (16 clips at 60 fps) | 25 fps for 147 clips, 24–30 fps for the other 29 |
| Resolution | 1080×1920 portrait (2 clips at 2160×3840) | 1920×1080 (101), 1080×1920 (48), 1440×1440 (11), 720×1280 (8), 1280×720 (6), 352×640 (2) |
| Codec | H.264 (254), HEVC (128) | H.264 |
| Content | A set of repetitions | A single raise, sometimes with part of another |

So the paper's "1080p at 30 fps" holds for experts only, and a non-expert clip is one repetition while an expert clip is about ten. Stage 1 results are in `outputs/stage1/`.

## Summary

1. **The paper's counts are right, but the structure behind them drives the whole design.** The dataset is two cohorts recorded differently. Experts (P01–P24) contribute 3,011 clips, all nine exercises, three views, 153 GB. Non-experts (P26–P62) contribute 675 clips, only five exercises, mostly front view, 6 GB. A pooled result can be explained by cohort unless the experiment is built to rule that out.
2. **An E01-only experiment can validate pose extraction and the movement signal. It cannot measure detection or multiclass accuracy**, because both need clips that are not E01.
3. **Negatives exist and come from the same people**, so a binary baseline is sound. Its first pass needs only the front-view, full-light clips of each class (266 clips, 5.1 GB), taken from the authors' copy rather than Dataverse (see the update below).
4. **The 0–100 score is not a movement-quality label as it stands.** It mostly records which cohort a clip belongs to and how many repetitions the clip contains. Section 7 gives the evidence and what can still be done with it.
5. **Four data defects need handling before any modelling**: Dataverse serves the wrong video under 370 E01 names, one clip in your copy is stored twice (`E01_P60_AF_VLR_GF` and `E01_P60_AF_VML_GF`), all three physiotherapists marked 242 clips "not the right exercise", and the public CSV lacks the non-expert scores.
6. **Next step:** read the Stage 1 results in `outputs/stage1/`, then obtain the other exercises from the same source as E01.

Files produced so far:

| File | Content |
|---|---|
| `data/meta/manifest.csv` | One row per clip: exercise, participant, cohort, view, variation, size, MD5, Dataverse file id, score fields, defect flags |
| `data/meta/scores_long.csv` | One row per clip and judge: the seven questionnaire items |
| `data/meta/participant_splits.csv` | Lockbox and five development folds, by participant |
| `scripts/build_manifest.py`, `make_splits.py`, `download_clips.py` | Regenerate the above; fetch a chosen subset of clips |
| `docs/stage1_report.md` | Stage 1 results and how they were obtained (added 2026-10-11) |
| `scripts/probe_videos.py`, `extract_pose.py`, `posekit.py`, `stage1_tables.py`, `stage1_figures.py`, `overlay_sheets.py`, `contact_sheets.py`, `make_watch_log.py` | The Stage 1 pipeline (MediaPipe Full since 2026-10-12) |
| `demo.py`, `run_demo.bat`, `poselab/`, `scripts/download_models.py`, `setup.bat` | The five-model comparison: pick a video, watch, get a report (added 2026-10-12); `setup.bat` installs the packages and the model weights in one step |
| `docs/pose_models.md`, `README.md` | The five models and how to use the demo (added 2026-10-12) |

## Terms

- **Clip**: one `.mp4` file.
- **Performance**: one participant doing one exercise under one variation. Experts usually have three clips per performance (front, left, right).
- **Cohort**: expert (P01–P24; physiotherapy students, according to the paper) or non-expert (P26–P62; described in the paper as computer science students with no physiotherapy training).
- **Item**: one of the seven questionnaire questions, PF1–PF4 (primary factors) and CF1–CF3 (control factors).
- **Usable frame**: a frame that passes every quality check in Stage 1.

## 1. What is established and what is not

### Paper claims checked against the files

| Paper claim | Status | Evidence |
|---|---|---|
| 3,686 clips; per-exercise counts (558, 384, 550, 383, 511, 354, 251, 356, 339) | Confirmed | Listing |
| 58 participants, 39 male and 19 female | Confirmed | Listing, README |
| Expert and non-expert clip counts per exercise (Table 1) | Confirmed, once cohort is taken as P01–P24 versus P26–P62 | Listing. The rule is not written anywhere; it reproduces Table 1 for all nine exercises, and the CSV writes judge names two ways ("Dr" and "Dr.") that split the clips the same way |
| Filename pattern `E01_P01_AF_VFL_GM.mp4` | Confirmed, with one exception: `E03_P41_AF_VP_GM.mp4` | Listing |
| No folders; README and score CSV in the repository | Confirmed | Listing |
| CSV holds scores "for all videos" | True of your copy. The public Dataverse CSV (MD5 `f517c958…`) holds only the 3,011 expert clips, one of them without a name | Both CSVs |
| Three judges, seven items, average scaled to 100 | Confirmed, and the CSV has every judge's score on every item | Your CSV |
| Judge correlations 0.958–0.971, mean absolute difference 0.208–0.282 | Reproduced exactly | Your CSV |
| ICC(3,k) = 0.6672 | Not reproduced. The standard two-way formula on the judges' totals gives 0.987 (single-rater 0.963) | Your CSV |
| Front 1,450 clips, left and right 1,100 each | Wrong in the text; the listing has 1,520, 1,087 and 1,079 | Listing |
| Phase I: 28 non-experts (25 male, 3 female) | The release has 34 non-expert IDs (23 male, 11 female), so it is not simply the Phase I set. It also has 60 non-expert occlusion clips, a variation the paper introduces only in Phase II | Listing |
| E08 is "Lumber Flexion" | README calls it "Both Leg Flexion" | README |
| 1080p at 30 fps | True for expert clips. Non-expert clips are mostly 25 fps and come in six resolutions (see the update above) | Clip metadata |

### What the files show that the paper does not say

| Finding | Why it matters |
|---|---|
| Non-experts recorded only E01, E03, E05, E08, E09 | For four classes there is no non-expert data at all; "cohort" partly predicts "class" |
| In the non-expert cohort, left and right views exist only under full light; every other variation is front only | View and variation are not crossed for non-experts |
| Low resolution exists only for non-experts (51 clips) | Any low-resolution result is also a cohort result |
| Expert median clip size differs by view (front 31 MB, left 71 MB, right 27 MB) | Suggests one phone per view, so view and device cannot be separated |
| E07, E08, E09 have no female participants | Body-shape shortcut risk in the nine-class stage |
| On Dataverse, 370 E01 names carry another E01 clip's video, and `E01_P02_AR_VFL_GM` duplicates `E01_P06_AR_VLL_GM` | Participant, view, variation and scores would all be attached to the wrong video. Your copy does not have this fault |
| In your copy, `E01_P60_AF_VLR_GF` and `E01_P60_AF_VML_GF` are byte-identical | One of the two labels is wrong; keep one and drop the other |
| 242 clips have PF1 = 0 from all three judges (E01 4, E02 45, E03 18, E05 43, E07 9, E08 19, E09 104), in blocks by participant | Class labels are wrong or atypical for these clips |
| Participant IDs 25, 44, 51, 54 were removed for "technical/quality issues", including clips with "non-exercise preparation footage" | Remaining clips may also contain idle footage |
| The repository is CC0 1.0 and 158.7 GB | Licence is permissive; download selectively |

### Still unknown until video is examined

- Frame rate, resolution, duration, rotation tag and codec, per cohort and phone.
- What a clip contains. The scores suggest expert E01 clips hold about nine or ten repetitions and non-expert clips far fewer, but that rests on reading PF4 as "repetitions ÷ 10", which is itself unconfirmed.
- Whether the three views of a performance were filmed at the same time. The form scores are identical across the three views in 98–100% of expert performances, which fits one performance filmed by three phones; the repetition item differs across views in 81% of E01 performances, which fits different trimming.
- Which arm moves. The questionnaire says the right arm; check that every participant follows it.
- How oblique the "left" and "right" views are, and which side of the body each one shows. Some of the paper's sample frames look oblique rather than square-on.
- Whether occlusion clips contain a second person, as the paper's sample frame does.
- What the PF1 = 0 clips actually show, and which participant the duplicated clip belongs to.
- Whether MediaPipe finds a pose at all on the lying exercises (E07–E09).

## 2. Stage 1: E01 pose feasibility

**Question.** Does MediaPipe produce usable landmarks on E01 clips, and do those landmarks carry the movement that defines abduction?

**Scope.** Front view first. In a true side view, a raise in the frontal plane projects onto a vertical line: the 2D shoulder angle jumps from 0° to 180° instead of sweeping through. Side views are therefore a stress test, not part of the pass criteria.

### Inputs

- E01 clips of the 41 development participants (436 clips, 16.3 GB). Work in this order: front view, then side views. Leave the 10 lockbox participants' clips unopened (Section 6).
- You can start before the full download finishes: the non-expert E01 clips (176 clips, 1.5 GB) and the expert front-view E01 clips (128 clips, 4.7 GB) cover the whole front-view analysis.
- `manifest.csv` and `participant_splits.csv`.
- Pose Landmarker model files, Lite and Full. Record their SHA-256 and the MediaPipe version.

### Steps

**A. Inspect before extracting.**
1. Probe every clip: duration, frame count, nominal and timestamp-derived frame rate, width, height, rotation tag, codec. Write `video_probe.csv`.
2. Make a contact sheet per clip (six evenly spaced frames) using the same decoder the pose step will use. A person lying sideways on a contact sheet means the rotation tag is being ignored.
3. Watch 30 clips in full: 12 ordinary expert clips, 12 ordinary non-expert clips, the four development-set E01 clips that all judges marked PF1 = 0 (`E01_P10_A{F,L,R}_VML_GF`, `E01_P31_AF_VHJ_GM`) and the duplicated pair. Log repetitions performed, which arm, peak height (below shoulder, shoulder level, overhead), idle time at start and end, how oblique the view is, and whether a second person appears. This log is the ground truth for the repetition counter and settles most of the unknowns above.

**B. Extract landmarks.**
- Pose Landmarker in video mode, a new landmarker instance for each clip (video mode keeps tracking state between frames), timestamps taken from the container and strictly increasing.
- Every frame at native rate. Decimation is tested later, on the stored landmarks.
- Default confidence thresholds, one pose. For occlusion clips also run with two poses and select the subject by torso size and continuity.
- Run Lite and Full on the same clips.
- Save per clip: timestamps, 33 image landmarks (x, y, z, visibility, presence), 33 world landmarks, frame size, and a sidecar with model hash, library version and options. Never overwrite a run.

**C. Flag unreliable frames.** Each flag is stored per frame; none silently drops data.

| Flag | Rule (starting value) | Catches |
|---|---|---|
| `no_pose` | No detection returned | Missed detection |
| `low_visibility` | Any required landmark below 0.5 visibility. Required: both shoulders, both hips, and the elbow and wrist of the moving arm | Occlusion, darkness |
| `out_of_frame` | A required landmark outside the image | Extrapolated points |
| `scale_jump` | Torso length differs from the clip's running median by more than 20% | Switch to another person, detector glitch |
| `teleport` | Hip centre moves more than half a torso length between consecutive frames | Same |
| `bone_length` | Upper-arm or forearm length in world coordinates differs from the clip median by more than 25% | Implausible limb |
| `side_swap` | Left and right shoulders exchange sides relative to the clip majority (front view only) | Mirrored skeleton |

Per clip, summarise: share of usable frames, longest unusable gap in seconds, number of gaps, and jitter (RMS difference between the raw and smoothed arm-elevation signal). Interpolate gaps up to 0.3 s; leave longer gaps missing and drop any repetition that overlaps one. Also store mean brightness and a blur measure per frame, as explanatory variables rather than flags.

**D. Extract the movement signal.**
- Compute arm elevation for both arms (definition in Section 3) and smooth it with a zero-phase low-pass filter near 3 Hz. Keep a causally filtered copy to preview what a phone would see.
- Segment repetitions as valley, peak, valley, with peak prominence of at least 20° and at least 30% of the clip's range, and peaks at least 1.5 s apart.
- Per repetition record amplitude, raise time, lower time and peak angular speed.
- Compare repetition counts with the watch log (primary) and with PF4 across all expert clips (exploratory, since the meaning of PF4 is unconfirmed). Compare raise and lower times with the questionnaire's 1.5–2.5 s window and with PF2.

**E. Visualise.**
1. Skeleton overlay videos for a few clips per view, variation and cohort, and for every clip that fails quality checks. This is the most informative single check.
2. Arm elevation against time per clip, both arms, unusable frames shaded, detected repetitions marked.
3. One heatmap of all clips (rows) against time (columns), coloured by arm elevation and sorted by cohort and view.
4. Repetitions overlaid on a 0–100% time axis with a mean and spread band, per cohort.
5. Wrist path relative to the shoulder, in torso lengths.
6. A table of median usable-frame share by variation and view, per cohort.
7. For one performance with three views, the 2D and the 3D elevation traces from each view on one axis.

**F. Optional: hand-label about 150 frames** (shoulder, elbow, wrist of the moving arm and both hips), sampled across views, variations and arm heights. MobiPhysio has no motion-capture reference, so this is the only way to put a number on landmark error for this dataset.

### Outputs

`video_probe.csv`, the watch log, landmark files per clip, `qc_frames` and `qc_clips` tables, a repetition table, `e01_features.csv`, the figures above, and a one-page note stating which criteria passed.

### Criteria

These are proposed defaults. Change them with your supervisor if you wish, but fix them before running. All apply to front-view clips of development participants.

| Criterion | Pass when |
|---|---|
| Coverage | In full-light and medium-light clips, at least 90% of clips in each cohort have 80% or more usable frames and no gap longer than 1 s |
| Signal | In at least 90% of usable clips, one arm clearly moves (its elevation range is at least twice the other arm's and at least 45°) |
| Repetitions | Pose count within one of the watched count in at least 22 of the 24 ordinary clips in the watch log |
| Model size | Lite and Full agree: median absolute difference in arm elevation of 10° or less, and clip range within 15° in at least 90% of clips |

Reported without a pass mark: coverage and jitter for every view, variation and cohort; 2D against 3D elevation in side views; occlusion clips with one pose against two.

### Decision

- Coverage, signal and repetitions pass: go to Stage 2.
- Coverage fails in full light: the approach is not viable as configured. Try the Full or Heavy model, check rotation and resolution, then consider another pose estimator.
- Coverage passes but signal or repetitions fail: landmarks exist but the derived signal is wrong. Check the pixel-space conversion and the angle definition against overlays.
- Lite fails where Full passes: continue with Full and carry on-device cost as an open risk into the mobile stage.

## 3. Features for E01

Convert landmarks to pixels first (x × width, y × height). MediaPipe normalises x and y separately, so angles computed on normalised coordinates are wrong in any non-square frame.

**Angles** (landmarks: shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24)

| Angle | Definition | Role |
|---|---|---|
| Arm elevation, each arm | Between shoulder-to-elbow and the trunk line (shoulder midpoint to hip midpoint). 0° hanging, 90° horizontal, 180° overhead | The abduction signal |
| Elbow angle, each arm | At the elbow, between upper arm and forearm. 180° is straight | The questionnaire requires a straight elbow; the rotation exercises hold it near 90° |
| Trunk inclination | Trunk line against image vertical | Standing against lying; "stand straight" |
| Plane of elevation (3D, exploratory) | Direction of the upper arm in a trunk-fixed frame built from world landmarks: near 0° for sideways, near 90° for forward | Abduction against flexion and circumduction. Depends on estimated depth, the least reliable output |

**Two 2D stand-ins for the plane of elevation** (front view): the sideways travel of the wrist relative to the shoulder, and the apparent arm length (shoulder to wrist). A sideways raise keeps the arm at full apparent length; an arm pointing at the camera shortens.

**Normalisation**
- Lengths and positions in torso lengths (clip median of shoulder-midpoint to hip-midpoint distance), relative to the shoulder or hip centre. Torso length holds up in oblique views; shoulder width does not.
- Prefer angles between body segments. They are unaffected by camera shift and roll, which is what the jitter clips contain.
- Label arms "moving" and "resting" by elevation range, not "left" and "right", so the features do not depend on which arm was used.
- Time in seconds from timestamps, signals resampled to one fixed rate.

**Temporal measurements** from the moving arm's elevation: 5th percentile, 95th percentile and their difference; median raise time and lower time (10% to 90% of amplitude); 95th percentile of angular speed; median repetition period. The period is undefined for a clip with one repetition, so it stays out of the minimum set below.

**Minimum feature set, 15 values per clip**

| Group | Features |
|---|---|
| Moving arm elevation | 5th percentile, 95th percentile, range |
| Resting arm elevation | 95th percentile, range |
| Moving elbow | median, 5th percentile |
| Trunk inclination | median, range |
| Wrist relative to shoulder | sideways travel, vertical travel |
| Apparent arm length | ratio of 5th to 95th percentile |
| Tempo | median raise time, median lower time, 95th percentile angular speed |

**Leave out of the classifier**: repetition count, clip duration, frame rate, resolution, share of missing frames, the view, variation and gender codes from filenames, and everything in the score CSV. Each is either unavailable on a phone or a shortcut to cohort.

## 4. What a one-class experiment can and cannot measure

| Goal | E01 clips alone | Reason |
|---|---|---|
| Pose extraction works on E01 | Yes | Coverage, jitter, Lite against Full, limb-length stability, repetition and tempo agreement with humans; landmark error if you hand-label frames |
| The abduction signal is present and consistent | Yes | Range, tempo, moving arm, variation between people, views and cohorts |
| A rule accepts E01 | Partly | You get the share of E01 clips the rule accepts (its sensitivity) and nothing else |
| A rule or model rejects other exercises | No | False-positive rate, precision, ROC and PR curves, F1 and accuracy all need clips that are not E01. A rule that accepts everything scores 100% on E01-only data |
| Multiclass accuracy | No | One class has no confusion matrix |
| Exercise quality or clinical correctness | No | Class labels say which exercise was intended, not whether it was done well or safely |
| Phone performance | No | Desktop throughput says little about a phone |

A one-class model (one-class SVM, isolation forest, distance to the E01 centroid) can be trained on E01 alone, but evaluating it still needs other clips. Rest periods inside E01 clips give a weak stand-in: they test "moving against idle", not "abduction against another exercise".

## 5. Stage 2: E01 against the other eight exercises

**Question.** Do the Section 3 features separate E01 from each other exercise, for people the model has never seen?

### Data

Negatives are suitable: 3,128 clips, and every expert who recorded another exercise also recorded E01 (as did 47 of the 51 E03 performers and 44 of the 45 E05 performers). Judging from the paper's sample frames, the classes differ in difficulty:

- **Near**: E05 circumduction (the arm passes through raised positions), E03 and E04 rotations (upper arm held out, elbow bent), E02 adduction (arm across the body).
- **Far**: E06 (both arms forward), E07–E09 (lying on a bench).

A pooled score is dominated by the far classes, so every result is reported per negative class.

| Download | Clips | Size |
|---|---|---|
| Stage 2a: front view, full light, all nine classes | 266 (221 beyond E01) | 5.1 GB (4.2 GB beyond E01) |
| Stage 2b: front view, all variations, E02–E05 | 746 | 17.1 GB |

Take these clips from the authors' copy. Dataverse is known to serve wrong videos under E01 names, and the other exercises have not been checked. `scripts/download_clips.py` can still fetch a subset from Dataverse, but it refuses unchecked names unless `--accept-unverified-names` is given, and anything fetched that way must be compared with the authors' copy before use.

Before modelling, run the Stage 1 coverage check on each negative class. A class where pose extraction fails is "not E01" for the wrong reason, and must be reported separately.

**Label hygiene.** Drop the duplicated pair. Set aside the 242 clips with PF1 = 0 from all judges, keep them out of training and the main test, and report what the model says about them as a separate result.

### Analyses

| Analysis | Data | Purpose |
|---|---|---|
| A, primary | Experts, front view, grouped folds | All nine classes from one cohort, site and set of phones (24 people, 19 of them in the development folds): no cohort shortcut |
| B, transfer | Train on experts, test on non-experts (E01 against E03, E05, E08, E09) | Different site, phones and clip structure; the closest this dataset gets to new users |
| C, pooled | Both cohorts, grouped folds, reported per cohort | Uses all data; read alongside A and B |
| D, stress | Side views and each variation | Descriptive |

**Shortcut probe.** Train a model to predict cohort from the same features within E01. If cohort is easy to predict, distrust analysis C and rely on A and B.

### Model ladder

1. **Floor**: always answer "not E01".
2. **Rules.** Standing (median trunk inclination under 30°), one arm moving (moving-arm range at least 45°, resting-arm range at most 25°), elbow straight (median at least 150°), sideways raise (wrist travels sideways at least 0.6 torso lengths). Write the thresholds down from the questionnaire before seeing data, then also report a version tuned inside training folds.
3. **Random Forest**, default settings, class weights balanced.
4. **XGBoost**, small fixed configuration (depth 3–4, up to 300 trees), early stopping on an inner participant-grouped split only.

Same folds and features for all. Report permutation importance on held-out folds. Choose the simplest model whose result lies within the fold-to-fold spread of the best one.

### Metrics and criteria

- For each negative class, the false-positive rate at the threshold that accepts 95% of E01, with the threshold set on training folds.
- Area under the precision-recall curve, and a calibration curve.
- Intervals from resampling participants, not clips. With 19 development experts, intervals for a single class will be wide; Stage 2a can show clear promise or clear failure, not fine differences.

Proposed pass mark for analysis A: false-positive rate of 10% or less for each of E02–E05, and 2% or less for each of E06–E09.

### Decision

- Pass: go to the nine-class stage with the same pipeline. If the rules pass, keep them as the reference implementation for the phone.
- Fail on one pair (most likely E05): add features aimed at that pair, then try per-repetition curves resampled to a fixed length with a Random Forest, or a cheap time-series classifier such as MiniRocket.
- **A temporal neural network is justified only when all three hold**: tree models still miss the pass mark after targeted features; the misclassified clips have clean pose tracks; and the confusion concerns order or shape over time, which a summary statistic cannot express. With about 19 development experts a network can easily memorise people, so use a small GRU or 1D CNN on body-normalised sequences, the same folds, and augmentation (mirroring, time stretch, small rotations, landmark noise).
- Fail because pose tracks are poor: fix pose extraction first. No classifier repairs bad landmarks.

## 6. Splitting by participant and preventing leakage

**The participant is the only unit that is split.** `participant_splits.csv` assigns each of the 58 participants to a lockbox (11 people) or one of five development folds (9–10 people), stratified by cohort, gender and whether they recorded the floor exercises. It was generated with seed 20261010 before any video was examined. Every model at every stage uses it.

The lockbox is opened once, at the end of the nine-class stage. It is small (three to five experts per class), so its estimate will be noisy; it is still the only one untouched by design choices.

| Leak | Evidence in this dataset | Control |
|---|---|---|
| Same performance from three views | 983 of 1,014 expert performances have all three views | Split by participant |
| Same person across variations and exercises | Experts have up to 162 clips each | Split by participant |
| Duplicate and misnamed files | In your copy `E01_P60_AF_VLR_GF` equals `E01_P60_AF_VML_GF`; on Dataverse 370 E01 names carry another clip's video | Use the authors' copy only and compare every new exercise against Dataverse with `build_manifest.py --videos`; keep one of a duplicated pair; also scan pose tracks for near-identical pairs across participants, since re-encoded copies will not share an MD5 |
| Windows or repetitions cut from one clip | Applies once windows are used | Windows inherit the participant's fold; weight so each clip counts once |
| Fitting on test data | Any scaler, imputer, feature selection, threshold or early stopping | Fit inside the training fold; tune with nested grouped folds or not at all |
| Cohort shortcut | Non-experts exist in five classes only, at another site with other phones | Pose features only; analyses A and B; shortcut probe |
| Clip-length shortcut | Clip size differs fivefold between cohorts | Exclude count and duration features |
| Missing-data shortcut | Pose failure may differ by class | Keep the missing-frame share out of the features |
| Metadata as input | View, variation and gender are in the filename | Never model inputs |
| Label noise in blocks | PF1 = 0 covers whole participant-exercise blocks, such as all 18 E03 clips of P02 | Set aside and report separately |
| Thresholds chosen by eye | Rule thresholds | Fix beforehand or tune inside folds |

## 7. Expert scores and questionnaires

### What the data contains

Your CSV has, for all 3,686 clips, three judges' marks on seven items. For abduction the questionnaire (paper, Fig. 4) asks:

| Item | Question | Values in the CSV |
|---|---|---|
| PF1 | Is the person doing the right exercise? | 0, 1 |
| PF2 | Ideal speed, 1.5–2.5 s to raise or lower | 0, 0.5, 1 |
| PF3 | Amplitude: completing the exercise? | 0, 1 |
| PF4 | Repetition: at least 10 | 0 to 1 in tenths |
| CF1 | Head and trunk neutral | 0, 0.5, 1 |
| CF2 | Arm alignment: right arm raised to 170–180°, palm forward, left arm at the side | 0, 0.5, 1 |
| CF3 | Legs straight | 0, 0.5, 0.6, 0.66, 1 |

The setup line adds: standing straight, shoulders level, elbows straight. The total out of 7 is rescaled to 100 and averaged over judges. Only the abduction questionnaire is in the paper; the other eight are in the article's online appendix and are not in the repository.

### What it could support

- **A repetition and tempo check.** PF4 and PF2 describe things pose can measure, and the judges agree on them almost completely.
- **Form indicators for CF1 and CF2**, tested within one cohort: for example, whether elbow bend or peak arm height tracks CF2, and trunk sway tracks CF1.
- **Clinician-written criteria.** The questionnaire text is a starting point for the reviewed guidance content in the LLM stage.
- **Label cleaning.** PF1 identifies clips whose class label is doubtful.

### Limitations

1. **Two of the seven points are one fact.** PF3 equals 1 exactly when PF4 equals 1, in all 11,058 judge rows. There is no independent amplitude or range-of-motion label.
2. **Within experts the score is mostly a repetition count.** PF3 and PF4 together account for 73% of the variance in the expert total; the three form items account for 30%.
3. **The repetition item depends on how the clip was cut.** Across the three views of one expert E01 performance, PF4 differs in 81% of cases and the 0–100 score differs by a median of 7 points and up to 23.
4. **Cohort is confounded with everything.** Cohort alone explains 58% of score variance (69% within E01). Experts average 82.3 and non-experts 37.3; no non-expert clip exceeds 67.1. In E01, about 30 of the 36 points separating the cohorts come from the speed, completion and repetition items, which are near zero for almost every non-expert clip, and 6 come from arm alignment. Trunk and leg marks are about the same in both cohorts. The cohorts also differ in site, phones and clip size, so a score regressor could succeed by telling them apart.
5. **Form marks are coarse and noisy.** Three levels, mostly 1. For expert E01 clips the three judges agree exactly on CF1, CF2 and CF3 in 47%, 57% and 80% of clips (single-judge ICC 0.46, 0.39, 0.49).
6. **Form marks belong to performances, not clips.** They are identical across the three views in 98–100% of expert performances, so a side-view clip may carry a mark for something not visible from that side.
7. **The judges are not independent on the primary items.** PF1–PF4 are identical across judges in over 99% of clips. Headline agreement figures are inflated by this and by the gap between cohorts.
8. **Few low scores within a protocol.** Only 27 of 3,011 expert clips score below 40.
9. **Effective sample size is the number of people.** Within non-expert E01 clips, 75% of score variance lies between participants (27 people).
10. **Scope.** Young, healthy students; no patients or older adults. Marks are for a whole clip, not for repetitions. The questionnaire is the authors' own instrument and the paper reports no validation against an established clinical measure. Nothing here supports a claim of clinical correctness.

### A sound later experiment

- Predict items, not the total. Treat repetitions and tempo as measurements and report error against PF4, PF2 and your own counts.
- Model CF1 and CF2 within one cohort, at the performance level, front view, split by participant, against the mean of the three judges.
- Compare against trivial predictors: cohort only, clip duration only, repetition count only. A quality model is of interest only if it beats them within a cohort.
- Use agreement between judges as the ceiling.
- For real quality feedback, record new data: deliberate, physiotherapist-defined errors, and later patients under ethics approval. The paper names this as future work.

## 8. Separate path: nine classes, phone, LLM

Keep this path apart from the E01 work. Each stage starts only when the one before it passes.

### Stage 3: nine-class classifier (offline)

- Repeat the Stage 1 pose check for each class on the Stage 2a subset. The lying exercises are the main risk; if detection fails, test rotating the frame by 90° before inference, the larger models, or another estimator, or leave those classes out of the first version.
- Extend the features: hip and knee angles (E07, E08), trunk extension (E09), hand against forearm (E06), and the direction of forearm rotation (E03 against E04).
- Same splits and model ladder. Report macro-F1, per-class recall and the confusion matrix, by cohort, view and variation.
- A phone app also needs a "none of these" answer, and the dataset has no such class. Use idle segments and the PF1 = 0 clips as a first test and plan to record more.
- Decimate the stored landmarks to 15 and 10 frames per second and switch to causal smoothing, to learn the lowest frame rate at which results hold.
- Proposed pass mark: macro-F1 of 0.90 or more and no class below 0.85 recall on development folds, front view. Then run once on the lockbox.

### Stage 4: real-time on a phone

- MediaPipe Pose Landmarker in live-stream mode, the same feature code ported, a sliding window of about 6–8 s updated every second, a repetition counter with hysteresis, and a quality gate that asks the user to reposition when landmarks are unreliable.
- **Parity test**: run one recorded clip through the Python pipeline and the phone pipeline and compare landmarks and features. This catches differences between training and serving code.
- **Measure, do not assume**: frame latency, sustained frame rate over ten minutes, temperature, memory and battery, on at least one low-end phone of the kind used to record the dataset, for Lite and Full.
- Pass when the sustained frame rate is at or above the lowest rate that held up in Stage 3, parity holds, and a live trial with at least ten people who are not in the dataset reports per-class recall.
- Process video on the device and store no footage.

### Stage 5: LLM guidance (last and narrow)

- The LLM does not look at video and does not judge movement. It receives a structured summary from the deterministic pipeline (exercise, confidence, repetitions, tempo, flagged deviations from a fixed list) and phrases feedback from a library of snippets that a physiotherapist has reviewed and signed off, keyed by exercise and finding.
- It must not diagnose, change a prescription, or state a number or threshold that is not in the summary or the library. When confidence or pose quality is low it says so. Fixed safety text (stop if in pain, contact your physiotherapist) is not generated.
- Start with fixed templates as the baseline. Add the LLM where it clearly helps, for example explanations in Bangla.
- Pass when a physiotherapist-reviewed test set of scenarios shows no unsupported statements and correct behaviour on low-confidence input.
- Software that advises on exercises for medical conditions may be regulated as a medical device. Raise this with your supervisor before describing the app as clinical.

## 9. Stage gates at a glance

| Stage | Inputs | Outputs | Go when |
|---|---|---|---|
| 0. Data hygiene (done) | Listing, README, CSV | Manifest, splits, defect list | Counts reconcile with the paper (they do) |
| 1. E01 pose feasibility | E01 development clips, Lite and Full models | Probe table, watch log, landmarks, QC tables, repetition table, features, figures | Coverage, signal and repetition criteria pass on front view |
| 2. E01 against the rest | Stage 2a clips, then 2b; folds | Per-class false-positive table, model comparison, shortcut probe, error gallery | Analysis A meets the false-positive marks |
| 3. Nine classes | All front-view clips needed; extended features | Confusion matrix on folds, one lockbox run, frame-rate tolerance | Macro-F1 and per-class recall marks; lockbox consistent with folds |
| 4. Phone | Stage 3 model, target phones | Parity report, measured latency and frame rate, live-trial results | Frame rate sufficient, parity holds, live trial reported |
| 5. LLM | Reviewed content library, structured summaries | Scenario test results, physiotherapist sign-off | No unsupported statements; safe handling of low confidence |

## 10. Setup notes

- This machine has Python 3.14 and 3.11, `uv`, and no `ffprobe`. Use PyAV (`pip install av`) or OpenCV for probing, or install FFmpeg.
- PyPI offers MediaPipe wheels for Python 3.14; the current release is 1.1.0. The API described here is the Pose Landmarker task as documented for the 0.10 series. Check it against the version you install, then pin every package version.
- Keep clips outside the repository (drive D: has about 790 GB free). Re-run `build_manifest.py --videos <folder>` to record which clips are present.
- The released clips show faces. Keep clips and overlay videos local.

## 11. Questions for the dataset authors or your supervisor

1. Were the three views of a performance filmed at the same time?
2. What does PF4 = 0 mean when PF1 = 1, and does PF4 count all repetitions or only complete ones?
3. Do the authors know that Dataverse serves the wrong video under 370 E01 names, and which variation does the duplicated P60 clip (`VLR` and `VML`) really show?
4. What is variation code `VP`?
5. Were the non-expert clips cut to single repetitions, and when were they recorded?
6. Where are the other eight questionnaires, and is your CSV going to replace the public one?
