# Five pose models on MobiPhysio clips: what was built and what it shows

Written 2026-10-12. This report explains the five pose estimation models now in the project, how they are compared, what came out on the first clips, and what that does and does not tell you. How to run the program is in `README.md`.

## 1. The short answer

- **All five models run on your computer** and can be tried on any video with `run_demo.bat`: MediaPipe Full, ViTPose Base, OpenPose, RTMPose-m and YOLO26x-pose.
- **On the five clips tried so far, the five models tell the same story.** They count the same number of repetitions on every clip, and their arm-elevation angles typically differ by 1° to 5°.
- **They do not give the same range of motion.** On one clip the measured range runs from 152° (ViTPose) to 162° (MediaPipe). Any rule of the kind "the arm must reach X degrees" would depend on which model is used.
- **With a passer-by in the picture, the four newer models lost fewer frames than MediaPipe**: 92% to 94% usable frames against 84% on the one occlusion clip tried.
- **Four of the five give foot points**: MediaPipe, RTMPose, OpenPose (since the switch to its 25-point model) and ViTPose (since the switch to its whole-body version). YOLO26 stops at the ankle, and Ultralytics publishes no version with feet.
- **Hands are read by a separate hand model that works with any of the five** (Section 7). It says whether each hand is open or closed, but it finds the hand reliably only in clear expert front-view clips: 86% of frames there, 31% in non-expert clips.
- **MediaPipe's 3D points are now shown and measured** (Section 8): a 3D panel in the live window and a 3D section in the report. They are an estimate from one camera and less steady than the picture points: in expert front-view clips the 3D upper arm is about a quarter shorter with the arm overhead than with the arm down. One 3D measure looks useful in front views, the direction of the raise, which picked out the one performance the judges rejected.
- **The five clips were easy ones.** The cases where Stage 1 found MediaPipe failing (right-side views, dark non-expert clips) have not been tried with the other four models. That is the comparison that would matter, and it is not done.
- **None of this measures accuracy.** The dataset has no true joint positions, so the models can only be compared with each other.

## 2. Terms used

| Term | Meaning |
|---|---|
| Pose model | A program that takes a picture and returns the positions of body points (shoulder, elbow, wrist and so on) |
| Point, keypoint, landmark | One of those body points. The three words mean the same |
| Score, confidence | A number the model attaches to each point saying how sure it is. Each model scales it differently |
| Person detector | A model that only finds people and draws a box around each |
| Participant | The person doing the exercise, as opposed to a passer-by |
| Arm elevation | Angle between the upper arm and the trunk: 0° hanging, 90° horizontal, 180° overhead |
| Usable frame | A frame in which the participant was found and all quality checks passed (Section 5) |
| 3D points, world landmarks | MediaPipe's second set of points: positions in space, in metres around the middle of the hips. The depth is estimated, not measured (Section 8) |
| Direction of the raise | Which way the upper arm points when seen from above: 0° out to the side, 90° straight ahead. Needs 3D points |
| GPU, CPU | The graphics card and the main processor. Large models run much faster on the GPU |

## 3. Four ways to find a pose

The five models use four different designs. Knowing the design explains most of their behaviour.

| Design | How it works | Strength | Weakness | Used by |
|---|---|---|---|---|
| Detect once, then track | Find the person in the first frame, then follow them from frame to frame without looking for anyone else | Fast; made for phones | Follows whoever it locked onto, including a passer-by | MediaPipe |
| Top-down | A person detector finds everyone; each person is cut out, resized and given to the pose model | The pose model always sees the person large and centred | Needs the detector first; cost grows with the number of people | ViTPose, RTMPose |
| Bottom-up | Find every shoulder, elbow and so on in the whole picture, then work out which belong together | One pass, however many people | Small people get few pixels; joining points can go wrong | OpenPose |
| One-stage | One network outputs a box and the body points for every person together | Simple and fast | Points come from one coarse pass, not a close-up | YOLO26-pose |

## 4. The five models

| Model | Published | Design | Points | Feet | Runs on | Licence |
|---|---|---|---|---|---|---|
| MediaPipe Pose Landmarker, Full | Google, 2020 (BlazePose) | Detect once, then track | 33 | Heel and toe tip | CPU | Apache-2.0 |
| ViTPose Base, whole-body version | Xu and others, 2022 | Top-down | 23 (of the 133 it outputs) | Heel, big toe, small toe | GPU | Apache-2.0 |
| OpenPose, 25-point body model | Cao and others (CMU), 2017; this model 2019 | Bottom-up | 25 | Heel, big toe, small toe | GPU | Non-commercial research only |
| RTMPose-m, 26-point version | Jiang and others (OpenMMLab), 2023 | Top-down | 26 | Heel, big toe, small toe | GPU | Apache-2.0 |
| YOLO26x-pose | Ultralytics, January 2026 | One-stage | 17 | None | GPU | AGPL-3.0 |

MediaPipe Full, ViTPose and OpenPose are your choices. I added RTMPose and YOLO26-pose because they are the current real-time models of the two designs not yet covered, and both are meant to run on small devices, which is where this project is heading.

**What the authors report.** These figures are from the papers and documentation, on the COCO benchmark of everyday photographs. They are not measurements on physiotherapy videos, they were obtained under different conditions, and they should not be read as a ranking.

| Model | Reported result | Source |
|---|---|---|
| ViTPose Base, whole-body version | I did not verify a figure for this version, so none is given. For reference: ViTPose++ Base, the body-only model used here until the switch, is reported at 77.0 AP on the COCO validation set | ViTPose++ paper, results table |
| RTMPose-m | 75.8 AP on COCO, for its 17-point version; the 26-point version used here has no figure in the abstract | RTMPose paper, abstract |
| YOLO26x-pose | 71.6 mAP on the COCO keypoints validation set | Ultralytics documentation |
| MediaPipe (BlazePose) | 33 points at over 30 frames per second on a Pixel 2 phone; no COCO figure | BlazePose paper, abstract |
| OpenPose | I did not verify a figure, so none is given | |

**Notes on each model as set up here**

- **MediaPipe Full.** The same settings as Stage 1. Its Windows package has no GPU support, so it runs on the CPU. A fresh extraction of eight clips in the enlarged environment was bit-for-bit identical to the Stage 1 files.
- **ViTPose.** One network, ViTPose Base trained on COCO-WholeBody, gives the 17 body points and the 6 foot points together.
  - *The file.* It outputs 133 points (body, feet, hands, face); the first 23 are kept. It is a conversion of the authors' checkpoint by a community project (`easy_ViTPose`), run with ONNX Runtime on the GPU. It is a third-party file with no official checksum to compare against.
  - *The check that stands in for a checksum.* I compared it with ViTPose++ Base from the official Hugging Face repository on three clips. Its 17 body points lie 1.4% to 2.1% of a torso length from that model's, which is as close as the closest pair among the five models. Its foot points lie a median of 6 to 8 pixels from RTMPose's and OpenPose's on the test clip.
  - *A convention that had to be found.* How this file's output maps back to the picture is not documented. I tried three standard conventions in that comparison; one gave the agreement above, the other two gave 6% to 7%.
  - *History.* The first setup was ViTPose++ Base through the `transformers` library: 17 body points and no feet, because that library ships only the body output head (its whole-body "expert" returned points that were not on the person). For a few hours on 2026-10-12 both networks ran together, the old one for the body and the new one for the feet. You asked why two were needed when one can do all of it, and the comparison had shown the new file could be trusted, so the old one was retired to the archive.
- **OpenPose.** CMU's original 25-point model with feet, but not CMU's program.
  - *The files.* CMU's model server is offline, so the weights come from a copy of their files on the Zenodo research archive. The copy is genuine: the body model's checksum equals the one CMU lists in its own build file, and the network definition is identical to the one in CMU's GitHub repository.
  - *Running it.* The files are in Caffe format, and CMU's program needs software that no longer runs on current graphics cards. `poselab/estimators/caffe.py` reads the two files directly and rebuilds the network in PyTorch; no Caffe is installed and nothing from the files is executed.
  - *From maps to people.* Written to follow CMU's code: its list of 26 limbs, its default thresholds, and its way of scoring a limb. Small differences from CMU's program in resizing and peak finding are still possible.
  - *History.* Until 2026-10-12 the older 18-point model without feet was used, through a third-party PyTorch conversion. It is in the archive.
- **RTMPose-m.** Run through the `rtmlib` library with ONNX Runtime on the GPU. The 26-point version was chosen because it has foot points.
- **YOLO26x-pose.** The largest of the YOLO26 pose models. On first use Ultralytics created a settings file in your Windows profile (`AppData\Roaming\Ultralytics`); the program switches off its usage-statistics setting there.

**Versions with more points that are not set up.** You asked whether other models are in OpenPose's earlier position, with a fuller version that was not downloaded.

| Model | Feet as set up | A fuller version exists? |
|---|---|---|
| MediaPipe Full | Yes | Nothing missing for feet. Fingers come from the separate hand model (Section 7) |
| OpenPose | Yes, since the switch to 25 points | CMU also has separate hand and face models. They are in the bundle that was downloaded, and are not used |
| RTMPose-m | Yes | A whole-body version (RTMW, 133 points with hands and face) can be run by the library already installed. Not downloaded |
| ViTPose Base | Yes, since the switch to its whole-body version | The same network also outputs 21 points per hand and 68 face points. They are not used |
| YOLO26x-pose | **No** | **No.** Ultralytics publishes pose weights with 17 body points only. Feet would mean training the model yourself |

**Licences matter for the phone app.** OpenPose may only be used for non-commercial research. YOLO26 is AGPL-3.0: an app that includes it must be released under the same licence, or you buy Ultralytics' commercial licence. Both are fine for a capstone comparison. MediaPipe, ViTPose and RTMPose are Apache-2.0 and free to ship.

## 5. How the comparison is kept fair

**The same for every model**

- **The same frames.** Each video is decoded once and every frame is given to every model.
- **The same rule for who the participant is.** At the start, the largest person near the middle. Afterwards, the person who overlaps the participant's usual position over the last five seconds. If nobody does for two seconds, start again. (`poselab/subject.py`)
- **The same person boxes for the two top-down models.** ViTPose and RTMPose share one person detector, so a difference between them comes from the pose model.
- **The same list of joints.** Each model's points are renamed into one shared list of 27; a model without heels simply has none. (`poselab/skeleton.py`)
- **The same five quality checks**, the Stage 1 checks: participant found, points confident, points inside the picture, no jump in body size, no jump in position. (`poselab/analysis.py`)
- **The same arm.** All models are made to describe the arm that most of them saw moving.

**Not the same, and why it matters**

- **Confidence thresholds.** "Confident" means above 0.5 for MediaPipe and YOLO, 0.3 for ViTPose and RTMPose, and 0.1 for OpenPose. These are each tool's customary values. The scores are not on a common scale, so "usable frames" is judged slightly differently for each model.
- **Where a joint is.** The models were trained on datasets that mark joints in slightly different places. A steady offset of a few pixels between two models is a difference of convention, not an error.
- **3D.** Only MediaPipe estimates depth. Stage 1's limb-length check used it; the comparison leaves that check out so that all models face the same checks. MediaPipe's usable-frame numbers here are therefore not the Stage 1 numbers. The 3D points are shown and measured separately (Section 8) and play no part in the comparison between models.
- **Hardware.** MediaPipe runs on the CPU and the other four on the GPU. The speed figures say what happens on this computer, not which model is lighter.

## 6. What was measured

Five E01 clips from three development participants, all five models, on 2026-10-12, re-run after OpenPose was switched to its 25-point model and ViTPose to its whole-body version. Reports for clips A and B, the non-expert clip, the left side view and `E01_P10_AF_VML_GF` (Section 8) are in `outputs/demo/`, re-made when the 3D section was added.

### Clip A: front view, full light, expert (E01_P02_AF_VFL_GM, 897 frames, 30 s)

| Model | Usable frames | Repetitions | Lowest | Highest | Range | Jitter |
|---|---|---|---|---|---|---|
| MediaPipe Full | 100% | 11 | 7° | 169° | 162° | 0.72° |
| ViTPose Base | 100% | 11 | 11° | 163° | 152° | 0.83° |
| OpenPose | 100% | 11 | 9° | 169° | 160° | 1.09° |
| RTMPose-m | 100% | 11 | 8° | 169° | 161° | 0.79° |
| YOLO26x-pose | 100% | 11 | 10° | 167° | 157° | 0.31° |

Lowest and highest are the 5th and 95th percentile of arm elevation. Jitter is how much the frame-by-frame angle wobbles around its smoothed curve.

**Agreement between pairs of models, clip A**

| Pair | Median difference in arm elevation | Median distance between the same joints |
|---|---|---|
| ViTPose and YOLO | 1.5° | 2.0% of torso length |
| OpenPose and YOLO | 1.6° | 2.4% |
| RTMPose and YOLO | 1.6° | 2.7% |
| OpenPose and RTMPose | 1.9° | 2.6% |
| ViTPose and RTMPose | 2.0° | 2.4% |
| ViTPose and OpenPose | 2.1° | 2.1% |
| MediaPipe and RTMPose | 2.2° | 2.9% |
| MediaPipe and YOLO | 2.9° | 3.1% |
| MediaPipe and OpenPose | 3.0° | 3.4% |
| MediaPipe and ViTPose | 3.3° | 3.3% |

**How to read this.** The models agree closely on the shape and timing of the movement, which is what repetition counting needs. They agree less on the extremes: the range differs by up to 10°. MediaPipe's differences from the others (2.2° to 3.3°) are larger than the differences among the other four (1.5° to 2.1°). That fits its different training scheme (its own 33 points, the others COCO-style points) and does not show which is closer to the truth.

### Clip B: front view with a passer-by, expert (E01_P02_AF_VO_GM, 817 frames, 27 s)

A second person walks across the picture four times.

| Model | Participant found | Usable frames | Longest gap | Repetitions |
|---|---|---|---|---|
| MediaPipe Full | 90.5% | 84.0% | 1.57 s | 8, plus 1 cut off by the video |
| ViTPose Base | 100% | 93.0% | 0.53 s | 8, plus 1 cut off |
| OpenPose | 99.9% | 94.1% | 0.47 s | 8, plus 1 cut off |
| RTMPose-m | 100% | 93.0% | 0.53 s | 8, plus 1 cut off |
| YOLO26x-pose | 100% | 91.9% | 0.60 s | 8, plus 1 cut off |

**How to read this.** Each crossing costs every model about half a second, while the passer-by hides the participant. MediaPipe loses more because it follows one person only: once it locks onto the passer-by, it stays with them until they leave. The other four see both people in every frame, so the participant is picked up again as soon as they are visible. This is one clip, and the participant rule was tuned on this same clip (Section 10), so treat the numbers as an illustration.

### Three more clips, checked for robustness

| Clip | Kind | Usable frames, all models | Repetitions, all models | Range across models |
|---|---|---|---|---|
| E01_P26_AF_VFL_GF | Non-expert, front, 720 × 1280, 4.7 s | 99% to 100% | 1, plus 1 cut off | 83° to 88° |
| E01_P01_AL_VFL_GM | Expert, left side view, 30.6 s | 100% | 9 | 172° to 176° |
| E01_P01_AF_VLL_GM | Expert, front, low light, 30.1 s | 100% | 8, plus 1 cut off | 160° to 165° |

### Speed on this computer (clip A, all five models loaded together)

| Model | Runs on | Median time per frame | Of which person detector | Frames per second |
|---|---|---|---|---|
| MediaPipe Full | CPU | 13.6 ms | | 73 |
| YOLO26x-pose | GPU | 17.5 ms | | 57 |
| OpenPose | GPU | 20.8 ms | | 48 |
| RTMPose-m | GPU | 25.8 ms | 21.9 ms | 39 |
| ViTPose Base | GPU | 28.9 ms | 21.9 ms | 35 |

- **The person detector is the slow part of both top-down models.** RTMPose itself takes 3.9 ms and ViTPose 6.9 ms; finding the person takes 21.9 ms. A common remedy is to run the detector only every few frames. I have not tried it.
- **With all five running, the program processes 9 to 10 frames per second** without the hand model and about 7 with it. One model alone ran at 27 to 43 frames per second without the live window (measured before the last two model switches).
- **These are desktop figures.** They say nothing about a phone.

## 7. Hands: open or closed

Added later on 2026-10-12, after you asked why the skeleton stops at the wrist.

**Why the wrist is the end.** Body pose models are trained on body points, and those end at the wrist. Only MediaPipe's pose model has anything beyond it: three rough points per hand (the knuckles of the little and index fingers, and the thumb). These are now drawn. They show which way the hand points, which is what the wrist-extension exercise (E06) will need, but they are not fingertips.

**What was added.** A separate hand model, MediaPipe Hand Landmarker, which gives 21 points per hand including every fingertip. A hand is small in these videos, so for each hand a square around the wrist is cut out of the frame, enlarged and given to the hand model.

**What happens to the pose models.** Nothing. The hand model only needs to know where the wrist and elbow are. It uses the position the ticked pose models agree on (their median), runs once per frame for each hand, and its result is shown on every panel. It therefore works with any of the five, and it is not part of the comparison between them.

**How open or closed is decided.** One number, the finger extension: for each of the four fingers, the straight distance from wrist to fingertip divided by the length of the finger's path from the wrist, averaged. A flat hand gives about 1. A fist gives a low value, because the fingertips come back to the palm. It is computed from the hand model's 3D points, so it does not depend on which way the hand faces. The program calls a hand **open** from 0.85, **closed** below 0.60, and **partly closed** in between, judged over seven frames so that one blurred frame does not flip the answer. These limits come from geometry (every finger joint bent 25° gives about 0.92, bent 45° about 0.72, a fist about 0.4), not from labelled examples.

**How often the hand is found.** I ran the hand model over 108 development clips from 22 participants, every fourth frame, with wrists from the stored MediaPipe Full poses.

| Kind of clip | Clips | Share of frames with the hand found (median over clips) |
|---|---|---|
| Expert, front view | 24 | 86% |
| Expert, left side view | 24 | 58% |
| Expert, right side view | 24 | 41% |
| Non-expert, front view | 28 | 31% |
| All | 108 | 55% |

Within the expert front-view clips it was 87% to 95% in full or medium light and with low jitter, 77% in low light, and 55% with high jitter. In the non-expert low-light clips the median was 0%. In 69 of the 108 clips at least one of the two hands was found in under half the frames.

**Are the "closed" readings real?** Across the scan, 2.8% of the frames with a hand read as closed and 25.5% as partly closed. I enlarged eight of the lowest readings and looked at them. Five showed a visibly curled hand: a hand hanging at rest curls into a loose fist. Two of those five belonged to a passer-by, not the participant, in an occlusion clip. The other three were too blurred to judge. So a low value does go with curled fingers, but a blurred frame can give one too, and I have no clip of a deliberately clenched fist to test the "closed" end properly.

**What to take from this**

- **Good enough to look at, not yet good enough to rely on.** In a clear expert front-view clip the hand reading is available most of the time. In side views, low light, shaky video and most non-expert clips it is missing more often than it is there.
- **Check by eye.** The report shows the enlarged hand cut-outs next to the numbers for this reason.
- **The cost** is about 30 ms per frame for both hands, on the CPU. With all five pose models and hands, the program runs at about 7 frames per second.
- **Not tried:** a larger hand model, or one of the whole-body models that predict hands together with the body. Those are the next things to test if hands turn out to matter.

## 8. Depth: MediaPipe's 3D points

Added later on 2026-10-12, after you asked for MediaPipe's 3D landmark estimation.

**What it is.** MediaPipe returns every pose twice. The first set is the 33 points in the picture, which everything else in this project uses. The second set, which Google calls world landmarks, is the same 33 points as positions in space: metres to the left or right of, above or below, and nearer or farther than the middle of the hips. None of the other four models gives this.

**It was already being saved.** The program has stored these points from the start (the `world` array in `pose/mediapipe.npz`), and Stage 1 used them for its limb-length check. So nothing was downloaded and no model changed. What is new is that you can see them, and that the report measures with them.

**What was added**

- **Live window.** A panel "MediaPipe Full in 3D" next to the model panels. It draws the 3D skeleton from two directions the camera does not have: from the side and from above. It takes the place of one more panel. Most clips in the dataset are upright phone videos (428 of the 558 E01 clips are 1080 × 1920); for those, five models now sit six panels across instead of five, so each panel is about a sixth narrower. On a landscape video it fills a slot that was empty.
- **Report.** A section "MediaPipe Full in 3D": six sample frames with their 3D skeletons, three charts over time, and two tables.
- **Files.** `space.jpg`, `figures/space.png`, `space.csv` (the numbers) and `space_signals.csv` (every frame).
- **Code.** `poselab/space.py` holds the 3D measures; the drawing is `space_tile` in `poselab/draw.py`.

It appears whenever MediaPipe is ticked. There is no separate switch, and it costs no noticeable time: clip A with all five models and hands took 123 s for 897 frames, about 7 frames per second as before.

**The depth is a guess.** One ordinary camera records no distance. The network fills in the depth from what bodies usually look like, and the body's size in metres in the same way. These videos have no depth camera or motion-capture recording, so the 3D points cannot be checked against anything. What can be checked is whether they are consistent with themselves and with the picture, and that is what the measurements below do.

**What 3D could add: which way the arm goes up.** Arm elevation is the same for an arm raised to the side and an arm raised to the front: both go from 0° to 180°. The direction of the raise, seen from above and relative to the chest, tells them apart: 0° is straight out to the side, 90° is straight ahead. Biomechanics calls this the plane of elevation, and the plan listed it as an exploratory feature (`docs/feasibility_plan.md`, Section 3). It is read only while the arm is between 45° and 135°, because an arm that hangs or points overhead has almost no sideways or forward direction.

### One clip: clip A (front view, full light, expert)

| Elevation of the right arm | Lowest | Highest | Range | Repetitions | Jitter |
|---|---|---|---|---|---|
| Measured in the picture (2D) | 7° | 169° | 162° | 11 | 0.72° |
| From the 3D points | 11° | 158° | 148° | 11 | 0.90° |

- **The two angles** differ by a median of 3.3° frame by frame (95th percentile 12.1°).
- **Direction of the raise:** median 1°, middle half of the frames −3° to 5°. That is "to the side", as an abduction should be.
- **Bone length:** the 3D upper arm is 27.7 cm long with the arm down and 20.1 cm with the arm overhead. A real upper arm does not change length.

Two more example reports show the other cases. In the left side view `E01_P01_AL_VFL_GM` the two angles part company: range 176° in the picture, 144° from 3D, a median of 15° apart frame by frame. In `E01_P10_AF_VML_GF` the middle chart sits near 57° instead of 0° (point 4 below).

### All 436 development clips

No model was re-run for this: the Stage 1 files already hold MediaPipe Full's 3D points for every development clip. `scripts/stage1_space.py` reads them and writes one row per clip to `outputs/stage1/full_p1/space_clips.csv`, with the summary in `space_summary.txt`. A clip is counted when at least 90% of its frames are usable and the arm spends at least 30 frames between 45° and 135°. That leaves 266 clips from 36 participants.

| Kind of clip | Clips counted | Range in the picture | Range from 3D | Difference per frame | Direction of the raise | Same repetition count from both |
|---|---|---|---|---|---|---|
| Expert, front view | 88 of 100 | 162° | 154° | 3.3° | 1° (−2° to 4°) | 88 of 88 |
| Expert, left side view | 75 of 101 | 174° | 140° | 12.9° | 1° (−5° to 11°) | 71 of 75 |
| Expert, right side view | 16 of 97 | 170° | 146° | 9.3° | 8° (0° to 37°) | 13 of 16 |
| Non-expert, front view | 72 of 106 | 112° | 110° | 2.9° | −2° (−5° to 4°) | 63 of 72 |
| Non-expert, left side view | 9 of 16 | 113° | 100° | 5.1° | 9° (5° to 14°) | 9 of 9 |
| Non-expert, right side view | 6 of 16 | 111° | 99° | 3.6° | 0° (−5° to 4°) | 3 of 6 |

Every figure is the median over the clips counted. "Difference per frame" is each clip's median difference between the two angles. "Direction of the raise" is each clip's median direction, with the middle half of the clips in brackets.

**What it shows**

1. **From the front, the 3D angle follows the picture angle but stops short at the top.** In expert front-view clips the highest elevation is 170° in the picture and 163° from 3D, and the 3D range is the smaller one in 85 of 88 clips. With a sideways raise filmed from the front, the arm lies flat in the picture, so the picture angle is the one to trust here.
2. **The 3D skeleton changes shape as the arm moves.** In expert front-view clips its upper arm measures 26.4 cm with the arm down and 20.2 cm with the arm overhead. The distance between the 5th and 95th percentile of the upper-arm length is 33% of its median there, and 24% over all 266 clips; for the forearm it is 20% and 22%. A rigid bone would give about 0%. Stage 1 saw the same thing through its limb-length check.
3. **From the front, the direction of the raise reads "to the side" for abduction.** In 159 of the 160 front-view clips counted, the clip's median direction lies between −11° and 14°. The median over all of them is 0°.
4. **The one exception is a performance the judges rejected.** `E01_P10_AF_VML_GF` reads 57° towards the front (middle half of its frames 53° to 62°). All three judges marked that performance as not the right exercise. The only other front-view development clip with that mark, `E01_P31_AF_VHJ_GM`, is not among the 160: its arm spends just 13 frames in the band, and there it reads 1°. So the direction does not explain that second one.
   - *Two checks that do not use depth agree.* In the picture, this clip's arm shrinks to 0.41 of its longest apparent length, the lowest of all 206 front-view clips (median 0.86), and its wrist travels only 0.41 torso lengths sideways, also the lowest (median 0.92). An arm that shortens in the picture is pointing at the camera.
   - *Looking at it.* I cut six frames through one raise of this clip and of an ordinary clip of the same participant. In the ordinary clip the arm is stretched out to the side at shoulder height. In this clip it points at the camera.
   - *What follows.* The participant raises the arm in front of the body instead of out to the side. In Stage 1 I wrote that this clip looked like a normal set of raises; that was wrong, and `docs/stage1_report.md` now says so. It is one clip, and you should still watch it yourself.
5. **From the side, the 3D points do not hold up.** The same performance filmed from the front and from the left (70 expert performances) gives a 3D range that is 12° smaller from the left (in the middle half of the performances, 6.5° to 18° smaller), and a direction that differs by a median of 10°. From the right the direction is unusable: participant P10's four right-side clips read 92° to 95°, "straight ahead", while the front view of the same performances reads −10° to −2° for the three ordinary ones. In those right-side clips MediaPipe calls the moving arm the left one, although the front view shows the right arm moving. My reading is that the direction is then worked out from the wrong shoulder; I have not confirmed it. A measure that depended only on the body would not change with the camera.
6. **Counting repetitions gains nothing from 3D.** Both angles give the same count in 247 of the 266 clips.

**What to take from this**

- **Use the 3D panel to look, not to measure.** It makes plain what the model thinks the body is doing in depth, and where that is implausible.
- **Keep the picture angle for range of movement** in front views. The 3D angle stops about 8° lower at the top in expert clips and rests on a skeleton whose upper arm is about a quarter shorter overhead than hanging.
- **The direction of the raise is the promising part**, in front views only. It behaved correctly on abduction and picked out the one rejected performance, which elevation alone cannot do. It has been seen to read "forward" on exactly one clip, so it is a lead to test, not a result.
- **Do not use the 3D points from side views.**

**Not tried**

- **A movement that should read differently.** No exercise other than abduction is on this computer. Circumduction (E05), where the arm sweeps through every direction, is the first real test of the direction measure.
- **MediaPipe Heavy.** Google's largest pose model might give a steadier depth. You asked for Full only, so it is not downloaded.
- **3D from another model.** The `rtmlib` library that runs RTMPose here also contains a 3D version of it (its classes `RTMPose3d` and `Wholebody3d`). The weights are not downloaded and I have not tested it.

## 9. What these results do not show

- **Accuracy.** No true joint positions exist for these videos. Agreement between models is not correctness: all five could share an error.
- **Behaviour on hard clips.** Stage 1 found MediaPipe failing on right-side views (the arm hidden behind the body) and on some dark non-expert clips. None of the five clips here is of that kind.
- **A ranking.** Five clips from three participants cannot rank five models.
- **Phone performance.** Nothing was run on a phone.
- **Anything clinical.** A pose model reports where it thinks the joints are. Whether an exercise was done correctly is a separate question that these numbers do not touch.
- **Other exercises.** The movement measure is arm elevation, built for abduction. The other eight exercises will need their own measures.

## 10. Decisions made along the way

| What happened | What I did | Why |
|---|---|---|
| CMU's OpenPose program cannot run on a current graphics card, and its model server is offline | Wrote a reader for the Caffe files, rebuilt the network in PyTorch and wrote the decoding steps to follow CMU's code | Keeps OpenPose in the comparison without running an old binary. On clip A it finds one person per frame and lands within 2.2% to 3.4% of torso length of the other models, which shows the decoding works |
| The first OpenPose set up was the older 18-point model, without feet | Switched to CMU's 25-point model after you showed a picture of it finding the foot | The 25-point weights exist only in Caffe format. A mirror on Zenodo had them; the checksum matched the one CMU publishes |
| After the switch OpenPose was shaky on a low-resolution clip: participant found in 93% of frames, jitter 5.3° | Changed how the frame is shrunk before the network, to the method that averages pixels | My first resize skipped pixels, which made the network's maps flicker from frame to frame. After the fix: found in 99% of frames, jitter 1.1° |
| The first ViTPose set up (ViTPose++ Base through the `transformers` library) gave no feet: the library ships only the body output head | At your request, downloaded the whole-body ViTPose Base file from a community project | It is the one ViTPose file found that outputs feet. It has no official checksum |
| The new file's points came out shifted by about half a grid cell | Tried three standard ways of mapping its output back to the picture and kept the one that fits | The shift was the same for every point and grew with distance across the cut-out, which is the signature of a coordinate convention, not of a bad model. With the right one its body points are 1.4% to 2.1% of a torso length from ViTPose++ Base's |
| I first ran both ViTPose networks together: the old one for the body, the new one only for the feet | After you asked why two were needed, switched to the new one for everything and archived the old one | My reason for two was that the new file could not be checked against an official copy. The comparison with the old model had already answered that. One network is simpler to describe and faster: ViTPose went from 45 ms to 29 ms per frame |
| What the switch changed in ViTPose's results | Re-ran all five clips | The same repetition counts on every clip. Range of movement within 1° of before on every clip. Usable frames on the passer-by clip 93.0% against 93.3% |
| `ultralytics` and `rtmlib` both ask for a second copy of OpenCV | Installed both without their dependency lists | A second OpenCV would overwrite parts of the one MediaPipe was validated with |
| The first participant rule followed the passer-by out of the picture | Changed it twice; it now compares with the median position of the last five seconds | A median does not move when a passer-by is accepted for a few frames. On clip B usable frames for the top-down models rose from about 75% to 93% |
| The recording of the live view came to 40 MB for 10 seconds | Record every second frame at three-quarter size, about 1 MB per second | Windows' built-in encoder, as OpenCV drives it, spends a fixed amount per pixel. Other encoders were either not playable in a browser or far too slow |
| You asked for MediaPipe's 3D landmark estimation | Showed and measured the 3D points the program was already saving: a live panel, a report section, and a scan of the stored development clips | No download and no model change were needed. Stage 1 had found these points unsteady, so they are presented with that warning and kept out of the comparison between models |
| You asked that installing the requirements should also install the models | Added `setup.bat` (packages, then weights) and made the program download a missing model file by itself; MediaPipe's own file is now in the download list too | A requirements file can only install Python packages. The download code was tried against an empty folder: Google's MediaPipe file came out identical to the one in use, and a file that fails CMU's checksum is refused. The first `setup.bat` then failed on a second computer that had neither uv nor Python 3.11. It now downloads uv itself (checked against uv's published checksum), and uv downloads Python. That version was run on a fresh copy of the repository with uv and Python hidden: it installed all 68 packages and the weights, and the demo ran with the graphics card and with the card hidden (all five models then run on the processor, 5 to 15 times slower) |
| Stage 1 scripts had their own copy of the analysis code | Moved the analysis into `poselab/` and made the Stage 1 scripts use it | One set of rules for all models. The Full tables came out identical afterwards (436 clips, 3,032 repetition rows) |
| The flag name `low_visibility` only fits MediaPipe | Renamed it `low_confidence` | The other models report a confidence, not a visibility |

## 11. Cleanup

- **MediaPipe Lite is out of the working folders.** Its model file, pose files, tables, figures, overlay sheets, the Lite-against-Full comparison and the scripts as they were run are in `archive/2026-10-12_lite/` (315 MB). Nothing was deleted.
- **Stage 1 now means MediaPipe Full.** `outputs/stage1/` holds the Full tables, figures and 65 overlay sheets, regenerated with the shared code. `docs/stage1_report.md` is kept as the record of what was measured with both models; its pictures load from the archive.
- **Model files are sorted by model** in `models/<model>/`, with sources, licences and checksums in `models/sources.json`.
- **Setting up a fresh copy is one step.** `setup.bat` installs the packages of `requirements.txt` and then downloads the model weights; the program also fetches any model file that is missing when that model is first used (`poselab/weights.py`). A requirements file cannot do this by itself, because it can only install Python packages.
- **OpenPose's 18-point weights and the code that ran them** are in `archive/2026-10-12_openpose_18point/` (200 MB).
- **The ViTPose++ Base weights and the code that ran them** are in `archive/2026-10-12_vitpose_plus_base/` (478 MB).
- **Test runs made while building the demo** are in `archive/2026-10-12_demo_test_runs/` and can be deleted.

## 12. Next steps

1. **Try the program yourself** on a few clips, including ones Stage 1 flagged: a right-side view and a dark non-expert clip. `outputs/stage1/full_p1/qc_clips.csv` lists the usable share of every clip.
2. **Decide whether to run all five models over the development clips.** That would turn the five-clip impression into numbers per view, lighting and cohort. At the measured 7 to 9 frames per second, the 436 development clips of E01 would take about ten hours; that is an estimate from the frame count, and the batch script for it is not written.
3. **Decide which models stay in the running for the phone.** Only MediaPipe, ViTPose and RTMPose have licences that allow shipping; of those, ViTPose is the heaviest.
4. **Watch `E01_P10_AF_VML_GF`.** The 3D points and two measures in the picture say the arm is raised to the front, not to the side (Section 8). Your own eyes settle it. `outputs/demo/` has a report for it.
5. **Test the direction of the raise on circumduction (E05)** when the other exercises arrive. Until a movement that should read differently has been tried, it is a lead, not a feature.
6. **Still open from Stage 1:** your own repetition counts in `outputs/stage1/watch_log.csv`, and the other eight exercises from the same source as E01.

## 13. Adding another model

Each model is one file in `poselab/estimators/` with the same three methods: load the weights, `start` a new video, and `estimate` one frame, returning the participant's points with a score each. To add one, write that file, list the model's point names in `poselab/skeleton.py` if they are new, and add one line to the registry in `poselab/estimators/__init__.py`. The demo, the checks and the report then work for it without further changes. A model that also returns 3D points sets `has_3d` and fills `Pose.world`; it then gets the 3D panel and report section. Past five models the charts need a sixth colour.

## 14. Sources

- BlazePose (the model behind MediaPipe Pose): Bazarevsky and others, 2020, https://arxiv.org/abs/2006.10204
- MediaPipe Pose Landmarker and its world landmarks: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker . The origin at the middle of the hips and the directions of the three axes were checked on stored clips of this project
- ViTPose and ViTPose++: Xu, Zhang, Zhang and Tao, 2022, https://arxiv.org/abs/2212.04246 ; ViTPose++ Base weights, used until the switch and for checking the new file, https://huggingface.co/usyd-community/vitpose-plus-base
- OpenPose: Cao, Simon, Wei and Sheikh, 2017, https://arxiv.org/abs/1611.08050 ; licence https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE ; 25-point weights and definition from the mirror at https://zenodo.org/records/18290485 ; the checksum, the limb list and the default thresholds from CMU's repository (`CMakeLists.txt` and `src/openpose/pose/poseParameters.cpp`)
- ViTPose whole-body conversion, the ViTPose model now in use: https://huggingface.co/JunkyByte/easy_ViTPose (file `onnx/wholebody/vitpose-b-wholebody.onnx`); project page https://github.com/JunkyByte/easy_ViTPose
- RTMPose: Jiang and others, 2023, https://arxiv.org/abs/2303.07399 ; rtmlib https://github.com/Tau-J/rtmlib
- YOLO26 pose models: https://docs.ultralytics.com/tasks/pose/ ; licence https://www.ultralytics.com/license
- ONNX Runtime and PyTorch sharing GPU libraries: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
