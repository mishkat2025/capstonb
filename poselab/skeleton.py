"""One joint vocabulary for every pose model.

Each model reports its own set of points in its own order (17 to 33 of them). Analysis
and drawing use the 27 joints below. A model that has no such point (for example no heel)
leaves it as NaN, so code that needs a joint must cope with it being absent.

The last six are the rough hand points only MediaPipe's pose model has: the knuckle of the
little finger, the knuckle of the index finger and the thumb. They show which way the hand
points, not whether it is open. Fingertips come from the separate hand model in hands.py.

Two points with the same name are not guaranteed to be the same anatomical spot: the
models were trained on different annotation rules, so a steady offset of a few pixels
between two models is a difference of convention, not an error of either. The toe is the
clearest case: it is the big toe for RTMPose and the "foot index" (tip of the foot) for
MediaPipe.
"""
import numpy as np

JOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
          "left_heel", "right_heel", "left_toe", "right_toe",
          "left_pinky_knuckle", "right_pinky_knuckle", "left_index_knuckle", "right_index_knuckle", "left_thumb", "right_thumb"]
J = {name: i for i, name in enumerate(JOINTS)}

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = J["left_shoulder"], J["right_shoulder"], J["left_hip"], J["right_hip"]
ARM = {side: (J[f"{side}_shoulder"], J[f"{side}_elbow"], J[f"{side}_wrist"]) for side in ("left", "right")}
# the twelve joints every model reports; agreement between models is measured on these
BODY = [J[f"{side}_{part}"] for part in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle") for side in ("left", "right")]


def _other_side(name):
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


MIRROR = np.array([J[_other_side(name)] for name in JOINTS])     # index of the same joint on the other side

# (joint a, joint b, side) for drawing
BONES = [(J[f"{s}_{a}"], J[f"{s}_{b}"], s) for s in ("right", "left")
         for a, b in (("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"),
                      ("ankle", "heel"), ("heel", "toe"), ("ankle", "toe"),
                      ("wrist", "index_knuckle"), ("wrist", "pinky_knuckle"), ("index_knuckle", "pinky_knuckle"), ("wrist", "thumb"))]
BONES += [(L_SHOULDER, R_SHOULDER, "mid"), (L_HIP, R_HIP, "mid")]
DRAWN_JOINTS = [i for i, name in enumerate(JOINTS) if name.split("_")[-1] not in ("nose", "eye", "ear")]

# ---- the point lists of each model, in the order the model returns them ----
COCO17 = JOINTS[:17]
HALPE26 = COCO17 + ["head", "neck", "pelvis", "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
                    "left_heel", "right_heel"]
COCO_WHOLEBODY_BODY_FEET = COCO17 + ["left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"]
OPENPOSE25 = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist",
              "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
              "right_eye", "left_eye", "right_ear", "left_ear",
              "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"]
MEDIAPIPE33 = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
               "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
               "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
               "left_foot_index", "right_foot_index"]

SAME_AS = {"left_foot_index": "left_toe", "right_foot_index": "right_toe", "left_big_toe": "left_toe", "right_big_toe": "right_toe",
           "left_pinky": "left_pinky_knuckle", "right_pinky": "right_pinky_knuckle",
           "left_index": "left_index_knuckle", "right_index": "right_index_knuckle"}


def to_canonical(values, names):
    """Reorder an array [..., K, D] of a model's own points into the shared joints; absent joints become NaN."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape[:-2] + (len(JOINTS), values.shape[-1]), np.nan)
    for k, name in enumerate(names):
        name = SAME_AS.get(str(name), str(name))
        if name in J:
            out[..., J[name], :] = values[..., k, :]
    return out
