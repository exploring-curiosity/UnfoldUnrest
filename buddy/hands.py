"""Where the builder's hands are, per frame: MediaPipe's hand landmarker (21 points a hand) -> a hand mask (each
finger's chain and the palm, as thick strokes and a filled hull) on the SIDE grid, and whether any hand is in view.

Hands are not parts, so a hand-specific model does not break the class-agnostic rule for parts."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import ROOT, SIDE

MODEL = ROOT / "models/hand_landmarker.task"
CHAINS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (0, 17, 18, 19, 20), (5, 9, 13, 17)]


def detect(frames, min_conf=.3):
    """frames (n, SIDE, SIDE, 3) RGB -> masks (n, SIDE, SIDE) bool, counts (n,) hands found."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision as V
    opts = V.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)), num_hands=2,
                                   running_mode=V.RunningMode.VIDEO, min_hand_detection_confidence=min_conf,
                                   min_hand_presence_confidence=min_conf, min_tracking_confidence=min_conf)
    masks = np.zeros(frames.shape[:3], bool); counts = np.zeros(len(frames), int)
    with V.HandLandmarker.create_from_options(opts) as lm:
        for k, fr in enumerate(frames):
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr)), k * 100)
            counts[k] = len(r.hand_landmarks)
            for hand in r.hand_landmarks:
                P = np.array([[p.x * SIDE, p.y * SIDE] for p in hand])
                masks[k] |= paint(P)
    return masks, counts


def paint(P):
    """A hand's mask from its 21 landmarks: palm hull filled, each finger a stroke as wide as the knuckle spacing."""
    m = np.zeros((SIDE, SIDE), np.uint8)
    w = max(int(np.linalg.norm(P[5] - P[17]) / 3), 6)
    palm = P[[0, 1, 2, 5, 9, 13, 17]].astype(np.int32)
    cv2.fillConvexPoly(m, cv2.convexHull(palm), 1)
    for ch in CHAINS:
        for a, b in zip(ch[:-1], ch[1:]):
            cv2.line(m, tuple(P[a].astype(int)), tuple(P[b].astype(int)), 1, w)
    # the forearm leaves the frame from the wrist, away from the fingers
    d = P[0] - P[9]
    if np.linalg.norm(d) > 1:
        cv2.line(m, tuple(P[0].astype(int)), tuple((P[0] + d / np.linalg.norm(d) * SIDE).astype(int)), 1, 3 * w)
    return m > 0


def run(out):
    from . import frames as F
    meta, fr = F.load(out)
    masks, counts = detect(fr)
    np.savez_compressed(Path(out) / "hands.npz", masks=np.packbits(masks, axis=-1), counts=counts)
    return masks, counts


def load(out):
    z = np.load(Path(out) / "hands.npz")
    return np.unpackbits(z["masks"], axis=-1)[..., :SIDE].astype(bool), z["counts"]


if __name__ == "__main__":
    import sys
    m, c = run(sys.argv[1])
    print("".join(str(min(x, 9)) for x in c))



def skin(frames, seed, still, min_blob=.004, ratio_min=3.0):
    """The builder's own skin colour, learned from this video: an (a*, b*) chroma histogram of the pixels that move on
    their own (seed: mostly hands) against the pixels that stay put (still), as a back-projected ratio (Swain & Ballard;
    Jones & Rehg's skin/non-skin ratio). Chroma only, so lighting (L*) does not matter, and the cream plate and grey
    cloth (near-neutral chroma) stay out. -> skin masks (n, SIDE, SIDE) bool, hand-present flags (n,)."""
    lab = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2LAB) for f in frames])[..., 1:].astype(np.int32)
    H_s = np.ones((256, 256)); H_b = np.ones((256, 256))
    np.add.at(H_s, (lab[seed][:, 0], lab[seed][:, 1]), 1)
    np.add.at(H_b, (lab[still][:, 0], lab[still][:, 1]), 1)
    H_s = cv2.GaussianBlur(H_s, (5, 5), 1); H_b = cv2.GaussianBlur(H_b, (5, 5), 1)
    ratio = (H_s / H_s.sum()) / (H_b / H_b.sum())
    out = np.zeros(frames.shape[:3], bool); present = np.zeros(len(frames), bool)
    for k in range(len(frames)):
        m = (ratio[lab[k, ..., 0], lab[k, ..., 1]] > ratio_min).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab_, st, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_blob * SIDE * SIDE
        out[k] = cv2.dilate(keep[lab_].astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        present[k] = keep.any()
    return out, present
