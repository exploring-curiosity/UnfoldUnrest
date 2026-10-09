"""What moves on its own, per frame: DIS optical flow from the previous frame, the camera's motion fitted as one
homography over the whole frame (RANSAC; the table dominates the view), and the pixels whose flow disagrees with it by
more than RES px. Hands and parts being handled are what moves on their own; a frame with (almost) none is at rest.

Replaces a hand detector: MediaPipe read the studded round plate as a palm and missed the real (close-up, partial)
hands, while 'the scene is static apart from the camera' is exactly what a rest state needs."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import SIDE

RES, SCALE = 1.5, 2          # residual threshold in px at SIDE / SCALE


def moving(frames):
    """-> masks (n, SIDE, SIDE) bool of independently moving pixels; share (n,) of the frame that moves."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    g = SIDE // SCALE
    grey = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (g, g), interpolation=cv2.INTER_AREA) for f in frames]
    ys, xs = np.mgrid[0:g:4, 0:g:4]
    P = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    masks = np.zeros(frames.shape[:3], bool); share = np.zeros(len(frames))
    for k in range(len(frames)):
        a, b = grey[max(k - 1, 0)], grey[k] if k else grey[min(1, len(grey) - 1)]
        fl = dis.calc(a, b, None)
        Q = P + fl[P[:, 1].astype(int), P[:, 0].astype(int)]
        H, _ = cv2.findHomography(P, Q, cv2.RANSAC, 1.0)
        if H is None:
            continue
        yy, xx = np.mgrid[0:g, 0:g].astype(np.float32)
        pred = cv2.perspectiveTransform(np.stack([xx, yy], -1).reshape(-1, 1, 2), H).reshape(g, g, 2) - np.stack([xx, yy], -1)
        r = np.linalg.norm(fl - pred, axis=-1) > RES
        r = cv2.morphologyEx(r.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(r)
        keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 40
        m = cv2.resize(keep[lab].astype(np.uint8), (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)
        masks[k] = cv2.dilate(m, np.ones((11, 11), np.uint8)) > 0
        share[k] = keep[lab].mean()
    return masks, share


def run(out):
    from . import frames as F
    meta, fr = F.load(out)
    masks, share = moving(fr)
    np.savez_compressed(Path(out) / "motion.npz", masks=np.packbits(masks, axis=-1), share=share)
    return masks, share


def load(out):
    z = np.load(Path(out) / "motion.npz")
    return np.unpackbits(z["masks"], axis=-1)[..., :SIDE].astype(bool), z["share"]


if __name__ == "__main__":
    import sys
    m, s = run(sys.argv[1])
    print(" ".join(f"{x:.2f}" for x in s))
