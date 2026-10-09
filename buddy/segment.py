"""Class-agnostic cuts per frame: FastSAM-s (ONNX on CoreML, ElideDB's model) on the SIDE frame resized to 640.

Masks come from the prototype logits upsampled to SIDE (not the 160 px cells), cut to each detection's box. A cut that
repeats a better-scored one (IoU > .5) is the same cut. Cuts bigger than a quarter of the frame are surfaces (the
table), not parts, and are dropped. Saved per frame as packed bits."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import ELIDE, SIDE

ONNX = ELIDE / "data/segment/o55/FastSAM-s.onnx"
CONF, SAME, MAX_AREA, MIN_AREA = .1, .5, .25, 30


def session():
    import onnxruntime as ort
    return ort.InferenceSession(str(ONNX), providers=[
        ("CoreMLExecutionProvider", {"MLComputeUnits": "CPUAndNeuralEngine", "ModelFormat": "MLProgram"}),
        "CPUExecutionProvider"])


def cut_frame(sess, rgb, nms):
    import torch
    x = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
    o = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32) / 255.0})
    det = nms.non_max_suppression(torch.from_numpy(o[0]), CONF, .9, agnostic=True, max_det=300, nc=1)[0].numpy()
    det = det[np.argsort(-det[:, 4])]
    proto = o[1][0].astype(np.float32)                                   # (32, 160, 160)
    masks, confs, kept_f = [], [], []
    for d in det:
        L = (d[6:] @ proto.reshape(32, -1)).reshape(160, 160)
        L = cv2.resize(L, (SIDE, SIDE), interpolation=cv2.INTER_LINEAR)
        b = d[:4] * SIDE / 640.0
        m = np.zeros((SIDE, SIDE), bool)
        x0, y0 = int(max(np.floor(b[0]), 0)), int(max(np.floor(b[1]), 0))
        x1, y1 = int(min(np.ceil(b[2]), SIDE)), int(min(np.ceil(b[3]), SIDE))
        m[y0:y1, x0:x1] = L[y0:y1, x0:x1] > 0
        a = int(m.sum())
        if a < MIN_AREA or a > MAX_AREA * SIDE * SIDE:
            continue
        f = m.ravel()
        if any((f & g).sum() / max((f | g).sum(), 1) > SAME for g in kept_f):
            continue
        kept_f.append(f); masks.append(m); confs.append(float(d[4]))
    return np.array(masks, bool).reshape(-1, SIDE, SIDE), np.array(confs, np.float32)


def run(out):
    from tqdm import tqdm
    from ultralytics.utils import nms
    from . import frames as F
    meta, fr = F.load(out)
    sess = session()
    allm, conf, frame = [], [], []
    for k in tqdm(range(len(fr)), desc="cuts"):
        m, c = cut_frame(sess, fr[k], nms)
        allm.append(m); conf.append(c); frame.append(np.full(len(m), k))
    M = np.concatenate(allm)
    np.savez_compressed(Path(out) / "cuts.npz", masks=np.packbits(M, axis=-1), conf=np.concatenate(conf),
                        frame=np.concatenate(frame))
    print(len(M), "cuts over", len(fr), "frames")


def load(out):
    """-> list over frames of (masks (k, SIDE, SIDE) bool, conf (k,))."""
    z = np.load(Path(out) / "cuts.npz")
    M = np.unpackbits(z["masks"], axis=-1)[..., :SIDE].astype(bool)
    n = int(z["frame"].max()) + 1 if len(z["frame"]) else 0
    return [(M[z["frame"] == k], z["conf"][z["frame"] == k]) for k in range(n)]


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
