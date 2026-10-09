"""Street things per frame: OWLv2 (open vocabulary, google/owlv2-base-patch16-ensemble, already in the HF cache).

One text query per kind; boxes in the frame's pixels (1280 x 720). Open vocabulary because the things that matter
for daylighting (crosswalks, planters, kiosks, shelters) are not in a fixed detector's classes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL = "google/owlv2-base-patch16-ensemble"
QUERIES = {
    "crosswalk": "a pedestrian crosswalk: wide parallel white stripes across the road",
    "raised crosswalk": "a raised crosswalk lit with a row of small lights",
    # other road paint competes with the crosswalk queries, so bike symbols and arrows are not called crosswalks
    "bike symbol": "a bicycle symbol painted on the road",
    "arrow": "an arrow painted on the road",
    "road text": "words painted on the road",
    "lane line": "a painted lane line",
    "car": "a car",
    "truck": "a truck",
    "van": "a van",
    "bus": "a bus",
    "bicycle": "a bicycle",
    "person": "a person",
    "bollard": "a bollard",
    "traffic light": "a traffic light",
    "fire hydrant": "a fire hydrant",
    # street furniture: what can hide a person at the curb from oncoming traffic (and what cannot, for labels)
    "planter": "a large planter box with bushes",
    "hedge": "a hedge",
    "bus shelter": "a bus shelter",
    "kiosk": "a kiosk or newsstand",
    "utility box": "a metal utility cabinet",
    "bench": "a bench",
    "trash can": "a trash can",
    "tree": "a tree",
    "pole": "a street lamp or sign pole",
}
THRESH = {"crosswalk": .15, "raised crosswalk": .15, "bike symbol": .15, "arrow": .15, "road text": .15, "lane line": .15, "car": .2, "truck": .2, "van": .2, "bus": .2, "bicycle": .2, "person": .2,
          "bollard": .2, "traffic light": .2, "fire hydrant": .2, "planter": .2, "hedge": .22, "bus shelter": .2, "kiosk": .2,
          "utility box": .2, "bench": .2, "trash can": .2, "tree": .22, "pole": .2}


def load_model(device="mps"):
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(device).eval()
    return proc, model, device


def nms(boxes, scores, iou=.5):
    order = np.argsort(-scores); keep = []
    while len(order):
        i = order[0]; keep.append(i)
        b = boxes[order[1:]]
        xx0 = np.maximum(boxes[i, 0], b[:, 0]); yy0 = np.maximum(boxes[i, 1], b[:, 1])
        xx1 = np.minimum(boxes[i, 2], b[:, 2]); yy1 = np.minimum(boxes[i, 3], b[:, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        order = order[1:][inter / np.maximum(a + ab - inter, 1e-9) < iou]
    return keep


def detect(proc, model, device, img):
    """img (H, W, 3) RGB -> list of dict(kind, score, box [x0, y0, x1, y1])."""
    import torch
    names = list(QUERIES)
    H, W = img.shape[:2]
    inp = proc(text=[[QUERIES[n] for n in names]], images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    # OWLv2 pads the image to a square: boxes come back relative to the padded side
    side = max(H, W)
    res = proc.post_process_object_detection(out, threshold=min(THRESH.values()),
                                              target_sizes=torch.tensor([[side, side]], device=device))[0]
    B = res["boxes"].float().cpu().numpy(); S = res["scores"].float().cpu().numpy(); L = res["labels"].cpu().numpy()
    dets = []
    for li, n in enumerate(names):
        m = (L == li) & (S >= THRESH[n])
        if not m.any():
            continue
        b, s = B[m], S[m]
        for i in nms(b, s):
            x0, y0, x1, y1 = np.clip(b[i], 0, [W, H, W, H])
            if x1 - x0 > 2 and y1 - y0 > 2:
                dets.append(dict(kind=n, score=round(float(s[i]), 3), box=[round(float(v), 1) for v in (x0, y0, x1, y1)]))
    return dets


def run(out, every=1):
    from tqdm import tqdm
    from . import frames as F
    out = Path(out)
    meta, fr = F.load(out)
    proc, model, device = load_model()
    allr = {}
    for k in tqdm(range(0, meta["n"], every), desc="owlv2"):
        allr[k] = detect(proc, model, device, fr[k])
    (out / "dets.json").write_text(json.dumps(allr))
    return allr


if __name__ == "__main__":
    import sys
    r = run(sys.argv[1])
    from collections import Counter
    print(Counter(d["kind"] for v in r.values() for d in v))
