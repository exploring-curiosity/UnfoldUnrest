"""Decode a ride once: sampled at `fps`, aspect kept, `width` px wide. meta.json keeps the times."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def extract(video, out, fps=5.0, width=1280):
    import av
    out = Path(out); fdir = out / "frames"; fdir.mkdir(parents=True, exist_ok=True)
    times, k, nxt = [], 0, 0.0
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            if t + 1e-6 < nxt:
                continue
            nxt += 1.0 / fps
            img = fr.to_ndarray(format="rgb24")
            H, W = img.shape[:2]
            h = int(round(H * width / W))
            img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(fdir / f"{k:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            times.append(round(t, 4)); k += 1
    meta = dict(video=str(Path(video).resolve()), fps=fps, n=k, times=times, size=[width, h], src=[W, H])
    (out / "meta.json").write_text(json.dumps(meta))
    return meta


def load(out, k=None):
    out = Path(out)
    meta = json.loads((out / "meta.json").read_text())
    rd = lambda i: cv2.cvtColor(cv2.imread(str(out / "frames" / f"{i:05d}.jpg")), cv2.COLOR_BGR2RGB)
    if k is not None:
        return meta, rd(k)
    return meta, np.stack([rd(i) for i in range(meta["n"])])


if __name__ == "__main__":
    import sys
    m = extract(sys.argv[1], sys.argv[2], fps=float(sys.argv[3]) if len(sys.argv) > 3 else 5.0)
    print(m["n"], "frames", m["size"])
