"""Decode a phone video once: rotation applied, sampled at `fps`, centre-cropped to a square and resized to SIDE.

Every later stage (hands, cuts, LingBot depth) works on these square frames, so one pixel grid is shared. The crop's
place in the source frame is kept in meta.json so a step's clip can be shown from the original video."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import SIDE


def rotation(path) -> int:
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        try:
            rot = int(s.metadata.get("rotate", 0))
        except (TypeError, ValueError):
            rot = 0
    return rot % 360


def extract(video, out, fps=10.0, centre_y=.5):
    """-> meta dict; writes out/frames/%05d.jpg (SIDE x SIDE RGB) and out/meta.json."""
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
            img = fr.to_ndarray(format="rgb24")         # PyAV applies no rotation: phone video carries it as a tag
            rot = rotation(video)
            if rot:
                img = np.ascontiguousarray(np.rot90(img, k=-rot // 90))
            H, W = img.shape[:2]
            side = min(H, W)
            y0 = int(np.clip(round(centre_y * H - side / 2), 0, H - side)); x0 = (W - side) // 2
            sq = cv2.resize(img[y0:y0 + side, x0:x0 + side], (SIDE, SIDE), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(fdir / f"{k:05d}.jpg"), cv2.cvtColor(sq, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            times.append(round(t, 4)); k += 1
    meta = dict(video=str(Path(video).resolve()), fps=fps, n=k, times=times, src_hw=[H, W],
                crop=dict(x0=x0, y0=y0, side=side), side=SIDE)
    (out / "meta.json").write_text(json.dumps(meta))
    return meta


def load(out):
    """-> (meta, frames uint8 (n, SIDE, SIDE, 3) RGB)."""
    out = Path(out)
    meta = json.loads((out / "meta.json").read_text())
    fr = np.stack([cv2.cvtColor(cv2.imread(str(out / "frames" / f"{k:05d}.jpg")), cv2.COLOR_BGR2RGB) for k in range(meta["n"])])
    return meta, fr


if __name__ == "__main__":
    import sys
    m = extract(sys.argv[1], sys.argv[2], fps=float(sys.argv[3]) if len(sys.argv) > 3 else 10.0)
    print(m["n"], "frames", m["src_hw"], m["crop"])
