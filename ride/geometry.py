"""Upstream LingBot-Map (ElideDB's MPS port) over the ride's frames: camera-to-world pose, depth, confidence."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from . import ELIDE, G


def run(out, size=518):
    sys.path.insert(0, str(ELIDE / "scripts"))
    import lingbot_upstream as LU
    out = Path(out)
    paths = sorted((out / "frames").glob("*.jpg"))
    t = time.time()
    model = LU.build(size=size)
    imgs = LU.load(paths, size=size)          # crop mode: 518 wide, height kept to the aspect (multiple of 14)
    print("model input", tuple(imgs.shape), flush=True)
    pose, depth, conf = LU.stream(model, imgs, g=G)
    np.savez_compressed(out / "geom.npz", pose=pose.astype(np.float32), depth=depth.astype(np.float16),
                        conf=conf.astype(np.float16), input_hw=np.array(imgs.shape[-2:]), secs=time.time() - t)
    print(f"geometry: {len(paths)} frames in {time.time() - t:.0f} s, peak {model.cfg.get('peak_gb')} GB", flush=True)


def load(out):
    z = np.load(Path(out) / "geom.npz")
    return z["pose"].astype(np.float64), z["depth"].astype(np.float32), z["conf"].astype(np.float32)


if __name__ == "__main__":
    run(sys.argv[1])
