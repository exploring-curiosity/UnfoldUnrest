"""Camera pose, depth and confidence for every frame: upstream LingBot-Map on MPS (ElideDB's port, handoff 4.1).

pose_enc per frame is camera-to-world (translation, quaternion xyzw, fov_h, fov_w); depth and confidence are on the
SIDE x SIDE grid of the frames, in one world frame (and one arbitrary scale) for the whole video."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from . import ELIDE, SIDE


def run(out, size=518):
    sys.path.insert(0, str(ELIDE / "scripts"))
    import lingbot_upstream as LU
    out = Path(out)
    paths = sorted((out / "frames").glob("*.jpg"))
    t = time.time()
    model = LU.build(size=size)
    imgs = LU.load(paths, size=size)
    assert imgs.shape[-2:] == (size, size), imgs.shape
    pose, depth, conf = LU.stream(model, imgs, g=SIDE)
    np.savez_compressed(out / "geom.npz", pose=pose.astype(np.float32), depth=depth.astype(np.float16),
                        conf=conf.astype(np.float16), peak_gb=model.cfg.get("peak_gb", 0), secs=time.time() - t)
    print(f"geometry: {len(paths)} frames in {time.time() - t:.0f} s, peak {model.cfg.get('peak_gb')} GB")


def load(out):
    z = np.load(Path(out) / "geom.npz")
    return z["pose"].astype(np.float64), z["depth"].astype(np.float32), z["conf"].astype(np.float32)


if __name__ == "__main__":
    run(sys.argv[1])
