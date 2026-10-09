"""One run end to end: frames -> geometry (LingBot) -> detect (OWLv2) -> export (map, objects, daylighting, app files).

python -m ride.pipeline <run id> [--from STAGE]

Each stage's start, end and seconds go into run.json as it goes; a failure records the stage and the error. Run as
its own process by the service, so each run starts with the GPU memory free.
"""
from __future__ import annotations

import sys
import time
import traceback

from . import runs as R

MAX_FRAMES = 400          # LingBot's memory grows with the frame count: long clips are sampled more sparsely
FPS = 5.0


def frames(d, rec):
    import av
    from . import frames as F
    clip = d / rec["clip"]
    with av.open(str(clip)) as c:
        s = c.streams.video[0]
        dur = float(s.duration * s.time_base) if s.duration else float(c.duration or 0) / 1e6
    fps = min(FPS, MAX_FRAMES / dur) if dur > 0 else FPS
    m = F.extract(clip, d, fps=fps)
    print(f"frames: {m['n']} at {fps:.2f} fps from {dur:.1f} s, {m['size']}", flush=True)


def geometry(d, rec):
    from . import geometry as GE
    GE.run(d)


def detect(d, rec):
    from collections import Counter
    from . import detect as DE
    r = DE.run(d)
    print("detect:", dict(Counter(x["kind"] for v in r.values() for x in v)), flush=True)


def export(d, rec):
    from . import export as EX
    data = EX.run(d)
    return data["stats"]


def run(rid, start="frames"):
    d = R.run_dir(rid)
    rec = R.read(rid)
    todo = R.STAGES[R.STAGES.index(start):]
    R.update(rid, status="running", error=None, code_version=R.code_version())
    for st in todo:
        t0 = time.time()
        rec = R.read(rid); rec["stage"] = st
        rec["stages"][st] = dict(start=R.now()); R.write(rid, rec)
        print(f"== {st} ==", flush=True)
        try:
            out = globals()[st](d, rec)
        except Exception as e:
            rec = R.read(rid)
            rec["stages"][st].update(end=R.now(), secs=round(time.time() - t0, 1), ok=False)
            R.write(rid, dict(rec, status="failed", error=f"{st}: {type(e).__name__}: {e}"))
            traceback.print_exc()
            return False
        rec = R.read(rid)
        rec["stages"][st].update(end=R.now(), secs=round(time.time() - t0, 1), ok=True)
        if st == "export":
            rec["summary"] = out
        R.write(rid, rec)
    R.update(rid, status="done", stage=None)
    return True


if __name__ == "__main__":
    args = sys.argv[1:]
    start = args[args.index("--from") + 1] if "--from" in args else "frames"
    ok = run(args[0], start)
    sys.exit(0 if ok else 1)
