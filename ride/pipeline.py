"""One run end to end: frames -> geometry (LingBot) -> detect (OWLv2) -> export (map, objects, daylighting, app files)
-> review (a vision model on W&B Inference checks each finding against the frames).

python -m ride.pipeline <run id> [--from STAGE]

Each stage's start, end and seconds go into run.json as it goes; a failure records the stage and the error. Run as
its own process by the service, so each run starts with the GPU memory free. Every run is traced to W&B Weave (the
project in WANDB_PROJECT): the stages with their inputs, outputs and timings, and the vision model's prompts and answers.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import weave

from . import ROOT
from . import runs as R

MAX_FRAMES = 400          # LingBot's memory grows with the frame count: long clips are sampled more sparsely
FPS = 5.0


def load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() and not k.lstrip().startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    missing = [k for k in ("WANDB_API_KEY", "WANDB_PROJECT") if not os.environ.get(k)]
    if missing:
        sys.exit(f"set {' and '.join(missing)} in .env (WANDB_PROJECT is <team>/<project>)")


@weave.op
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
    return dict(n=m["n"], fps=round(fps, 2), duration=round(dur, 1), size=m["size"])


@weave.op
def geometry(d, rec):
    from . import geometry as GE
    GE.run(d)
    pose, depth, _ = GE.load(d)
    return dict(frames=len(pose), depth_grid=list(depth.shape[1:]))


@weave.op
def detect(d, rec):
    from collections import Counter
    from . import detect as DE
    r = DE.run(d)
    n = dict(Counter(x["kind"] for v in r.values() for x in v))
    print("detect:", n, flush=True)
    return n


@weave.op
def export(d, rec):
    from . import export as EX
    data = EX.run(d)
    return data["stats"]


@weave.op
def review(d, rec):
    from . import review as RV
    return RV.run(str(d))


@weave.op
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
        if st in ("export", "review"):
            rec["summary"] = out
        R.write(rid, rec)
    R.update(rid, status="done", stage=None)
    return True


if __name__ == "__main__":
    args = sys.argv[1:]
    start = args[args.index("--from") + 1] if "--from" in args else "frames"
    load_env()
    weave.init(os.environ["WANDB_PROJECT"])
    rec = R.read(args[0])
    with weave.attributes(dict(run_id=args[0], name=rec["name"], source=rec.get("source") or {}, start=start,
                               code_version=R.code_version())):
        ok = run(args[0], start)
    weave.finish()
    sys.exit(0 if ok else 1)
