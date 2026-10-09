"""Runs: one folder per clip under runs/<id>/, with run.json tracking where it came from, what ran and what came out.

runs/<id>/clip.<ext>   the clip as sent
runs/<id>/run.json     id, name, source, status (uploading | queued | running | done | failed), stage, per-stage
                       timings, the code version that produced the results, a summary of them, the last error
runs/<id>/log.txt      the pipeline's output
runs/<id>/...          meta.json, frames/, geom.npz, dets.json, app/ (what the viewer reads)

The layout is the same on the Mac and on the VM's mirror, so viewer/ride.html?run=<id> works on both.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path

from . import ROOT

RUNS = ROOT / "runs"
ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z0-9-]{1,40}$")
STAGES = ["frames", "geometry", "detect", "export", "review"]
RESULTS = ("run.json", "log.txt", "meta.json", "dets.json")     # plus everything under app/
SHARED = ("viewer/ride.html", "viewer/runs.html")                # the viewer travels with the results


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def new_id(name):
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "clip"
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + slug


def run_dir(rid):
    if not ID_RE.match(rid or ""):
        raise ValueError(f"bad run id {rid!r}")
    return RUNS / rid


def code_version():
    """Short hash of the analysis code: which version produced a run's results."""
    h = hashlib.sha1()
    for p in sorted((ROOT / "ride").glob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:10]


def read(rid):
    return json.loads((run_dir(rid) / "run.json").read_text())


def write(rid, rec):
    p = run_dir(rid) / "run.json"
    tmp = p.with_suffix(".json.tmp")
    rec["updated"] = now()
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, p)
    return rec


def update(rid, **kw):
    rec = read(rid); rec.update(kw)
    return write(rid, rec)


def create(name, source=None, clip_name="clip.mp4"):
    rid = new_id(name)
    d = run_dir(rid); d.mkdir(parents=True)
    rec = dict(id=rid, name=name, created=now(), source=source or {}, clip=clip_name, status="uploading", stage=None,
               stages={}, error=None, code_version=None, summary=None)
    return write(rid, rec)


def all_runs():
    out = []
    for d in sorted(RUNS.glob("*/run.json")):
        try:
            out.append(json.loads(d.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(out, key=lambda r: r["created"], reverse=True)


def result_files(rid, everything=False):
    d = run_dir(rid)
    if everything:
        files = [p for p in d.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
    else:
        files = [d / f for f in RESULTS if (d / f).exists()] + [p for p in (d / "app").rglob("*") if p.is_file()]
    return sorted(files)


_sha = {}


def sha256(p):
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _sha:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        _sha[key] = h.hexdigest()
    return _sha[key]


def manifest(everything=False):
    """Every run's result files (path relative to the project root, size, sha256) and the shared viewer files."""
    runs = {}
    for r in all_runs():
        files = {}
        for p in result_files(r["id"], everything):
            files[str(p.relative_to(ROOT))] = dict(size=p.stat().st_size, sha256=sha256(p))
        runs[r["id"]] = dict(run=r, files=files)
    shared = {f: dict(size=(ROOT / f).stat().st_size, sha256=sha256(ROOT / f)) for f in SHARED if (ROOT / f).exists()}
    return dict(generated=now(), code_version=code_version(), runs=runs, shared=shared)
