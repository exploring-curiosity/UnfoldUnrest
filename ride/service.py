"""The ride service: the VM sends clips here (through a Cloudflare tunnel), the Mac runs them, the VM pulls the results.

python -m ride.service [--port 8791]          then   cloudflared tunnel --url http://localhost:8791

Every request but /health needs the token in SERVICE_TOKEN (.env): `Authorization: Bearer <token>`, or for a browser
a cookie set by signing in once at /login. The VM always starts the connection (the Mac cannot reach the VM), so it pushes
clips and pulls results; the manifest's hashes keep its mirror in step with runs/ here.

API
  GET  /health                         no token; is the service up
  POST /api/runs                       {"name", "source"} -> a new run waiting for its clip
  PUT  /api/runs/{id}/clip?offset=N    a chunk of the clip at byte N (Cloudflare caps a request at 100 MB)
  POST /api/runs/{id}/start            the clip is complete: queue the run
  POST /api/runs/{id}/rerun?start=S    run again from stage S (frames, geometry, detect, export) with today's code
  GET  /api/runs, /api/runs/{id}       run records
  GET  /api/manifest?all=1             every run's files with sizes and sha256 (all=1 adds frames, depth, clip)
Files, laid out as on disk (so the same viewer URLs work here and on the VM's mirror)
  GET  /runs/index.json                the run records
  GET  /runs/{id}/{path}               a run's file (byte ranges for the video)
  GET  /viewer/{file}                  the viewer; /viewer/runs.html lists the runs
"""
from __future__ import annotations

import hmac
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import Body, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from . import ROOT
from . import runs as R

CHUNK_MAX = 95 * 2 ** 20
CLIP_MAX = 4 * 2 ** 30
CLIP_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def env(name):
    for line in (ROOT / ".env").read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"').strip("'")
    return os.environ.get(name)


TOKEN = env("SERVICE_TOKEN")
if not TOKEN:
    sys.exit("SERVICE_TOKEN is not set in .env")


def auth(request: Request):
    h = request.headers.get("authorization", "")
    given = h[7:] if h.lower().startswith("bearer ") else request.cookies.get("ride_token", "")
    if not hmac.compare_digest(given.encode(), TOKEN.encode()):
        raise HTTPException(401, "missing or wrong token")


def rid_ok(rid):
    try:
        d = R.run_dir(rid)
    except ValueError:
        raise HTTPException(404, "no such run")
    if not (d / "run.json").exists():
        raise HTTPException(404, "no such run")
    return d


# -- the worker: one run at a time (one GPU) ---------------------------------------------------------------------
jobs: queue.Queue = queue.Queue()


def worker():
    while True:
        rid, start = jobs.get()
        d = R.run_dir(rid)
        envp = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1", PYTHONUNBUFFERED="1")
        with open(d / "log.txt", "a") as log:
            log.write(f"\n### {R.now()} run from {start} (code {R.code_version()})\n"); log.flush()
            p = subprocess.run([sys.executable, "-W", "ignore", "-m", "ride.pipeline", rid, "--from", start],
                               cwd=ROOT, env=envp, stdout=log, stderr=subprocess.STDOUT)
        if p.returncode != 0 and R.read(rid)["status"] != "failed":
            R.update(rid, status="failed", error=f"pipeline exited with {p.returncode}")
        jobs.task_done()


def enqueue(rid, start):
    R.update(rid, status="queued", queued_from=start, error=None)
    jobs.put((rid, start))


app = FastAPI(title="ride service")


@app.on_event("startup")
def resume():
    R.RUNS.mkdir(exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    for r in sorted(R.all_runs(), key=lambda r: r["created"]):
        if r["status"] in ("queued", "running"):       # interrupted by a restart: pick up where it was
            enqueue(r["id"], r.get("stage") or r.get("queued_from") or "frames")


@app.get("/health")
def health():
    return dict(ok=True, code_version=R.code_version(), queued=jobs.qsize())


LOGIN = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ride Sign In</title><style>body{margin:0;background:#2e3133;color:#f2f1ec;font:16px/1.5 system-ui,sans-serif}
form{max-width:420px;margin:12vh auto;padding:0 16px}input,button{font:inherit;width:100%;box-sizing:border-box;padding:10px;margin:8px 0;border-radius:4px;border:1px solid #44494c}
button{background:#f2f1ec;color:#2e3133;font-weight:600;cursor:pointer}</style></head><body><form method="post" action="/login">
<h1>Ride service</h1><label for="t">Service token</label><input id="t" name="token" type="password" autocomplete="current-password" required>
<button type="submit">Open the rides</button></form></body></html>"""


@app.get("/login")
def login_form():
    return HTMLResponse(LOGIN)


@app.post("/login")
def login(token: str = Form(...)):
    if not hmac.compare_digest(token.encode(), TOKEN.encode()):
        raise HTTPException(401, "wrong token")
    r = RedirectResponse("/viewer/runs.html", status_code=303)
    r.set_cookie("ride_token", token, httponly=True, secure=True, samesite="lax", max_age=7 * 86400)
    return r


@app.post("/api/runs", dependencies=[Depends(auth)])
def create(body: dict = Body(...)):
    name = str(body.get("name") or "clip")[:80]
    ext = Path(str(body.get("filename") or "clip.mp4")).suffix.lower() or ".mp4"
    if ext not in CLIP_EXT:
        raise HTTPException(400, f"clip type {ext} not accepted")
    src = body.get("source") if isinstance(body.get("source"), dict) else {}
    return R.create(name, src, clip_name="clip" + ext)


@app.put("/api/runs/{rid}/clip", dependencies=[Depends(auth)])
async def put_clip(rid: str, offset: int, request: Request):
    d = rid_ok(rid)
    rec = R.read(rid)
    if rec["status"] != "uploading":
        raise HTTPException(409, f"run is {rec['status']}, not taking a clip")
    p = d / rec["clip"]
    have = p.stat().st_size if p.exists() else 0
    if offset != have:
        raise HTTPException(409, f"expected offset {have}")
    n = 0
    with open(p, "ab") as f:
        async for b in request.stream():
            n += len(b)
            if n > CHUNK_MAX or have + n > CLIP_MAX:
                f.truncate(have)
                raise HTTPException(413, "chunk or clip too large")
            f.write(b)
    return dict(size=have + n)


@app.post("/api/runs/{rid}/start", dependencies=[Depends(auth)])
def start(rid: str):
    d = rid_ok(rid)
    rec = R.read(rid)
    if rec["status"] != "uploading" or not (d / rec["clip"]).exists():
        raise HTTPException(409, f"run is {rec['status']}" if rec["status"] != "uploading" else "no clip yet")
    enqueue(rid, "frames")
    return R.read(rid)


@app.post("/api/runs/{rid}/rerun", dependencies=[Depends(auth)])
def rerun(rid: str, start: str = "export"):
    rid_ok(rid)
    if start not in R.STAGES:
        raise HTTPException(400, f"stage must be one of {R.STAGES}")
    if R.read(rid)["status"] in ("queued", "running", "uploading"):
        raise HTTPException(409, "run is busy")
    enqueue(rid, start)
    return R.read(rid)


@app.get("/api/runs", dependencies=[Depends(auth)])
def list_runs():
    return R.all_runs()


@app.get("/api/runs/{rid}", dependencies=[Depends(auth)])
def get_run(rid: str):
    rid_ok(rid)
    return R.read(rid)


@app.get("/api/manifest", dependencies=[Depends(auth)])
def manifest(all: int = 0):
    return JSONResponse(R.manifest(everything=bool(all)))


@app.get("/runs/index.json", dependencies=[Depends(auth)])
def index():
    return R.all_runs()


def inside(base: Path, rel: str):
    p = (base / rel).resolve()
    if not p.is_relative_to(base.resolve()) or not p.is_file():
        raise HTTPException(404, "not found")
    return p


@app.get("/runs/{rid}/{path:path}", dependencies=[Depends(auth)])
def run_file(rid: str, path: str):
    return FileResponse(inside(rid_ok(rid), path), headers={"Cache-Control": "no-store"})


@app.get("/viewer/{path:path}", dependencies=[Depends(auth)])
def viewer(path: str):
    return FileResponse(inside(ROOT / "viewer", path), headers={"Cache-Control": "no-store"})


@app.get("/", dependencies=[Depends(auth)])
def home():
    return RedirectResponse("/viewer/runs.html")


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8791
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
