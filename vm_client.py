#!/usr/bin/env python3
"""VM side of the ride service: send clips to the Mac through its Cloudflare tunnel and keep a mirror of every run.

Python 3.8+, standard library only (nothing to install on the VM).

  export RIDE_URL=https://<name>.trycloudflare.com     # printed by cloudflared on the Mac; changes when it restarts
  export RIDE_TOKEN=...                                 # SERVICE_TOKEN from the Mac's .env

  python3 vm_client.py health
  python3 vm_client.py submit ride.mp4 --name "west st" --source '{"vast_video_id": "..."}' [--wait]
  python3 vm_client.py runs
  python3 vm_client.py rerun <run id> --from export
  python3 vm_client.py sync  [--dest ride_mirror] [--all]
  python3 vm_client.py watch [--dest ride_mirror] [--every 30]

The mirror has the Mac's layout: <dest>/runs/<id>/... and <dest>/viewer/. Open <dest>/viewer/runs.html (serve the
folder, e.g. `python3 -m http.server -d <dest>`) to browse the runs here. sync only fetches files whose sha256 changed,
writes each atomically, and logs what it did to <dest>/sync_log.jsonl.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 64 * 2 ** 20          # under Cloudflare's 100 MB request cap
UA = "ride-vm-client/1"


def cfg():
    url, tok = os.environ.get("RIDE_URL", "").rstrip("/"), os.environ.get("RIDE_TOKEN", "")
    if not url or not tok:
        sys.exit("set RIDE_URL and RIDE_TOKEN")
    return url, tok


def call(method, path, body=None, ctype="application/json", raw=False, timeout=600):
    url, tok = cfg()
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    req = urllib.request.Request(url + path, data=data, method=method,
                                 headers={"Authorization": "Bearer " + tok, "User-Agent": UA, "Content-Type": ctype})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        if raw:
            return r                                   # the caller streams and closes it
        with r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path}: HTTP {e.code} {e.read()[:300].decode(errors='replace')}")


def health(a):
    url, _ = cfg()
    req = urllib.request.Request(url + "/health", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        print(json.loads(r.read()))


def submit(a):
    clip = Path(a.clip)
    src = json.loads(a.source) if a.source else {}
    src.setdefault("vm_path", str(clip.resolve()))
    rec = call("POST", "/api/runs", dict(name=a.name or clip.stem, filename=clip.name, source=src))
    rid, size, off = rec["id"], clip.stat().st_size, 0
    with open(clip, "rb") as f:
        while off < size:
            b = f.read(CHUNK)
            r = call("PUT", f"/api/runs/{rid}/clip?offset={off}", b, ctype="application/octet-stream")
            off = r["size"]
            print(f"\r{rid}: sent {off / 2 ** 20:.0f} of {size / 2 ** 20:.0f} MB", end="", flush=True)
    print()
    rec = call("POST", f"/api/runs/{rid}/start")
    print(f"{rid}: {rec['status']}")
    if a.wait:
        wait(rid)
        sync(a)


def wait(rid, every=10):
    last = None
    while True:
        r = call("GET", f"/api/runs/{rid}")
        now = (r["status"], r.get("stage"))
        if now != last:
            print(f"{rid}: {r['status']}" + (f" ({r['stage']})" if r.get("stage") else ""), flush=True)
            last = now
        if r["status"] in ("done", "failed"):
            if r["status"] == "failed":
                print("error:", r.get("error"))
            return r
        time.sleep(every)


def runs(a):
    for r in call("GET", "/api/runs"):
        s = r.get("summary") or {}
        print(f"{r['id']}  {r['status']:<9} {r.get('stage') or '':<9} code {r.get('code_version') or '-':<10} "
              + (f"{s.get('crosswalks', '?')} crossings, {s.get('crossings_judged', '?')} judged, "
                 f"{s.get('crossings_exposed', '?')} with a side hidden from the rider at stopping distance" if s else ""))


def rerun(a):
    r = call("POST", f"/api/runs/{a.run}/rerun?start={a.start}")
    print(f"{r['id']}: {r['status']} from {a.start}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(rel, dest, want):
    out = dest / rel
    if out.exists() and out.stat().st_size == want["size"] and sha256(out) == want["sha256"]:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    r = call("GET", "/" + rel, raw=True, timeout=1800)
    h = hashlib.sha256()
    with r, open(tmp, "wb") as f:
        for b in iter(lambda: r.read(1 << 20), b""):
            f.write(b); h.update(b)
    if h.hexdigest() != want["sha256"]:
        tmp.unlink()
        raise IOError(f"{rel}: hash mismatch (changed while syncing?)")
    os.replace(tmp, out)
    return True


def sync(a):
    dest = Path(a.dest)
    m = call("GET", "/api/manifest" + ("?all=1" if a.all else ""))
    got, same, failed = [], 0, []
    wanted = list(m["shared"].items())
    for rid, run in m["runs"].items():
        if run["run"]["status"] not in ("uploading", "queued", "running"):   # else results half-written: next sync
            wanted += list(run["files"].items())
    for rel, want in wanted:
        try:
            if fetch(rel, dest, want):
                got.append(rel)
            else:
                same += 1
        except IOError as e:                       # changed on the Mac mid-sync: the next sync picks it up
            failed.append(str(e))
    idx = dest / "runs" / "index.json"
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text(json.dumps([r["run"] for r in m["runs"].values()], indent=1))
    with open(dest / "sync_log.jsonl", "a") as f:
        f.write(json.dumps(dict(at=time.strftime("%Y-%m-%dT%H:%M:%S"), runs=len(m["runs"]), fetched=got,
                                unchanged=same, failed=failed, code_version=m["code_version"])) + "\n")
    print(f"sync: {len(m['runs'])} runs, {len(got)} files fetched, {same} unchanged, {len(failed)} failed -> {dest}")
    for e in failed:
        print("  ", e)


def watch(a):
    while True:
        sync(a)
        time.sleep(a.every)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("health").set_defaults(fn=health)
    s = sp.add_parser("submit"); s.add_argument("clip"); s.add_argument("--name"); s.add_argument("--source")
    s.add_argument("--wait", action="store_true"); s.add_argument("--dest", default="ride_mirror")
    s.add_argument("--all", action="store_true"); s.set_defaults(fn=submit)
    sp.add_parser("runs").set_defaults(fn=runs)
    s = sp.add_parser("rerun"); s.add_argument("run"); s.add_argument("--from", dest="start", default="export")
    s.set_defaults(fn=rerun)
    for name, fn in (("sync", sync), ("watch", watch)):
        s = sp.add_parser(name); s.add_argument("--dest", default="ride_mirror"); s.add_argument("--all", action="store_true")
        s.add_argument("--every", type=int, default=30); s.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
