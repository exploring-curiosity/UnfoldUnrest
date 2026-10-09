"""A small client for the team's VSS backend (the VAST Builders Challenge stack), from this Mac.

Credentials come from the gitignored .env at the project root (INGRESS_URL, USERNAME, PASSWORD, copied from the
workshop VM's /config/<team>.config). Nothing here prints them or the token.

  python -m vast.client probe                 # who am I, what is filterable, what is indexed (by camera/location)
  python -m vast.client search "query" [k]    # hybrid search, grouped by parent video
  python -m vast.client fetch <s3 source> out.mp4
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def env():
    vals = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in ("INGRESS_URL", "USERNAME", "PASSWORD") if not vals.get(k)]
    if missing:
        raise SystemExit(f".env is missing {', '.join(missing)}")
    return vals


class VSS:
    def __init__(self):
        e = env()
        self.base = e["INGRESS_URL"].rstrip("/")
        r = self._req("POST", "/api/v1/auth/login", dict(username=e["USERNAME"], password=e["PASSWORD"]), auth=False)
        self.token = r["access_token"]

    def _req(self, method, path, body=None, auth=True, params=None, raw=False, timeout=120):
        url = self.base + path + ("?" + urllib.parse.urlencode(params) if params else "")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if auth:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read() if raw else json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise SystemExit(f"{method} {path}: HTTP {e.code} {e.read()[:300]!r}")

    def get(self, path, **params):
        return self._req("GET", path, params=params or None)

    def post(self, path, body):
        return self._req("POST", path, body)

    def search(self, query, top_k=20, min_similarity=.2, **kw):
        return self.post("/api/v1/search", dict(query=query, top_k=top_k, llm_top_n=0, min_similarity=min_similarity,
                                                include_public=True, **kw))

    def explore(self, limit=200):
        out, off = [], 0
        while True:
            r = self.get("/api/v1/videos/explore", scope="all", limit=48, offset=off)
            items = r.get("videos") or r.get("items") or r.get("results") or []
            out += items
            if len(items) < 48 or len(out) >= limit:
                return out, r
            off += 48

    def fetch(self, source, dest):
        """Download a clip through the backend's range-capable stream proxy (token as a query param, never printed)."""
        url = self.base + "/api/v1/videos/stream?" + urllib.parse.urlencode(dict(source=source, token=self.token))
        with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        return dest


def short(s, n=160):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[:n] + "…"


def probe(v):
    me = v.get("/api/v1/auth/me")
    print("logged in as", me.get("username"))
    sch = v.get("/api/v1/metadata/schema")
    for f in sch.get("schema", []):
        opts = f.get("options")
        print(f"  filter {f['name']} ({f.get('ui_type')}): {opts if opts is not None else '(free text)'}")
    for field in ("camera_id", "location", "capture_type"):
        try:
            vals = v.get("/api/v1/metadata/values", field=field, limit=100)
            print(f"  values {field}: {vals.get('values')}")
        except SystemExit as e:
            print(f"  values {field}: {e}")
    items, raw = v.explore()
    print(f"indexed parent videos (first {len(items)}):")
    if items:
        keys = list(items[0].keys())
        print("  fields:", keys)
        for k in ("camera_id", "location", "capture_type", "scenario"):
            if k in items[0]:
                print(f"  by {k}:", Counter(str(i.get(k)) for i in items).most_common())
    else:
        print("  raw:", short(json.dumps(raw), 400))


def show(r):
    ch = r.get("chunk_results") or []
    print(f"{len(r.get('results') or [])} segment hits, {len(ch)} videos")
    for c in ch:
        print(f"- {c.get('original_video')}  [{c.get('best_match_start_sec')}-{c.get('best_match_end_sec')} s] "
              f"segments {c.get('matched_segment_count')}  best {c.get('best_similarity', c.get('similarity_score'))}")
    for s in (r.get("results") or [])[:8]:
        meta = {k: s.get(k) for k in ("camera_id", "location", "capture_type") if s.get(k)}
        print(f"  * {s.get('similarity_score', 0):.3f} {meta} {short(s.get('source'), 90)}\n      {short(s.get('reasoning_content'))}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    v = VSS()
    if cmd == "probe":
        probe(v)
    elif cmd == "search":
        show(v.search(sys.argv[2], top_k=int(sys.argv[3]) if len(sys.argv) > 3 else 20))
    elif cmd == "fetch":
        print(v.fetch(sys.argv[2], sys.argv[3]))
