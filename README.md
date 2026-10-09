# Ride: daylighting checks from a bike video

A first-person bike clip goes in. Out come the street rebuilt in 3D, its crosswalks, the parked cars and street furniture
around them, and whether anything hides a person waiting to cross from the traffic coming toward the crossing.

The Mac runs the pipeline (LingBot-Map for 3D, OWLv2 for detection) behind a small service. The VM reaches it through
a Cloudflare tunnel: it sends clips in and pulls every run's results back, so the Mac and the VM each keep a full copy.

```
VM  --- vm_client.py --->  https://<name>.trycloudflare.com  --->  Mac: ride.service (127.0.0.1:8791)
    <-- sync (pull) ----                                                runs/<id>/...
```

The VM always starts the connection; the Mac never reaches the VM.

## On the Mac

Needs Python with `fastapi`, `uvicorn`, `python-multipart`, `av`, `torch` and `transformers`, plus `cloudflared`
(`brew install cloudflared`). LingBot-Map is loaded from the ElideDB checkout set in `ride/__init__.py`.

1. Put a token in `.env` (it is gitignored):

   ```
   SERVICE_TOKEN=<a long random string>
   ```

   For example, generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

2. Start the service:

   ```bash
   python -m ride.service --port 8791
   ```

3. In a second terminal, open the tunnel:

   ```bash
   cloudflared tunnel --no-autoupdate --url http://localhost:8791
   ```

   Wait for `Registered tunnel connection`, then copy the `https://<name>.trycloudflare.com` URL it prints. This is a
   quick tunnel: the URL changes every time cloudflared restarts, so send the VM the new one.

4. Check it from anywhere:

   ```bash
   curl https://<name>.trycloudflare.com/health
   ```

Runs go one at a time (one GPU). If the service restarts, it resumes runs that were queued or running.

## On the VM

`vm_client.py` needs only Python 3.8+ and the standard library. Copy it to the VM, then:

```bash
export RIDE_URL=https://<name>.trycloudflare.com
export RIDE_TOKEN=<SERVICE_TOKEN from the Mac's .env>
python3 vm_client.py health
```

| Command | What it does |
|---|---|
| `python3 vm_client.py submit ride.mp4 --name "west st" --wait` | Upload the clip in 64 MB pieces, run it, wait, then sync |
| `python3 vm_client.py submit ride.mp4 --source '{"vast_video_id": "..."}'` | Same, recording where the clip came from |
| `python3 vm_client.py runs` | List every run with its status and findings |
| `python3 vm_client.py rerun <run id> --from export` | Run again from a stage (`frames`, `geometry`, `detect`, `export`) with the current code |
| `python3 vm_client.py sync` | Pull results into `ride_mirror/` (only files whose hash changed) |
| `python3 vm_client.py sync --all` | Also pull frames, depth and the clip |
| `python3 vm_client.py watch --every 30` | Sync every 30 s |

Each file is checked against its sha256 before it replaces the old copy. Runs still uploading, queued or running are
skipped until they finish. Each sync is logged to `ride_mirror/sync_log.jsonl`.

## Viewing the results

- **In a browser, through the tunnel:** open `https://<name>.trycloudflare.com/login`, paste the token, and you land
  on the list of runs. The sign-in lasts 7 days.
- **On the VM's copy:** `python3 -m http.server -d ride_mirror 8000`, then open `http://localhost:8000/viewer/runs.html`.
- **On the Mac:** `python3 serve.py 8790`, then open `http://localhost:8790/viewer/runs.html`.

Open a run to see it: `viewer/ride.html?run=<run id>` (with no `run`, it shows the newest finished run). The 3D view
loads three.js and fonts from a CDN, so it needs internet.

## API

Everything except `/health` needs `Authorization: Bearer <token>`, or the cookie set by `/login`. Never put the token in
a URL.

| Route | |
|---|---|
| `GET /health` | Is the service up (no token) |
| `POST /api/runs` | `{"name", "filename", "source"}`: a new run waiting for its clip |
| `PUT /api/runs/{id}/clip?offset=N` | One piece of the clip at byte N (at most 95 MB per request; Cloudflare caps 100 MB) |
| `POST /api/runs/{id}/start` | The clip is complete: queue the run |
| `POST /api/runs/{id}/rerun?start=STAGE` | Run again from a stage |
| `GET /api/runs`, `GET /api/runs/{id}` | Run records |
| `GET /api/manifest?all=1` | Every run's files with sizes and sha256 |
| `GET /runs/{id}/{path}`, `GET /viewer/{path}` | Files, laid out as on disk |

## Layout

```
ride/          pipeline: frames -> geometry -> detect -> export; service.py, runs.py
viewer/        runs.html (list), ride.html (one run)
vm_client.py   the VM side
runs/<id>/     one run: clip, run.json, log.txt, app/ (ride.json, video, map, evidence)   [gitignored]
```
