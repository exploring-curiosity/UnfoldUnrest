"""Everything the ride viewer needs, in work/<name>/app/: ride.json, ortho.jpg (the street from above), ride.mp4, and the
street's coloured point cloud for the 3D view (cloud.bin: float32 x, y, z plan metres; cloud_rgb.bin: uint8 r, g, b)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import objects as O, streetmap as SM

CLOUD_MAX = 400_000


def run(out, log=print):
    out = Path(out)
    st, meta, fr, items, objs = O.run(out, log=log)
    app = out / "app"; app.mkdir(exist_ok=True)
    img, ext = st.ortho(fr)
    cv2.imwrite(str(app / "ortho.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    P, Cc = st.cloud(fr)
    if len(P) > CLOUD_MAX:
        pick = np.random.default_rng(0).choice(len(P), CLOUD_MAX, replace=False)
        P, Cc = P[pick], Cc[pick]
    P.astype(np.float32).tofile(app / "cloud.bin"); Cc.astype(np.uint8).tofile(app / "cloud_rgb.bin")
    cam = st.cam_plan()
    fwd = st.R[:, :, 2]
    hx, hy = fwd @ st.ex, fwd @ st.ey
    path = [dict(t=meta["times"][k], x=round(float(cam[k, 0]), 2), y=round(float(cam[k, 1]), 2),
                 h=round(float(np.degrees(np.arctan2(hy[k], hx[k]))), 1)) for k in range(len(cam))]
    hfov = float(np.degrees(np.median(st.pose[:, 8])))
    # evidence: per crosswalk, the frame where it shows largest, with it and the parked cars near it boxed
    ev = app / "evidence"; shutil.rmtree(ev, ignore_errors=True); ev.mkdir()
    byid = {o["id"]: o for o in objs}
    for o in objs:
        if o["group"] != "crosswalk":
            continue
        k, *b = max(o["dets"], key=lambda d: (d[3] - d[1]) * (d[4] - d[2]))
        im = fr[k].copy()
        near = {i for a in o["approaches"] for i in a["in_zone"] + [b["id"] for b in a["blockers"]]}
        for v in near:
            for kk, *vb in byid[v]["dets"]:
                if kk == k:
                    cv2.rectangle(im, (int(vb[0]), int(vb[1])), (int(vb[2]), int(vb[3])), (224, 54, 44), 4)
        col = (47, 163, 107) if o["daylit"] else (224, 54, 44)
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 5)
        im = cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(ev / f"cw{o['id']}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        o["evidence"] = f"evidence/cw{o['id']}.jpg"; o["evidence_t"] = meta["times"][k]
    overlays = {}
    for o in objs:
        for k, *box in o["dets"]:
            overlays.setdefault(str(k), []).append([o["id"], *[round(v) for v in box]])
    for o in objs:
        o.pop("dets", None)
        if o["group"] == "crosswalk":
            d = np.linalg.norm(cam[:, :2] - np.array(o["xy"]), axis=1)
            o["t_pass"] = meta["times"][int(np.argmin(d))]
    cws = [o for o in objs if o["group"] == "crosswalk"]
    stats = dict(duration=round(meta["times"][-1], 1),
                 distance=round(float(np.linalg.norm(np.diff(cam[:, :2], axis=0), axis=1).sum()), 1),
                 crosswalks=len(cws), hidden=sum(c["hidden"] for c in cws), zone_blocked=sum(c["zone_blocked"] for c in cws),
                 failing=sum(not c["daylit"] for c in cws),
                 parked=sum(o.get("parked", False) for o in objs),
                 vehicles=sum(o["group"] == "vehicle" for o in objs),
                 furniture=sum(o["group"] in O.FURNITURE for o in objs),
                 occluders=sum(bool(o.get("occluder")) for o in objs), scale_note="metres from a 1.1 m camera height")
    data = dict(video="ride.mp4", frame_size=meta["size"], fps=meta["fps"], hfov=round(hfov, 1), ortho="ortho.jpg",
                extent=ext, cloud=dict(points="cloud.bin", colours="cloud_rgb.bin", n=int(len(P))), cam_h=SM.CAM_H,
                path=path, objects=objs, overlays=overlays, stats=stats, daylight_m=O.DAYLIGHT,
                rules=dict(occlude_h=O.OCCLUDE_H, react_s=O.REACT, decel=O.DECEL, sight_max=O.SIGHT_MAX))
    (app / "ride.json").write_text(json.dumps(data))
    src = Path(meta["video"])
    dst = app / "ride.mp4"
    if not dst.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "scale=1280:-2", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "26", "-an", "-movflags", "+faststart", str(dst)], check=True)
    log(f"app data: {app} ({len(objs)} objects, {len(cws)} crosswalks, {stats})")
    return data


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
