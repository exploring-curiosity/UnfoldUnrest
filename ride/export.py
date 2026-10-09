"""Everything the ride viewer needs, in <run>/app/: ride.json, ortho.jpg (the street from above), ride.mp4, the street's
coloured point cloud for the 3D view (cloud.bin: float32 x, y, z plan metres; cloud_rgb.bin: uint8 r, g, b), and what the
rider saw (view.bin: uint8 per frame per VIEW_RES grid cell of the map, 0 not judged, 1 in view, 2 hidden)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import objects as O, streetmap as SM

CLOUD_MAX = 400_000
VIEW_RES = .5            # m: the rider's-view grid


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
    # what the rider saw: per frame, which cells of the street within WINDOW m ahead were in view at TARGET_H
    res = VIEW_RES
    x0, x1, y0, y1 = ext["x0"], ext["x1"], ext["y0"], ext["y1"]
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    gx, gy = np.meshgrid(x0 + (np.arange(nx) + .5) * res, y0 + (np.arange(ny) + .5) * res)
    cells = np.stack([gx.ravel(), gy.ravel()], 1)
    view = np.zeros((len(cam), ny * nx), np.uint8)                    # 0 not judged, 1 in view, 2 hidden
    for k in range(len(cam)):
        near = np.nonzero(np.linalg.norm(cells - cam[k, :2], axis=1) <= O.WINDOW)[0]
        if len(near):
            r = st.sees(k, np.c_[cells[near], np.full(len(near), O.TARGET_H)])
            view[k, near] = np.where(r == 1, 1, np.where(r == 0, 2, 0))
    view.tofile(app / "view.bin")
    # evidence: per judged end of a crossing, the frame at the stopping distance and the one where it came into view
    ev = app / "evidence"; shutil.rmtree(ev, ignore_errors=True); ev.mkdir()
    byid = {o["id"]: o for o in objs}

    def shot(k, poly, seen, ids, note):
        im = fr[k].copy()
        for v in ids:
            for kk, *vb in byid[v]["dets"]:
                if kk == k:
                    cv2.rectangle(im, (int(vb[0]), int(vb[1])), (int(vb[2]), int(vb[3])), (245, 197, 24), 4)
        if poly:                                       # the watch area as a TARGET_H tall box: top, base, edges
            col = (47, 163, 107) if seen >= O.CLEAR else (224, 54, 44)
            top, base = np.int32(poly[:4]), np.int32(poly[4:])
            over = im.copy()
            cv2.fillPoly(over, [base], col)
            im = cv2.addWeighted(over, .35, im, .65, 0)
            cv2.polylines(im, [top, base], True, col, 4)
            for a, b in zip(top, base):
                cv2.line(im, tuple(int(v) for v in a), tuple(int(v) for v in b), col, 3)
        im = cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(im, (0, 0), (640, 30), (20, 20, 20), -1)
        cv2.putText(im, note, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (242, 241, 236), 1, cv2.LINE_AA)
        return im
    for o in objs:
        if o["group"] != "crosswalk":
            continue
        for e in o["ends"]:
            if not e["covered"]:
                continue
            fk = {f["k"]: f for f in e["frames"]}
            ids = [b["id"] for b in e["blockers"][:2]]
            a_ = fk[e["stop_k"]]
            pair = [shot(a_["k"], a_["poly"], a_["seen"], ids, f"{a_['d']:.0f} m out, where stopping must start: "
                         f"{round(100 * a_['seen'])}% of the {e['side']} side in view")]
            if e["clear_k"] is not None and e["clear_k"] != e["stop_k"]:
                b_ = fk[e["clear_k"]]
                pair.append(shot(b_["k"], b_["poly"], b_["seen"], ids, f"{b_['d']:.0f} m out: fully in view"))
            im = np.concatenate(pair, 1)
            name = f"cw{o['id']}_{e['side']}.jpg"
            cv2.imwrite(str(ev / name), cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            e["evidence"] = f"evidence/{name}"
            e["stop_t"] = meta["times"][a_["k"]]
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
    ends = [e for c in cws for e in c["ends"]]
    stats = dict(duration=round(meta["times"][-1], 1),
                 distance=round(float(np.linalg.norm(np.diff(cam[:, :2], axis=0), axis=1).sum()), 1),
                 crosswalks=len(cws), crossings_judged=sum(any(e["covered"] for e in c["ends"]) for c in cws),
                 crossings_exposed=sum(c["exposed"] for c in cws), ends_judged=sum(e["covered"] for e in ends),
                 ends_exposed=sum(e["exposed"] for e in ends),
                 stopped=sum(o.get("parked", False) for o in objs),
                 vehicles=sum(o["group"] == "vehicle" for o in objs),
                 furniture=sum(o["group"] in O.FURNITURE for o in objs),
                 blocking=sum(bool(o.get("blocks")) for o in objs), scale_note="metres from a 1.1 m camera height")
    data = dict(video="ride.mp4", frame_size=meta["size"], fps=meta["fps"], hfov=round(hfov, 1), ortho="ortho.jpg",
                extent=ext, cloud=dict(points="cloud.bin", colours="cloud_rgb.bin", n=int(len(P))), cam_h=SM.CAM_H,
                path=path, objects=objs, overlays=overlays, stats=stats, daylight_m=O.DAYLIGHT,
                view=dict(file="view.bin", res=res, x0=x0, y0=y0, nx=nx, ny=ny, frames=len(cam)),
                rules=dict(react_s=O.REACT, decel=O.DECEL, window_m=O.WINDOW, target_h=O.TARGET_H, clear=O.CLEAR,
                           cam_h=SM.CAM_H))
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
