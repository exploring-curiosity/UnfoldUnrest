"""The assembly guide: one GLB per part and guide.json, in a table frame the viewer can show as is.

frame   up is the normal of the surface the build stands on (a RANSAC plane under the base part; the cameras' own up is
        not gravity on hand-held video); the origin is the base part's footprint centre on that surface; one unit is
        a tenth of the finished build's size (video gives no metric scale)
steps   step 0 places the base part (the larger part of the first contact); each later step adds the part a new
        contact brings in, flying in from where it is not touching: from the contact's centre out through the part's
        centre
clips   each step's moment in the source video: from the end of the last rest state without its contact to the start
        of the first rest state with it
"""
from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np

HUES = [(20, "red"), (55, "orange"), (100, "yellow"), (165, "green"), (230, "teal"), (300, "blue"), (335, "purple"),
        (360, "red")]


def colour_name(rgb):
    """A plain colour word from CIE Lab: neutral (chroma < 8) by lightness, else by hue angle, 'dark'/'light' by L."""
    import cv2
    L, a, b = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    L *= 100 / 255; a -= 128; b -= 128
    C, h = np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360
    if C < 8:
        return "white" if L > 85 else "light grey" if L > 65 else "grey" if L > 35 else "black"
    if L > 80 and 60 <= h <= 110 and C < 25:
        return "cream"
    name = next(n for lim, n in HUES if h <= lim)
    return ("dark " if L < 40 else "light " if L > 75 else "") + name


def shape_name(V):
    """A plain shape word from the part's principal extents: bar, disc, panel or block."""
    X = V - V.mean(0)
    _, sv, Vt = np.linalg.svd(X[np.random.default_rng(0).choice(len(X), min(len(X), 5000), replace=False)],
                              full_matrices=False)
    E = np.sort(np.ptp(X @ Vt.T, 0))[::-1]                     # extents along the principal axes, largest first
    if E[0] > 2.5 * E[1]:
        return "bar"
    if E[2] < .35 * E[1]:
        return "disc" if E[0] < 1.25 * E[1] else "panel"
    return "block"


def frame_of(meshes, base):
    """(R 3x3 world->guide rows, origin, scale) from the base part's support plane."""
    m = meshes[base]
    pl = m["plane"]
    up = pl["n"] if pl is not None else m["cam_up"]
    up = up / np.linalg.norm(up)
    allv = np.concatenate([meshes[j]["mesh"].vertices for j in meshes if meshes[j]])
    # x: the direction across the build that the cameras looked along least (any perpendicular works; this one is stable)
    a = np.eye(3)[np.argmin(np.abs(np.eye(3) @ up))]
    x = np.cross(up, a); x /= np.linalg.norm(x)
    z = np.cross(x, up)
    R = np.stack([x, up, z])
    bv = m["mesh"].vertices
    c = bv.mean(0)
    if pl is not None:
        c = c - (c @ pl["n"] + pl["d"]) * pl["n"]            # footprint centre on the surface
    else:
        c = c - ((bv - c) @ up).min() * up
    size = float(np.linalg.norm(np.ptp(allv @ R.T, 0)))
    return R, c, 10.0 / max(size, 1e-9)


def run(out, title=None):
    import trimesh
    out = Path(out)
    meta = json.loads((out / "meta.json").read_text())
    A = pickle.load(open(out / "assembly.pkl", "rb"))
    meshes = {j: m for j, m in A["meshes"].items() if m is not None}
    steps = A["steps"]
    rests = A["rests"]
    times = meta["times"]
    gdir = out / "guide"; gdir.mkdir(exist_ok=True)
    # build order: contacts by the state that first holds them; a part joins at its first contact, onto the partners
    # it touches most firmly in that state (a weaker contact in the same state is a mask bleeding across a part)
    placed, order = [], []
    for i in sorted({c[0] for c in steps}):
        here = [(e, sh) for st, e, sh in steps if st == i]
        if not placed:
            (a, b), _ = max(here, key=lambda x: x[1])
            base = max((a, b), key=lambda j: len(meshes[j]["mesh"].vertices) if j in meshes else 0)
            placed.append(base); order.append(dict(part=base, onto=[], state=i))
        while True:
            cand = {}
            for (a, b), sh in here:
                if (a in placed) != (b in placed):
                    n, o = (b, a) if a in placed else (a, b)
                    cand.setdefault(n, []).append((sh, o))
            if not cand:
                break
            n = max(cand, key=lambda n: max(cand[n])[0])
            top = max(cand[n])[0]
            onto = [o for sh, o in cand[n] if sh >= top - .15]
            placed.append(n); order.append(dict(part=n, onto=onto, state=i))
    if not order:
        raise SystemExit("no steps found")
    R, origin, s = frame_of(meshes, order[0]["part"])
    to_g = lambda X: (np.asarray(X) - origin) @ R.T * s
    from scipy.spatial import cKDTree
    from . import clean as CL, frames as FR
    _, frames_rgb = FR.load(out)
    base = order[0]["part"]
    scan, tidy, info = {}, {}, {}
    for j, m in sorted(meshes.items()):
        mesh = m["mesh"].copy()
        mesh.vertices = to_g(mesh.vertices)
        mesh.fix_normals()
        scan[j] = mesh
        cols = np.asarray(mesh.visual.vertex_colors)[:, :3][m["seen"]]
        # the body colour: the darker half's median (highlights on glossy plastic read lighter than the part)
        if len(cols):
            lum = cols.astype(float) @ [.299, .587, .114]
            rgb = np.median(cols[lum <= np.median(lum)], 0).astype(int)
        else:
            rgb = np.array([180, 180, 180])
        from . import silfit as SF
        X, C = CL.clean_points(to_g(m["pts"]), np.asarray(m["rgb"], np.float64))
        kind, q, loss = m["prim"]
        tm = SF.to_mesh(kind, q)                                  # measured from the silhouettes, in world units
        vw = m["views"]
        Mk = np.unpackbits(vw["masks"], axis=-1)[..., :518].astype(bool)
        col, seen = SF.texture(tm, vw["poses"], Mk, frames_rgb[vw["frames"]], rgb)
        tm.vertices = to_g(tm.vertices)
        size = float(np.ptp(tm.vertices, 0).max())
        prim = dict(kind=kind, loss=loss, size=size)
        tm.visual.vertex_colors = np.c_[col, np.full(len(col), 255, np.uint8)]
        tidy[j] = tm
        info[j] = dict(rgb=rgb, prim=prim, seen_clean=float(seen.mean()), seen_scan=float(m["seen"].mean()), views=len(m["frames"]))
        print(f"part {j}: {kind} from silhouettes (loss {loss:.3f}), size {np.ptp(tm.vertices, 0).round(2)}")
    # snap each added part onto its partners along its approach (closes the gap the noisy depth leaves)
    for st in order[1:]:
        j, on = st["part"], [o for o in st["onto"] if o in tidy]
        if j not in tidy or not on:
            continue
        c = tidy[j].vertices.mean(0)
        W = np.concatenate([tidy[o].vertices for o in on])
        d, _ = cKDTree(W).query(tidy[j].vertices)
        v = c - tidy[j].vertices[d <= np.quantile(d, .05)].mean(0)
        v = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else np.array([0, 1, 0.])
        mv = CL.snap(tidy[j], [tidy[o] for o in on], v, max_move=.3 * info[j]["prim"]["size"])
        if mv:
            tidy[j].vertices = tidy[j].vertices - v * mv
            print(f"part {j}: snapped {mv:.3f} onto {on}")
    parts = []
    for j in sorted(tidy):
        scan[j].export(gdir / f"part{j}_scan.glb")
        tidy[j].export(gdir / f"part{j}.glb")
        V = tidy[j].vertices
        rgb = info[j]["rgb"]
        parts.append(dict(id=j, glb=f"part{j}.glb", glb_scan=f"part{j}_scan.glb", colour=rgb.tolist(),
                          colour_name=colour_name(rgb), shape=shape_name(V), primitive=info[j]["prim"]["kind"],
                          centre=V.mean(0).round(4).tolist(), size=np.ptp(V, 0).round(3).tolist(),
                          observed=round(info[j]["seen_scan"], 3), views=info[j]["views"]))
    byid = {p["id"]: p for p in parts}
    for p in parts:
        p["name"] = f"{p['colour_name']} {p['shape']}"
    gsteps = []
    for si, st in enumerate(order):
        j = st["part"]
        if j not in byid:
            continue
        c = np.array(byid[j]["centre"])
        on = [o for o in st["onto"] if o in byid]
        if on:
            # the contact's centre: the points of the new part nearest the parts it joins
            V = tidy[j].vertices
            W = np.concatenate([tidy[o].vertices for o in on])
            d, _ = cKDTree(W).query(V)
            contact = V[d <= np.quantile(d, .05)].mean(0)
            v = c - contact
            v = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else np.array([0, 1, 0.])
            text = f"Attach the {byid[j]['name']} to the {' and '.join(byid[o]['name'] for o in on)}."
        else:
            v = np.array([0, 1, 0.]); contact = c
            text = f"Start with the {byid[j]['name']}."
        # the clip: from the end of the rest state that first held the previous step to the start of this step's
        i = st["state"]
        if si == 0:
            t0, t1 = 0.0, min(times[rests[0]["frames"][-1]] + 2.0, times[-1])
        else:
            prev = order[si - 1]["state"] if si > 1 else 0
            t0, t1 = times[rests[prev]["frames"][-1]], times[rests[i]["frames"][0]]
        gsteps.append(dict(index=len(gsteps), part=j, onto=on, approach=np.round(v, 4).tolist(),
                           contact=np.round(contact, 4).tolist(), text=text, clip=[round(t0, 2), round(t1, 2)]))
    video = Path(meta["video"])
    shutil.copy(video, gdir / video.name)
    guide = dict(title=title or video.stem.replace("_", " "), video=video.name, units="1 = a tenth of the build",
                 parts=parts, steps=gsteps,
                 note="Every part is measured from the phone video alone (no part library). Clean: the box or cylinder that fits its points, coloured where the camera saw it. Scan: the raw fused surface; grey was never seen.")
    (gdir / "guide.json").write_text(json.dumps(guide, indent=1))
    print(json.dumps(gsteps, indent=1))
    return guide


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
