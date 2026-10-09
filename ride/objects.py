"""Street objects on the map, standing vehicles, street furniture, crosswalks and the daylighting check.

placing     a detection stands on the road: the ray through the middle of its box's bottom edge meets the ground
            plane (flat-ground localisation); a vehicle's centre is half a car width further along that ray.
            Furniture also gets a width (both bottom corners cast onto the road) and a height (the box's top edge at
            that distance). Detections farther than MAX_RANGE m (per kind) are dropped
filters     the rider's own bike (a box on the bottom edge of the frame), road-sized boxes, crosswalks cut off by
            the frame's bottom edge
clusters    per kind, placements across frames are grouped greedily within EPS m; a cluster seen in fewer than
            MIN_SEEN frames is noise
standing    a vehicle cluster whose placements stay within PARKED_SPREAD m while the bike rides at least
            PARKED_PASS m past it: it did not move while we did (parked, or waiting at a light: one pass cannot tell)
crosswalk   each crosswalk detection's box is cast onto the ground as a quadrilateral; a cluster's crosswalk is the
            minimum-area rectangle around the middle 80% of its corners
the rider's view  daylighting from the saddle: riding up to a crossing, the rider has to see whoever could come into
            it from either side (a person, a bike, a car) early enough to stop. A parked or stopped vehicle or tall
            street furniture just before the crossing hides them; California AB 413 and NACTO keep the 20 ft (6.1 m)
            before a crosswalk clear for this. The video is the rider's view, so what hides what is read from it:
              watch points    each end of the crossing (left and right of the ride), STEP_IN m in from its near
                              edge and OUT m past its end, TARGET_H m above the road (a child, a person on a bike, a
                              car's bonnet)
              seen            per frame, the point is cast into the camera and compared with the rebuilt depth
                              there: seen when the depth reaches it, hidden when something nearer covers it
                              (Street.sees). The rebuilt depth matches the road's true distance to ~13 m and falls
                              short beyond; the open road ahead reads as seen 94-96% of the time at 10 m, so only
                              the last WINDOW m before the crossing are judged, and only at stopping distances
                              inside that
              seen from       how far before the crossing the end comes into view and stays in view, against the
                              stopping distance at the rider's speed there (reaction REACT s, braking DECEL m/s^2).
                              Exposed when it comes into view later than that
              hidden by       the detected object whose box covers the point in the frames where it is hidden
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

MAX_RANGE = {"crosswalk": 25.0, "vehicle": 18.0}      # m; others 12. The placement error grows with the square of
MAX_RANGE_OTHER = 12.0                                # the range (a pixel below a 20 m foot spans ~1 m of road)
EPS = {"vehicle": 3.0, "person": 1.2, "bicycle": 1.5, "bollard": .8, "fire hydrant": .8, "crosswalk": 5.0,
       "planter": 2.0, "hedge": 2.0, "bus shelter": 2.5, "kiosk": 2.0, "utility box": 1.2, "bench": 1.2,
       "trash can": 1.0, "tree": 1.2, "pole": .8}
MIN_SEEN = {"vehicle": 3, "person": 3, "bicycle": 3, "bollard": 3, "fire hydrant": 2, "crosswalk": 2, "planter": 2,
            "hedge": 2, "bus shelter": 2, "kiosk": 2, "utility box": 2, "bench": 2, "trash can": 2, "tree": 2, "pole": 3}
VEHICLES = {"car", "van", "truck", "bus"}
CAR_L, CAR_W = 4.6, 1.9
PARKED_SPREAD, PARKED_PASS = 1.6, 4.0
DAYLIGHT = 6.1          # m (20 ft): the stretch before a crossing that daylighting keeps clear
WINDOW = 10.5           # m before the crossing that are judged: the road ahead reads as seen 94-96% of the time at
                        # 10 m, ~70% at 13 m (depth falls short there), so a reading beyond ~10 m is not trusted
TARGET_H = 1.0          # m: the height the rider must see at the crossing's ends
AREA_IN, AREA_OUT = .5, 2.5     # m: each end's watch area, from just inside the crossing's end to beyond it
AREA_BEFORE, AREA_INTO = 1.0, 3.0   # m: and from just before the crossing's near edge into it
CLEAR = .9              # an end is in view when the rider sees this share of its area
TURN_MAX = 35.0         # degrees: turning harder than this on the way in, the camera is not looking where the rider is
ON_PATH = 1.2           # m: furniture placed this close to where we rode is misplaced
REACT, DECEL = 1.5, 3.0  # s, m/s^2: stopping distance = v * REACT + v^2 / (2 DECEL)
FURNITURE = {"planter", "hedge", "bus shelter", "kiosk", "utility box"}          # can hide a person
MARKERS = {"person", "bicycle", "bollard", "fire hydrant", "bench", "trash can", "tree", "pole"}   # shown, thin or low


def group(kind, d):
    if d["kind"] in VEHICLES:
        return "vehicle"
    return d["kind"]


def keep(d, W, H):
    x0, y0, x1, y1 = d["box"]
    if d["kind"] in ("traffic light", "raised crosswalk", "bike symbol", "arrow", "road text", "lane line"):
        return False
    if d["kind"] == "crosswalk" and (y1 > .97 * H or d["score"] < .2):
        return False                                                    # cut off by the frame: its extent is unknown
    if y1 > .93 * H and abs((x0 + x1) / 2 - W / 2) < .25 * W:          # the rider's own bike
        return False
    if (x1 - x0) * (y1 - y0) > (.35 if d["kind"] == "crosswalk" else .25) * W * H:
        return False
    return True                                    # a foot above the horizon never meets the road: place() drops it


def place(st, dets, times):
    """-> list of placements dict(k, t, kind, group, score, box, xy (plan m), range)."""
    cam = st.cam_plan()
    out = []
    for ks, ds in dets.items():
        k = int(ks)
        for d in ds:
            if not keep(d, st.W, st.H):
                continue
            x0, y0, x1, y1 = d["box"]
            X = st.ray_ground(k, [(x0 + x1) / 2], [y1])[0]
            if not np.isfinite(X).all():
                continue
            p = st.plan(X)[0]
            ray = p[:2] - cam[k, :2]
            rng = float(np.linalg.norm(ray))
            g = group(d["kind"], d)
            if rng > MAX_RANGE.get(g, MAX_RANGE_OTHER) or rng < 1.0:
                continue
            if g == "vehicle":
                p[:2] += ray / rng * CAR_W / 2
            item = dict(k=k, t=times[k], kind=d["kind"], group=g, score=d["score"], box=d["box"],
                        xy=p[:2].round(3).tolist(), range=round(rng, 2))
            if g in FURNITURE:
                lr = st.ray_ground(k, [x0, x1], [y1, y1])
                if not np.isfinite(lr).all() or y0 < 2:          # cut off at the top: its height is unknown
                    continue
                item["seg"] = st.plan(lr)[:, :2].round(3).tolist()
                item["height"] = round(float(st.height_at(k, (x0 + x1) / 2, y0, rng)[0]), 2)
            if g == "crosswalk":
                cs = st.ray_ground(k, [x0, x1, x1, x0], [y1, y1, y0, y0])
                q = []
                for c in cs:
                    if np.isfinite(c).all():
                        pc = st.plan(c)[0][:2]
                        v = pc - cam[k, :2]
                        r = np.linalg.norm(v)
                        q.append((cam[k, :2] + v / r * min(r, MAX_RANGE[g])).tolist())
                if len(q) < 4:
                    continue
                item["quad"] = np.round(q, 3).tolist()
            out.append(item)
    return out


def cluster(items, eps, min_seen):
    """Greedy single pass in time order: a placement joins the nearest cluster whose running median is within eps."""
    cl = []
    for it in sorted(items, key=lambda i: i["k"]):
        p = np.array(it["xy"])
        best, bd = None, eps
        for c in cl:
            d = np.linalg.norm(np.median(c["pts"][-12:], 0) - p)
            if d < bd:
                best, bd = c, d
        if best is None:
            cl.append(dict(pts=[p], items=[it]))
        else:
            best["pts"].append(p); best["items"].append(it)
    return [c for c in cl if len({i["k"] for i in c["items"]}) >= min_seen]


def rect(points):
    r = cv2.minAreaRect(np.asarray(points, np.float32))
    return r, cv2.boxPoints(r)


def seg_hits_poly(a, b, poly):
    """Does segment a-b cross polygon poly (convex, 4 corners)?"""
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
    for i in range(4):
        c, d = poly[i], poly[(i + 1) % 4]
        if ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d):
            return True
    return cv2.pointPolygonTest(np.asarray(poly, np.float32), (float(a[0]), float(a[1])), False) >= 0


def poly_dist(P, Q):
    """Smallest distance between two convex polygons (0 if they overlap)."""
    P, Q = np.asarray(P, np.float32), np.asarray(Q, np.float32)
    if any(cv2.pointPolygonTest(Q, (float(p[0]), float(p[1])), False) >= 0 for p in P) or \
       any(cv2.pointPolygonTest(P, (float(q[0]), float(q[1])), False) >= 0 for q in Q):
        return 0.0
    return float(min(min(abs(cv2.pointPolygonTest(Q, (float(p[0]), float(p[1])), True)) for p in P),
                     min(abs(cv2.pointPolygonTest(P, (float(q[0]), float(q[1])), True)) for q in Q)))


def street_axis(xy, cam):
    """Unit direction of the street at plan point xy: the ride's direction where it passed closest."""
    j = int(np.argmin(np.linalg.norm(cam[:, :2] - np.asarray(xy), axis=1)))
    ax = cam[min(j + 3, len(cam) - 1), :2] - cam[max(j - 3, 0), :2]
    return ax / max(np.linalg.norm(ax), 1e-9)


def footprint(xy, cam):
    """A car along the street, which runs with the ride where we passed it."""
    ax = street_axis(xy, cam)
    return cv2.boxPoints(((xy[0], xy[1]), (CAR_L, CAR_W), float(np.degrees(np.arctan2(ax[1], ax[0]))))).round(3).tolist()


def merge_vehicles(objs, cam, along=3.0, across=1.4):
    """One car is often two clusters (its placement jumps when planters hide its wheels): vehicles closer than `along` m
    along the street and `across` m across it are one; the more-seen cluster keeps its verdict, positions average by
    sightings. Cars queued in a lane stand ~5.5 m apart, cars side by side ~3 m, so neither merges."""
    vs = sorted([o for o in objs if o["group"] == "vehicle"], key=lambda o: -o["seen"])
    rest = [o for o in objs if o["group"] != "vehicle"]
    kept = []
    for v in vs:
        for k in kept:
            ax = street_axis(k["xy"], cam)
            d = np.asarray(v["xy"]) - np.asarray(k["xy"])
            if abs(d @ ax) < along and abs(d @ np.array([-ax[1], ax[0]])) < across:
                w = k["seen"] / (k["seen"] + v["seen"])
                k.update(xy=(w * np.asarray(k["xy"]) + (1 - w) * np.asarray(v["xy"])).round(2).tolist(),
                         seen=k["seen"] + v["seen"], t0=min(k["t0"], v["t0"]), t1=max(k["t1"], v["t1"]),
                         k0=min(k["k0"], v["k0"]), k1=max(k["k1"], v["k1"]), dets=k["dets"] + v["dets"],
                         kinds=sorted(set(k["kinds"]) | set(v["kinds"])))
                k["footprint"] = footprint(k["xy"], cam)
                break
        else:
            kept.append(v)
    return rest + kept


def merge_crosswalks(objs, gap=2.0):
    """One crossing is often seen as two clusters (its near and far halves): crosswalks whose rectangles come within
    `gap` m are one, re-boxed around both."""
    cws = [o for o in objs if o["group"] == "crosswalk"]
    rest = [o for o in objs if o["group"] != "crosswalk"]
    changed = True
    while changed:
        changed = False
        for i in range(len(cws)):
            for j in range(i + 1, len(cws)):
                if poly_dist(cws[i]["rect"], cws[j]["rect"]) <= gap:
                    a, b = cws[i], cws[j]
                    r, box = rect(np.concatenate([a["rect"], b["rect"]]))
                    a.update(rect=box.round(3).tolist(), size=[round(r[1][0], 1), round(r[1][1], 1)],
                             xy=[round(float(r[0][0]), 2), round(float(r[0][1]), 2)], seen=a["seen"] + b["seen"],
                             t0=min(a["t0"], b["t0"]), t1=max(a["t1"], b["t1"]), k0=min(a["k0"], b["k0"]),
                             k1=max(a["k1"], b["k1"]), dets=a["dets"] + b["dets"])
                    cws.pop(j); changed = True
                    break
            if changed:
                break
    out = rest + cws
    for i, o in enumerate(out):
        o["id"] = i
    return out


def furniture_rect(segs, cam, depth=.8, max_len=6.0):
    """Furniture lines the street: a box along the street (as cars are), as long as the median width it showed across
    the frames (at most max_len m), depth m deep, centred on the median of its bottom edges' midpoints."""
    S = np.asarray(segs, np.float64)                                    # (n, 2, 2) bottom-edge segments on the road
    c = np.median(S.mean(1), 0)
    ax = street_axis(c, cam)
    L = float(np.clip(np.median(np.abs((S[:, 1] - S[:, 0]) @ ax)), .6, max_len))
    return cv2.boxPoints(((c[0], c[1]), (L, depth), float(np.degrees(np.arctan2(ax[1], ax[0]))))).round(3).tolist()


def stopping(v):
    return v * REACT + v * v / (2 * DECEL)


def along_path(cam, j, step=.25):
    """The ride's plan path resampled every `step` m: (points (n, 2), arc length (n,)), and the arc length at frame j."""
    P = cam[:, :2]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    ss = np.arange(0, s[-1], step)
    return np.stack([np.interp(ss, s, P[:, 0]), np.interp(ss, s, P[:, 1])], 1), ss, s[j]


def rider_view(cw, st, cam, times, objs):
    """What the rider saw of the crossing's two ends on the way in (see the module docstring)."""
    R = np.asarray(cw["rect"], np.float64); c = R.mean(0)
    v = street_axis(c, cam)
    u = np.array([-v[1], v[0]])                                           # across the road, to the ride's left
    path, arc, _ = along_path(cam, 0)
    j = int(np.argmin(np.linalg.norm(path - c, axis=1)))
    s_edge = float(arc[j] - np.abs((R - c) @ v).max())                    # the crossing's near edge along the ride
    across = (R - path[j]) @ u                                             # its corners, measured from the ride line
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cam[:, :2], axis=0), axis=1))])
    d = s_edge - s                                                         # each frame's distance to the crossing
    ks = [k for k in range(len(cam)) if 0 < d[k] <= WINDOW + .5]
    if len(ks) >= 2:
        speed = (s[ks[-1]] - s[ks[0]]) / max(times[ks[-1]] - times[ks[0]], 1e-6)
    else:
        jj = int(np.argmin(np.abs(d)))
        i0, i1 = max(jj - 3, 0), min(jj + 3, len(cam) - 1)
        speed = (s[i1] - s[i0]) / max(times[i1] - times[i0], 1e-6)
    speed = max(float(speed), 3.0)
    need = stopping(speed)
    hd = [np.arctan2(*(cam[min(k + 2, len(cam) - 1), 1::-1] - cam[max(k - 2, 0), 1::-1])) for k in ks] if ks else [0.0]
    turn = float(np.degrees(np.ptp(np.unwrap(hd))))                       # how much the bike turned on the way in
    boxes = {}
    for o in objs:
        if o["group"] in ("vehicle", "tree", "pole", "person", "bicycle") or o["group"] in FURNITURE:
            for k, *b in o["dets"]:
                boxes.setdefault(k, []).append((o, b))
    out = []
    for side, end in (("right", min(across.min(), -1.0)), ("left", max(across.max(), 1.0))):
        sg = np.sign(end)
        offs = np.abs(end) + np.arange(-AREA_IN, AREA_OUT + .01, .5)       # across: just inside the end to beyond it
        alongs = np.arange(-AREA_BEFORE, AREA_INTO + .01, .5)             # along: just before the crossing, into it
        A = []
        for al in alongs:
            q = np.array([np.interp(s_edge + al, arc, path[:, 0]), np.interp(s_edge + al, arc, path[:, 1])])
            A += [[*(q + sg * u * o_), TARGET_H] for o_ in offs]
        A = np.array(A)
        corners = A[[0, len(offs) - 1, len(A) - 1, len(A) - len(offs)]]
        rows = []
        for k in ks:
            r = st.sees(k, A)
            n_in = int((r >= 0).sum())
            xy, zc = st.project(k, np.r_[corners, corners * [1, 1, 0]])   # the area's top (TARGET_H) and its base
            ok = np.isfinite(xy).all() and (zc > 0).all()
            rows.append(dict(k=k, d=float(d[k]), r=r, frac=(float((r == 1).sum()) / n_in) if n_in else None,
                             counted=n_in >= .6 * len(A),
                             poly=[[round(float(x), 1), round(float(y), 1)] for x, y in xy] if ok else None))
        cnt = [x for x in rows if x["counted"]]
        fr_ = [x["frac"] for x in cnt]
        sm = [float(np.median(fr_[max(i - 1, 0):i + 2])) for i in range(len(fr_))]   # one noisy frame is not a verdict
        for x, f in zip(cnt, sm):
            x["smooth"] = f
        clear_from = None
        for x in sorted(cnt, key=lambda x: x["d"]):                       # from the crossing outward
            if x["smooth"] < CLEAR:
                break
            clear_from = x["d"]
        at_stop = min(cnt, key=lambda x: abs(x["d"] - need)) if cnt else None
        if need > WINDOW:
            why = f"at {speed * 3.6:.0f} km/h stopping takes {need:.0f} m, farther out than the rebuild can judge ({WINDOW:.0f} m)"
        elif at_stop is None or abs(at_stop["d"] - need) > 1.5:
            why = "the ride starts too close to this crossing" if d[0] < need + 1.5 \
                else "the camera did not face this side of the crossing from the stopping distance"
        elif turn > TURN_MAX:
            why = f"the bike was turning ({turn:.0f} degrees) on the way in"
        else:
            why = None
        covered = why is None
        counts, unnamed = {}, 0
        for x in cnt:
            if not ((clear_from or 0) < x["d"] <= need + 2):
                continue
            xy, _ = st.project(x["k"], A[x["r"] == 0])
            for px, py in xy:
                hits = [(np.linalg.norm(np.asarray(o["xy"]) - cam[x["k"], :2]), o["id"]) for o, b in boxes.get(x["k"], [])
                        if b[0] <= px <= b[2] and b[1] <= py <= b[3]]
                if hits:
                    oid = min(hits)[1]
                    counts[oid] = counts.get(oid, 0) + 1
                else:
                    unnamed += 1
        exposed = bool(covered and at_stop["smooth"] < CLEAR)
        out.append(dict(side=side, area=corners[:, :2].round(2).tolist(),
                        clear_from_m=None if clear_from is None else round(clear_from, 1),
                        needed_m=round(need, 1), speed_kmh=round(speed * 3.6, 1), covered=covered, why=why, exposed=exposed,
                        seen_at_stop=round(at_stop["smooth"], 2) if at_stop else None,
                        stop_k=at_stop["k"] if at_stop else None, stop_d=round(at_stop["d"], 1) if at_stop else None,
                        clear_k=next((x["k"] for x in cnt if x["d"] == clear_from), None),
                        blockers=[dict(id=i, n=n) for i, n in sorted(counts.items(), key=lambda kv: -kv[1])],
                        unnamed=unnamed,
                        frames=[dict(k=x["k"], d=round(x["d"], 1), seen=None if x["frac"] is None else round(x.get("smooth", x["frac"]), 2),
                                     counted=x["counted"], poly=x["poly"]) for x in rows]))
    edge = np.array([np.interp(s_edge, arc, path[:, 0]), np.interp(s_edge, arc, path[:, 1])])
    return out, edge.round(2).tolist(), round(s_edge, 2)


def on_path(poly, cam, step=.25):
    """Does the ride pass through polygon poly?"""
    path, _, _ = along_path(cam, 0, step)
    P = np.asarray(poly, np.float32)
    return any(cv2.pointPolygonTest(P, (float(x), float(y)), False) >= 0 for x, y in path)


def label(o):
    if o["group"] == "vehicle":
        kind = "bus" if "bus" in o["kinds"] else "truck" if "truck" in o["kinds"] else "car"
        return ("stopped " if o.get("parked") else "moving ") + kind      # stopped: parked, or waiting in traffic
    return o["group"]


def analyse(st, dets, times, log=print):
    cam = st.cam_plan()
    items = place(st, dets, times)
    objs = []
    for g in ["vehicle", "crosswalk", *sorted(FURNITURE), *sorted(MARKERS)]:
        for c in cluster([i for i in items if i["group"] == g], EPS[g], MIN_SEEN[g]):
            P = np.array(c["pts"])
            ks = sorted({i["k"] for i in c["items"]})
            o = dict(id=len(objs), group=g, kinds=sorted({i["kind"] for i in c["items"]}), seen=len(ks),
                     t0=times[ks[0]], t1=times[ks[-1]], k0=ks[0], k1=ks[-1],
                     xy=np.median(P, 0).round(2).tolist(), spread=round(float(np.linalg.norm(P.std(0))), 2),
                     score=round(float(np.mean([i["score"] for i in c["items"]])), 3),
                     dets=[[i["k"], *i["box"]] for i in c["items"]])
            if g == "vehicle":
                passed = float(np.linalg.norm(cam[ks[-1], :2] - cam[ks[0], :2]))
                o["passed"] = round(passed, 1)
                o["parked"] = bool(o["spread"] <= PARKED_SPREAD and passed >= PARKED_PASS)
                o["footprint"] = footprint(o["xy"], cam)
                o["height"] = 1.45
            if g in FURNITURE:
                o["footprint"] = furniture_rect([i["seg"] for i in c["items"]], cam)
                o["height"] = round(float(np.median([i["height"] for i in c["items"]])), 2)
                o["xy"] = np.mean(o["footprint"], 0).round(2).tolist()
            if g == "crosswalk":
                Q = np.concatenate([np.array(i["quad"]) for i in c["items"]])
                cen = np.median(Q, 0)
                d = np.linalg.norm(Q - cen, axis=1)
                r, box = rect(Q[d <= np.quantile(d, .8)])
                o["rect"] = box.round(3).tolist(); o["size"] = [round(r[1][0], 1), round(r[1][1], 1)]
                o["xy"] = [round(float(r[0][0]), 2), round(float(r[0][1]), 2)]
            if g in FURNITURE and np.linalg.norm(cam[:, :2] - np.mean(o["footprint"], 0), axis=1).min() < ON_PATH:
                continue                                       # we rode through it: misplaced, not furniture
            if g == "vehicle" and on_path(o["footprint"], cam):
                continue                                       # we rode through it: misplaced
            objs.append(o)
    objs = merge_crosswalks(merge_vehicles(objs, cam))
    for o in objs:
        o["label"] = label(o)
    byid = {o["id"]: o for o in objs}
    path, arc, _ = along_path(cam, 0)
    for cw in [o for o in objs if o["group"] == "crosswalk"]:
        cw["ends"], cw["edge"], s_edge = rider_view(cw, st, cam, times, objs)
        for e in cw["ends"]:
            for b in e["blockers"]:
                o = byid[b["id"]]
                o["blocks"] = True
                if "footprint" in o:                       # how far before the crossing it stands
                    s_o = max(float(arc[np.argmin(np.linalg.norm(path - q, axis=1))]) for q in np.asarray(o["footprint"]))
                    b["before_m"] = round(s_edge - s_o, 1)
                    b["in_20ft"] = bool(-1.0 <= s_edge - s_o <= DAYLIGHT)
        cw["exposed"] = any(e["exposed"] for e in cw["ends"])
        for e in cw["ends"]:
            log(f"crosswalk {cw['id']} ({cw['size'][0]}x{cw['size'][1]} m, t {cw['t0']:.1f}-{cw['t1']:.1f}s) {e['side']} end: "
                f"in view from {e['clear_from_m']} m, {e['seen_at_stop']} of it seen at {e['stop_d']} m; "
                f"stopping needs {e['needed_m']} m at {e['speed_kmh']} km/h; {'judged' if e['covered'] else 'not judged: ' + e['why']}; "
                f"hidden by {[(byid[b['id']]['label'], b['n'], b.get('before_m')) for b in e['blockers']]} + {e['unnamed']} unnamed")
    return items, objs


def run(out, log=print):
    from . import streetmap as SM
    out = Path(out)
    st, meta, fr = SM.build(out)
    dets = json.loads((out / "dets.json").read_text())
    items, objs = analyse(st, dets, meta["times"], log=log)
    from collections import Counter
    log("objects: " + str(Counter(o["group"] for o in objs)) +
        f"; stopped vehicles {sum(o.get('parked', False) for o in objs)}; hiding a crossing's end "
        + str(Counter(o["label"] for o in objs if o.get("blocks"))))
    return st, meta, fr, items, objs


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
