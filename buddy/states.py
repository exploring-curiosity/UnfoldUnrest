"""Rest states, parts and contacts.

rest state  a run of frames in which the hands cover little of the view (the builder has let go): the scene is still,
            and the moving phone sees it from several sides
parts       the objects of the first rest state, laid out apart before the build (the filming routine): each cut there
            is one part; its look (a Lab colour histogram) is what later cuts are matched to
labels      in every later rest frame each object cut takes the part it looks most like (one cut a part a frame: the
            best match wins)
contacts    two parts touch in a frame when their masks meet (one grown by a few px) and their depths agree where they
            meet; a contact holds in a state when it is seen in at least half the frames that show both parts
steps       each contact that appears between two rest states is an attachment, in the order of the states
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import SIDE

HAND_MAX = .10       # a rest frame: the hands cover under 10% of the view
MIN_RUN = 3          # frames (0.3 s at 10 fps)
CUT_CONF = .25
INSIDE = .8          # a cut lying >= 80% inside a larger kept cut is a piece of it (a stud, a logo)
HAND_OVERLAP = .1
MIN_AREA = 150
TOUCH_PX = 4
DEPTH_AGREE = .06    # relative depth difference where two masks meet
HIST_BINS = (6, 8, 8)


@dataclass
class Rest:
    frames: list
    objects: dict = field(default_factory=dict)       # frame -> list of (mask, conf)
    labels: dict = field(default_factory=dict)        # frame -> {part: mask}
    dist: dict = field(default_factory=dict)          # frame -> {part: distance of its label}
    contacts: dict = field(default_factory=dict)      # (a, b) -> share of frames


def rest_runs(hand_share, cut_lists):
    ok = hand_share < HAND_MAX
    runs, cur = [], []
    for k, v in enumerate(ok):
        if v:
            cur.append(k)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    runs = [r for r in runs if len(r) >= MIN_RUN]
    return runs


def cloth_lab(frame, masks):
    """The surface's colour: the median Lab of the pixels no cut covers."""
    free = ~masks.any(0) if len(masks) else np.ones(frame.shape[:2], bool)
    return np.median(cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[free].reshape(-1, 3), 0)


def object_cuts(frame, masks, conf, hand):
    """Cuts that are things: confident, not the hand, not the cloth's own colour. Nested cuts are all kept (a part may
    show only as the piece of a larger cut, e.g. the bar inside a flag+bar cut); label() picks the granularity."""
    hd = cv2.dilate(hand.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    keep = [i for i in np.argsort([-m.sum() for m in masks])
            if conf[i] >= CUT_CONF and masks[i].sum() >= MIN_AREA and (masks[i] & hd).sum() < HAND_OVERLAP * masks[i].sum()]
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg = cloth_lab(frame, masks)
    out = []
    for i in keep:
        m = masks[i]
        if np.linalg.norm(np.median(lab[m], 0) - bg) < 12:     # the cloth's own fold or shadow
            continue
        out.append((m, float(conf[i])))
    return out


def elong(m):
    """sqrt of the ratio of the mask's principal second moments: ~1 round, large for a bar."""
    mu = cv2.moments(m.astype(np.uint8), True)
    if mu["m00"] < 1:
        return 1.0
    ev = np.sort(np.linalg.eigvalsh(np.array([[mu["mu20"], mu["mu11"]], [mu["mu11"], mu["mu02"]]]) / mu["m00"]))
    return float(np.sqrt(ev[1] / max(ev[0], 1e-6)))


def hist(frame, m):
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    h = cv2.calcHist([lab], [0, 1, 2], m.astype(np.uint8), list(HIST_BINS), [0, 256, 0, 256, 0, 256]).ravel()
    return h / max(h.sum(), 1)


def chi2(a, b):
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9)))


def inventory(rest, frames):
    """The first rest state's objects, clustered over its frames by mask overlap (the camera barely moves in a run),
    nested cuts dropped (the parts lie apart, so the outermost cut is the part): -> part dicts (hist, elong, area)."""
    parts = []
    for k in rest.frames:
        outer = []
        for m, c in rest.objects[k]:                  # largest first
            if not any((m & o).sum() >= INSIDE * m.sum() for o in outer):
                outer.append(m)
        for m in outer:
            best, bi = 0, None
            for j, p in enumerate(parts):
                iou = (m & p["last"]).sum() / max((m | p["last"]).sum(), 1)
                if iou > best:
                    best, bi = iou, j
            if best > .3:
                p = parts[bi]
            else:
                p = dict(hists=[], last=m, n=0, area=[], el=[]); parts.append(p)
            p["hists"].append(hist(frames[k], m)); p["last"] = m; p["n"] += 1; p["area"].append(m.sum()); p["el"].append(elong(m))
    parts = [p for p in parts if p["n"] >= max(2, len(rest.frames) // 2)]
    for p in parts:
        p["hist"] = np.mean(p["hists"], 0)
        p["area"] = float(np.median(p["area"])); p["elong"] = float(np.median(p["el"]))
    return parts


def distance(frame, m, p):
    """How unlike part p a cut looks: colour (chi-square of Lab histograms) plus shape (log elongation ratio, halved:
    a bar seen end-on looks round)."""
    return chi2(hist(frame, m), p["hist"]) + .5 * abs(np.log(elong(m) / p["elong"]))


def label(rest, frames, parts, max_d=.8, keep_min=.3):
    """Each part -> its pixels in each frame. Each part takes its best-matching cut (distance() <= max_d; a cut that is
    the best for two parts goes to the closer one, the other takes its next best). Then pixels are handed out smallest
    cut first, so a part that shows as its own small cut keeps it and a larger cut that also covers it (flag+bar) gives
    the other part only what is left; a part left with under keep_min of its cut is dropped for that frame."""
    for k in rest.frames:
        objs = [m for m, _ in rest.objects[k]]
        D = np.array([[distance(frames[k], m, p) for p in parts] for m in objs]).reshape(len(objs), len(parts))
        pick = {}
        for flat in np.argsort(D, axis=None):
            i, j = divmod(int(flat), len(parts))
            if D[i, j] > max_d or j in pick or i in pick.values():
                continue
            pick[j] = i
        lab, dist, taken = {}, {}, np.zeros((SIDE, SIDE), bool)
        for j, i in sorted(pick.items(), key=lambda ji: objs[ji[1]].sum()):
            m = objs[i] & ~taken
            if m.sum() < keep_min * objs[i].sum():
                continue
            lab[j] = m; dist[j] = float(D[i, j]); taken |= m
        rest.labels[k] = lab
        rest.dist[k] = dist


def touching(ma, mb, depth):
    """Do two masks meet with agreeing depth? -> (meets, n boundary pixels)."""
    k = np.ones((2 * TOUCH_PX + 1,) * 2, np.uint8)
    ga = cv2.dilate(ma.astype(np.uint8), k) > 0
    gb = cv2.dilate(mb.astype(np.uint8), k) > 0
    band_a = ga & mb                                  # b's pixels next to a
    band_b = gb & ma
    if band_a.sum() < 3 or band_b.sum() < 3:
        return False, 0
    da, db = np.median(depth[band_b]), np.median(depth[band_a])
    return abs(da - db) / max(min(da, db), 1e-6) < DEPTH_AGREE, int(band_a.sum())


def contacts(rest, depth):
    seen, hit = {}, {}
    for k in rest.frames:
        lab = rest.labels[k]
        ps = sorted(lab)
        for x in range(len(ps)):
            for y in range(x + 1, len(ps)):
                a, b = ps[x], ps[y]
                seen[(a, b)] = seen.get((a, b), 0) + 1
                t, _ = touching(lab[a], lab[b], depth[k])
                hit[(a, b)] = hit.get((a, b), 0) + int(t)
    rest.contacts = {e: hit[e] / seen[e] for e in seen if seen[e] >= 2}


def steps(rests, min_share=.5):
    """The finished build's contacts (held in >= min_share of the last rest state's frames), each dated by the first
    rest state that already holds it -> [(state, (a, b), share in the finished state)] in build order. A contact seen earlier but gone at the end is
    flicker or a step undone, and is not a step."""
    final = {e: s for e, s in rests[-1].contacts.items() if s >= min_share}
    out = []
    for e, s in final.items():
        first = next(i for i, r in enumerate(rests) if r.contacts.get(e, 0) >= min_share)
        out.append((first, e, s))
    return sorted(out)
