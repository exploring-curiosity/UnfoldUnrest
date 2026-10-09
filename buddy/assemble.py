"""From labelled rest states to the assembly: per-part masks in the finished state, contacts, build order.

Appearance labels (states.label) are good while parts lie apart and look like themselves, but a finished build fuses
look-alike parts into one cut (the flag and its bar, both dark blue). So the finished state is explained by continuity
first: the latest earlier state in which parts were already joined is moved onto the finished state's points (one
rigid ICP: a joined group moves as one), and each of its parts claims the pixels it projects onto, where the observed
depth agrees. Only the pixels nobody claims are new, and go to the parts not yet placed, by appearance.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import SIDE, mapcheck as MC, object3d as O3, states as St

CONF_Q = .3          # a frame's lowest-confidence 30% of depth cells are not lifted
DEPTH_TOL = .05      # a projected point is visible where the observed depth is within 5% of its own
SPLAT = 2            # px radius of a projected point


class Geo:
    """LingBot's cameras and depth with per-frame lifting and projection on the SIDE grid."""

    def __init__(self, pose, depth, conf):
        self.pose, self.depth, self.conf = pose, depth, conf
        self.R, self.C = MC.cameras(pose)
        self.fx, self.fy = MC.focals(pose, depth.shape[-1])

    def lift(self, k, mask, cap=4000, seed=0):
        m = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if m.sum() < 20:
            m = mask
        c = self.conf[k]
        m = m & (c >= np.quantile(c, CONF_Q)) & (self.depth[k] > 0)
        ys, xs = np.nonzero(m)
        if len(ys) > cap:
            i = np.random.default_rng(seed).choice(len(ys), cap, replace=False); ys, xs = ys[i], xs[i]
        z = self.depth[k][ys, xs].astype(np.float64)
        g = SIDE
        X = np.stack([(xs + .5 - g / 2) / self.fx[k] * z, (ys + .5 - g / 2) / self.fy[k] * z, z], 1)
        return X @ self.R[k].T + self.C[k]

    def project(self, k, X):
        """-> (u, v, z) of world points X in frame k (z <= 0 behind)."""
        c = (X - self.C[k]) @ self.R[k]
        z = c[:, 2]
        zs = np.where(z > 1e-6, z, 1.0)
        return c[:, 0] / zs * self.fx[k] + SIDE / 2, c[:, 1] / zs * self.fy[k] + SIDE / 2, z


def part_points(geo, rest, parts=None):
    """{part: world points over the state's frames} from its labels."""
    out = {}
    for k in rest.frames:
        for j, m in rest.labels[k].items():
            if parts is None or j in parts:
                out.setdefault(j, []).append(geo.lift(k, m, cap=1500, seed=k))
    return {j: np.concatenate(v) for j, v in out.items() if sum(map(len, v)) >= 50}


def state_cloud(geo, rest):
    """All object pixels of a state (every kept cut), lifted."""
    P = []
    for k in rest.frames:
        objs = [m for m, _ in rest.objects[k]]
        if objs:
            P.append(geo.lift(k, np.any(objs, 0), cap=3000, seed=k))
    return np.concatenate(P) if P else np.zeros((0, 3))


def claim(geo, k, X, observed):
    """Pixels of frame k that the points X explain: projected (splatted), in front, and agreeing with the observed
    depth; restricted to the observed object pixels."""
    u, v, z = geo.project(k, X)
    ok = (z > 1e-6) & (u >= 0) & (u < SIDE) & (v >= 0) & (v < SIDE)
    u, v, z = u[ok].astype(int), v[ok].astype(int), z[ok]
    d = geo.depth[k][v, u]
    vis = np.abs(z - d) <= DEPTH_TOL * d
    m = np.zeros((SIDE, SIDE), np.uint8)
    m[v[vis], u[vis]] = 1
    m = cv2.dilate(m, np.ones((2 * SPLAT + 1,) * 2, np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    return m & observed


def joined_groups(rest, min_share=.5):
    """Connected groups of parts by the state's contacts."""
    adj = {}
    for (a, b), s in rest.contacts.items():
        if s >= min_share:
            adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    seen, groups = set(), []
    for a in adj:
        if a in seen:
            continue
        g, todo = set(), [a]
        while todo:
            x = todo.pop()
            if x in g:
                continue
            g.add(x); todo += list(adj[x] - g)
        seen |= g; groups.append(g)
    return groups


def table_normal(geo, rest, seed=0):
    """The surface the build stands on: a RANSAC plane through the lifted pixels no object cut covers (the cloth),
    oriented towards the cameras."""
    P = []
    for k in rest.frames:
        objs = [m for m, _ in rest.objects[k]]
        free = ~np.any(objs, 0) if objs else np.ones((SIDE, SIDE), bool)
        P.append(geo.lift(k, free, cap=2000, seed=k))
    P = np.concatenate(P)
    rng = np.random.default_rng(seed)
    scale = float(np.linalg.norm(np.quantile(P, .9, 0) - np.quantile(P, .1, 0)))
    best = (0, None)
    for _ in range(800):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        k = int((np.abs((P - a) @ n) < .01 * scale).sum())
        if k > best[0]:
            best = (k, (n, a))
    n, a = best[1]
    if n @ (geo.C[rest.frames].mean(0) - a) < 0:
        n = -n
    return n, a


def rot_about(n, ang):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(n * ang).as_matrix()


def multi_icp(P, Q, n, tries=12):
    """Rigid ICP from several turns about the table normal (a part moved on the table mostly turns about it); the
    lowest trimmed rms wins. -> (R, t, rms)."""
    best = None
    cp, cq = P.mean(0), Q.mean(0)
    for i in range(tries):
        R0 = rot_about(n, 2 * np.pi * i / tries)
        P0 = (P - cp) @ R0.T + cq
        R, t, rms = O3.icp(P0, Q, iters=40, trim=.7)
        if best is None or rms < best[2]:
            Rt = R @ R0
            best = (Rt, R @ (cq - R0 @ cp) + t, rms)
    return best


def source_state(rests):
    """The earlier state that best shows a joined group: the most frames in which every part of the group is labelled
    (latest on ties). -> (rest, group) or (None, None)."""
    best = (0, None, None)
    for r in rests[1:-1]:
        for g in joined_groups(r):
            if len(g) < 2:
                continue
            n = sum(all(j in r.labels[k] for j in g) for k in r.frames)
            if n >= best[0]:
                best = (n, r, g)
    return best[1], best[2]


def explain_final(geo, rests, frames, parts, sure=.3, log=print):
    """Relabel the finished state: confident appearance labels stay (distance <= sure); the parts of the best earlier
    joined group that are not confidently seen claim their pixels by continuity (the group moved onto the finished
    state by multi-start ICP); what is left goes to the part it looks most like (merged into a part already labelled,
    or as a new part's label)."""
    final = rests[-1]
    src, group = source_state(rests)
    if src is None:
        log("explain: no earlier joined group; appearance labels kept"); return
    pts = part_points(geo, src, group)
    if set(pts) != set(group):
        log(f"explain: group {sorted(group)} lacks points for some parts; appearance labels kept"); return
    n, _ = table_normal(geo, final)
    P = np.concatenate([pts[j] for j in sorted(group)])
    Q = state_cloud(geo, final)
    R, t, rms = multi_icp(P, Q, n)
    log(f"explain: group {sorted(group)} from frames {src.frames[0]}-{src.frames[-1]} moved onto the finished state "
        f"(multi-start ICP rms {rms:.4f}, turn {np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))):.1f} deg)")
    moved = {j: pts[j] @ R.T + t for j in group}
    for k in final.frames:
        objs = [m for m, _ in final.objects[k]]
        if not objs:
            continue
        observed = np.any(objs, 0)
        lab = {j: m for j, m in final.labels[k].items() if final.dist[k].get(j, 9) <= sure}
        taken = np.any(list(lab.values()), 0) if lab else np.zeros((SIDE, SIDE), bool)
        for j in sorted(group, key=lambda j: len(moved[j])):       # the smaller part claims first
            if j in lab:
                continue
            m = claim(geo, k, moved[j], observed) & ~taken
            if m.sum() >= 30:
                lab[j] = m; taken |= m
        left = observed & ~taken
        left = cv2.morphologyEx(left.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        nc, cc, st, _ = cv2.connectedComponentsWithStats(left.astype(np.uint8))
        for c in np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1:
            if st[c, cv2.CC_STAT_AREA] < 150:
                break
            m = cc == c
            j = min(range(len(parts)), key=lambda j: St.distance(frames[k], m, parts[j]))
            lab[j] = lab[j] | m if j in lab else m
        final.labels[k] = lab


def state_moves(geo, rests, max_rel_rms=.02, log=print):
    """Earlier states whose joined group is still joined the same way at the end: the group moved onto the finished
    state's points (multi-start ICP). A state's frames then see those parts as they sit in the finished build, so
    their views add to the parts' 3D. -> {state index: (R, t, group)}; the finished state itself is the identity."""
    final = rests[-1]
    Q = state_cloud(geo, final)
    scale = float(np.linalg.norm(np.quantile(Q, .95, 0) - np.quantile(Q, .05, 0)))
    n, _ = table_normal(geo, final)
    held = {e for e, s in final.contacts.items() if s >= .5}
    out = {len(rests) - 1: (np.eye(3), np.zeros(3), set(range(99)))}
    for i, r in enumerate(rests[1:-1], 1):
        for g in joined_groups(r):
            edges = {e for e, s in r.contacts.items() if s >= .5 and e[0] in g and e[1] in g}
            if len(g) < 2 or not edges <= held:
                continue
            pts = part_points(geo, r, g)
            if set(pts) != g:
                continue
            R, t, rms = multi_icp(np.concatenate([pts[j] for j in sorted(g)]), Q, n)
            ok = rms / scale <= max_rel_rms
            log(f"  state {r.frames[0]}-{r.frames[-1]} group {sorted(g)} -> finished: rms/size {rms / scale:.4f}"
                f" {'kept' if ok else 'dropped'}")
            if ok:
                out[i] = (R, t, g)
    return out


def moved_pose(pose_k, R, t):
    """A frame's pose_enc with its camera moved by the rigid (R, t) (camera-to-world: R Rc, R C + t)."""
    from scipy.spatial.transform import Rotation
    Rc = Rotation.from_quat(pose_k[3:7]).as_matrix()
    out = np.array(pose_k, np.float64).copy()
    out[:3] = R @ pose_k[:3] + t
    out[3:7] = Rotation.from_matrix(R @ Rc).as_quat()
    return out
