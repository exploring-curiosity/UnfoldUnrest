"""A part's shape measured from its silhouettes: the box or capped cylinder whose projection matches its masks.

Depth from one phone frame is too noisy for a small part (fused points form a cloud as thick as the part), but its
outline in each view is sharp and every view's camera is known. So the primitive is fitted by analysis by synthesis:
project it into every view (a primitive is convex, so its silhouette is the convex hull of its projected outline
points) and count the pixels it covers outside what may be the part (its own mask, or another part or a hand that may
hide it) plus the part's pixels it leaves uncovered, over the part's area. The fused points only start the search and
hold it near the right depth (a weak term: their median distance to the surface over the part's size).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from . import mapcheck as MC

G = 259            # silhouettes are compared at half the frames' grid
SLIDE = 5          # px at G: the per-view slack for camera jitter


def cameras(poses, g=518):
    R, C = MC.cameras(np.asarray(poses, np.float64))
    fx, fy = MC.focals(np.asarray(poses, np.float64), g)
    s = G / g
    return R, C, fx * s, fy * s


def down(m):
    return cv2.resize(m.astype(np.uint8), (G, G), interpolation=cv2.INTER_AREA) > 0


def unit(th, ph):
    return np.array([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)])


def basis(a):
    t = np.eye(3)[np.argmin(np.abs(a))]
    u = np.cross(a, t); u /= np.linalg.norm(u)
    return u, np.cross(a, u)


RING = np.linspace(0, 2 * np.pi, 40, endpoint=False)
CORN = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])


def outline(kind, q):
    """Outline points of the primitive with parameters q (world frame)."""
    if kind == "cylinder":
        c, (th, ph), r, h = q[:3], q[3:5], np.exp(q[5]), np.exp(q[6])
        a = unit(th, ph); u, v = basis(a)
        ring = np.cos(RING)[:, None] * u + np.sin(RING)[:, None] * v
        return np.concatenate([c + r * ring + h / 2 * a, c + r * ring - h / 2 * a])
    c, rv, ext = q[:3], q[3:6], np.exp(q[6:9])
    return c + (CORN * ext) @ Rotation.from_rotvec(rv).as_matrix().T


def surface_dist(kind, q, X):
    if kind == "cylinder":
        c, (th, ph), r, h = q[:3], q[3:5], np.exp(q[5]), np.exp(q[6])
        a = unit(th, ph)
        d = X - c
        hh = d @ a
        rho = np.linalg.norm(d - hh[:, None] * a, axis=1)
        orr, oh = np.maximum(rho - r, 0), np.maximum(np.abs(hh) - h / 2, 0)
        inside = (orr == 0) & (oh == 0)
        return np.where(inside, np.minimum(r - rho, h / 2 - np.abs(hh)), np.hypot(orr, oh))
    c, rv, ext = q[:3], q[3:6], np.exp(q[6:9])
    Q = np.abs((X - c) @ Rotation.from_rotvec(rv).as_matrix())
    half = ext / 2
    out = np.maximum(Q - half, 0)
    inside = (out == 0).all(1)
    return np.where(inside, (half - Q).min(1), np.linalg.norm(out, axis=1))


class Views:
    """Up to `cap` views spread over the given ones. The loss is smooth (chamfer-like, Borgefors 1988): a silhouette
    pixel outside what may be the part costs its distance to that region, a part pixel left uncovered costs its
    distance to the silhouette; both over the part's area times its size in pixels, averaged over views."""

    def __init__(self, poses, masks, allowed, cap=16):
        idx = np.unique(np.linspace(0, len(poses) - 1, min(cap, len(poses))).round().astype(int))
        self.R, self.C, self.fx, self.fy = cameras(np.asarray(poses)[idx])
        self.m = [down(masks[i]) for i in idx]
        self.ok = [down(allowed[i]) | mm for i, mm in zip(idx, self.m)]
        self.dout = [cv2.distanceTransform((~o).astype(np.uint8), cv2.DIST_L2, 3) for o in self.ok]
        self.area = np.array([max(m.sum(), 1) for m in self.m], np.float64)
        self.norm = self.area * np.sqrt(self.area)
        self.cm = [np.array(np.nonzero(m)[::-1], np.float64).mean(1) for m in self.m]

    def silhouette(self, k, P):
        c = (P - self.C[k]) @ self.R[k]
        if (c[:, 2] <= 1e-6).any():
            return None
        uv = np.stack([c[:, 0] / c[:, 2] * self.fx[k] + G / 2, c[:, 1] / c[:, 2] * self.fy[k] + G / 2], 1)
        if np.abs(uv).max() > 1e5:
            return None
        # LingBot's cameras jitter by a few pixels between views of a still scene: each view may slide the outline by
        # up to SLIDE px towards the part's mask, so the fit measures shape rather than the jitter
        uv = uv + np.clip(self.cm[k] - uv.mean(0), -SLIDE, SLIDE)
        S = np.zeros((G, G), np.uint8)
        cv2.fillConvexPoly(S, cv2.convexHull(np.round(uv * 4).astype(np.int32)), 1, lineType=cv2.LINE_8, shift=2)
        return S > 0

    def loss(self, P, keep=.7):
        """Mean over the best `keep` share of views (a view whose mask is wrong, or whose camera is off, is outvoted)."""
        per = []
        for k in range(len(self.m)):
            S = self.silhouette(k, P)
            if S is None or not S.any():
                return 1e3
            ds = cv2.distanceTransform((~S).astype(np.uint8), cv2.DIST_L2, 3)
            per.append((self.dout[k][S].sum() + ds[self.m[k]].sum()) / self.norm[k])
        per = np.sort(per)[:max(1, int(round(keep * len(per))))]
        return 100 * float(per.mean())

    def iou(self, P):
        r = []
        for k in range(len(self.m)):
            S = self.silhouette(k, P)
            if S is None:
                return 0.0
            r.append((S & self.m[k]).sum() / max((S | self.m[k]).sum(), 1))
        return float(np.median(r))


def simplex(kind, q0, size):
    """A starting simplex sized to the part: centre +-20% of its size, angles +-.35 rad, log sizes +-.35."""
    st = np.r_[np.full(3, .2 * size), np.full(2 if kind == "cylinder" else 3, .35), np.full(len(q0) - (5 if kind == "cylinder" else 6), .35)]
    S = [q0]
    for i in range(len(q0)):
        e = q0.copy(); e[i] += st[i]; S.append(e)
    return np.array(S)


def fit(views, X, size, inits, lam=.1, shrink=.05, log=print):
    """Best (kind, q, loss) over the initial guesses [(kind, q0)]: Nelder-Mead from a part-sized simplex, then Powell
    on the smooth loss, then Nelder-Mead again from where Powell stopped. A small shrink term (the sum of the sizes
    over the part's size) lets a size no view constrains (a flag's thickness, seen only face-on) fall to what the
    outlines need instead of staying at the noisy depth's spread."""
    best = None
    for kind, q0 in inits:
        nsz = 2 if kind == "cylinder" else 3
        f = lambda q: (views.loss(outline(kind, q)) + lam * float(np.median(np.abs(surface_dist(kind, q, X)))) / size
                       + shrink * float(np.sum(np.exp(q[-nsz:]))) / size)
        r = minimize(f, q0, method="Nelder-Mead", options=dict(initial_simplex=simplex(kind, q0, size), maxfev=1200))
        r = minimize(f, r.x, method="Powell", options=dict(maxfev=1500, xtol=1e-4, ftol=1e-5))
        r = minimize(f, r.x, method="Nelder-Mead", options=dict(initial_simplex=simplex(kind, r.x, size / 3), maxfev=800))
        q = r.x
        log(f"    {kind}: loss {r.fun:.3f} (from {f(q0):.3f}), median IoU {views.iou(outline(kind, q)):.2f}")
        # a cylinder must beat the best box by 5% (a box is the plainer claim; round needs the evidence)
        score = r.fun * (1.0 if kind == "box" else 1.05)
        if best is None or score < best[3]:
            best = (kind, q, float(r.fun), score)
    return best[:3]


def inits_from(X, up):
    """Starting guesses from the points: a cylinder along up and along each principal axis; a box on the principal
    axes. Sizes start at the points' 10-90% spread (the cloud is thicker than the part)."""
    c = np.median(X, 0)
    _, _, Vt = np.linalg.svd(X - c, full_matrices=False)
    out = []
    for a in [up, Vt[0], Vt[2]]:
        a = a / np.linalg.norm(a)
        if a[1] < 0:
            a = -a
        th, ph = np.arccos(np.clip(a[1], -1, 1)), np.arctan2(a[2], a[0])
        h = X @ a; u, v = basis(a)
        rad = np.linalg.norm(np.c_[X @ u, X @ v] - np.median(np.c_[X @ u, X @ v], 0), axis=1)
        r0 = max(np.quantile(rad, .8), 1e-4); h0 = max(np.ptp(np.quantile(h, [.1, .9])), 1e-4)
        out.append(("cylinder", np.r_[c, th, ph, np.log(r0), np.log(h0)]))
    Q = (X - c) @ Vt.T
    ext = np.maximum(np.quantile(Q, .9, 0) - np.quantile(Q, .1, 0), 1e-4)
    Rm = Vt.T if np.linalg.det(Vt.T) > 0 else Vt.T * [1, 1, -1]
    out.append(("box", np.r_[c, Rotation.from_matrix(Rm).as_rotvec(), np.log(ext)]))
    return out


def to_mesh(kind, q, detail=60):
    import trimesh
    if kind == "cylinder":
        c, (th, ph), r, h = q[:3], q[3:5], np.exp(q[5]), np.exp(q[6])
        a = unit(th, ph); u, v = basis(a)
        m = trimesh.creation.cylinder(radius=r, height=h, sections=96)
        T = np.eye(4); T[:3, :3] = np.stack([u, v, a], 1); T[:3, 3] = c
        size = max(2 * r, h)
    else:
        c, rv, ext = q[:3], q[3:6], np.exp(q[6:9])
        m = trimesh.creation.box(extents=ext)
        T = np.eye(4); T[:3, :3] = Rotation.from_rotvec(rv).as_matrix(); T[:3, 3] = c
        size = float(ext.max())
    m.apply_transform(T)
    vv, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=size / detail)
    return trimesh.Trimesh(vv, f, process=True)


def texture(mesh, poses, masks, images, body, g=518):
    """Vertex colours by projection: each vertex takes the median pixel colour over the views in which it faces the
    camera and lands inside the part's own mask (after that view's jitter slide); a vertex no view shows that way takes
    the body colour. -> (colours uint8 (n, 3), seen (n,) bool). The primitive is convex, so facing the camera is the
    whole visibility test against itself; other parts that hide it are outside its mask already."""
    R, C = MC.cameras(np.asarray(poses, np.float64))
    fx, fy = MC.focals(np.asarray(poses, np.float64), g)
    V = np.asarray(mesh.vertices); N = np.asarray(mesh.vertex_normals)
    acc = [[] for _ in range(len(V))]
    cols = np.zeros((len(V), 3)); cnt = np.zeros(len(V))
    samples = []
    for k in range(len(poses)):
        c = (V - C[k]) @ R[k]
        z = c[:, 2]
        if (z <= 1e-6).any():
            continue
        u = c[:, 0] / z * fx[k] + g / 2; v = c[:, 1] / z * fy[k] + g / 2
        m = masks[k]
        ys, xs = np.nonzero(m)
        if not len(xs):
            continue
        sl = np.clip(np.array([xs.mean(), ys.mean()]) - np.array([u.mean(), v.mean()]), -2 * SLIDE, 2 * SLIDE)
        u, v = u + sl[0], v + sl[1]
        ray = V - C[k]; ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        facing = (N * ray).sum(1) < -.15
        iu, iv = np.round(u).astype(int), np.round(v).astype(int)
        inside = facing & (iu >= 0) & (iu < g) & (iv >= 0) & (iv < g)
        i = np.flatnonzero(inside)
        i = i[m[iv[i], iu[i]]]
        samples.append((i, images[k][iv[i], iu[i]].astype(np.float64)))
    if not samples:
        return np.tile(np.asarray(body, np.uint8), (len(V), 1)), np.zeros(len(V), bool)
    I = np.concatenate([s[0] for s in samples]); P = np.concatenate([s[1] for s in samples])
    order = np.argsort(I, kind="stable"); I, P = I[order], P[order]
    starts = np.r_[0, np.flatnonzero(np.diff(I)) + 1]
    out = np.tile(np.asarray(body, np.float64), (len(V), 1)); seen = np.zeros(len(V), bool)
    for a, b in zip(starts, np.r_[starts[1:], len(I)]):
        out[I[a]] = np.median(P[a:b], 0); seen[I[a]] = True
    return np.clip(out, 0, 255).astype(np.uint8), seen
