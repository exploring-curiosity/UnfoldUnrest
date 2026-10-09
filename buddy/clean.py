"""A clean shape per part, measured from its own observed points: the box or capped cylinder that fits them best.

Manufactured parts are mostly made of boxes and cylinders, so a primitive fitted to the noisy fused points gives the
crisp shape the guide needs; its size is measured from this video, no part library is used. Candidates: a cylinder
along each principal axis and along the table's up, a box on the principal axes and an up-aligned box. Each is scored
by the median distance of the points to its surface over the part's size; the lowest wins. A cylinder's circle is
fitted to the convex hull of the points across its axis (a disc seen from above is a filled set of points, a pole seen
from the side a ring), a box's extents are the points' 2-98% quantile ranges. The surface is coloured from the
observed points within reach; elsewhere it takes the part's body colour and counts as inferred.
"""
from __future__ import annotations

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def clean_points(X, C, k=8):
    """Drop isolated points (mean distance to k neighbours past median + 3 robust sigmas)."""
    from scipy.spatial import cKDTree
    if len(X) < 50:
        return X, C
    d, _ = cKDTree(X).query(X, k=k + 1)
    m = d[:, 1:].mean(1)
    med = np.median(m); mad = 1.4826 * np.median(np.abs(m - med))
    keep = m <= med + 3 * max(mad, 1e-9)
    return X[keep], C[keep]


def basis(a):
    a = a / np.linalg.norm(a)
    t = np.eye(3)[np.argmin(np.abs(a))]
    u = np.cross(a, t); u /= np.linalg.norm(u)
    return a, u, np.cross(a, u)


def circle_lsq(P):
    """Kasa least-squares circle through 2D points -> (centre, r)."""
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:2]
    return c, float(np.sqrt(max(x[2] + c @ c, 1e-12)))


def fit_cylinder(X, a):
    from scipy.spatial import ConvexHull
    a, u, v = basis(a)
    Y = np.c_[X @ u, X @ v]
    h = X @ a
    try:
        hull = Y[ConvexHull(Y).vertices]
    except Exception:
        return None
    c, r = circle_lsq(hull)
    lo, hi = np.quantile(h, [.02, .98])
    rho = np.linalg.norm(Y - c, axis=1)
    out_r = np.maximum(rho - r, 0); out_h = np.maximum(np.maximum(lo - h, h - hi), 0)
    inside = (out_r == 0) & (out_h == 0)
    d = np.where(inside, np.minimum(r - rho, np.minimum(h - lo, hi - h)), np.hypot(out_r, out_h))
    centre = c[0] * u + c[1] * v + (lo + hi) / 2 * a
    return dict(kind="cylinder", axis=a, centre=centre, r=r, h=hi - lo, d=d)


def fit_box(X, axes):
    Q = X @ axes.T
    lo, hi = np.quantile(Q, .02, 0), np.quantile(Q, .98, 0)
    half = (hi - lo) / 2; mid = (hi + lo) / 2
    q = np.abs(Q - mid)
    out = np.maximum(q - half, 0)
    inside = (out == 0).all(1)
    d = np.where(inside, (half - q).min(1), np.linalg.norm(out, axis=1))
    return dict(kind="box", axes=axes, centre=mid @ axes, ext=hi - lo, d=d)


def fit(X):
    """The best primitive for points X (guide frame, y up)."""
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    size = float(np.linalg.norm(np.quantile(X, .98, 0) - np.quantile(X, .02, 0)))
    cands = [fit_cylinder(X, a) for a in [*Vt, UP]]
    hz = Vt[np.argmax(np.abs(Vt[:, [0, 2]]).sum(1))].copy(); hz[1] = 0
    if np.linalg.norm(hz) > 1e-6:
        hz /= np.linalg.norm(hz)
        cands.append(fit_box(X, np.stack([hz, UP, np.cross(hz, UP)])))
    cands.append(fit_box(X, Vt))
    cands = [c for c in cands if c is not None]
    for c in cands:
        c["score"] = float(np.median(np.abs(c["d"]))) / size
    best = min(cands, key=lambda c: c["score"])
    best["size"] = size
    best["scores"] = {f"{c['kind']}{i}": round(c["score"], 4) for i, c in enumerate(cands)}
    return best


def to_ground(p):
    """A part standing on the table reaches down to it (y = 0): an up-aligned primitive is extended to the table."""
    if p["kind"] == "cylinder" and abs(p["axis"] @ UP) > .95:
        top = p["centre"][1] + p["h"] / 2
        if p["centre"][1] - p["h"] / 2 > 0:
            p["h"] = top; p["centre"] = p["centre"].copy(); p["centre"][1] = top / 2
    elif p["kind"] == "box" and abs(p["axes"][1] @ UP) > .95:
        top = p["centre"][1] + p["ext"][1] / 2
        if p["centre"][1] - p["ext"][1] / 2 > 0:
            p["ext"] = p["ext"].copy(); p["ext"][1] = top; p["centre"] = p["centre"].copy(); p["centre"][1] = top / 2
    return p


def mesh_of(p, detail=60):
    import trimesh
    if p["kind"] == "cylinder":
        m = trimesh.creation.cylinder(radius=p["r"], height=p["h"], sections=96)
        a, u, v = basis(p["axis"])
        T = np.eye(4); T[:3, :3] = np.stack([u, v, a], 1); T[:3, 3] = p["centre"]
    else:
        m = trimesh.creation.box(extents=p["ext"])
        T = np.eye(4); T[:3, :3] = p["axes"].T; T[:3, 3] = p["centre"]
    m.apply_transform(T)
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=p["size"] / detail)
    return trimesh.Trimesh(v, f, process=True)


def colour(mesh, X, C, body, reach):
    """Vertex colours from the observed points within reach (median of the 6 nearest), else the body colour."""
    from scipy.spatial import cKDTree
    d, j = cKDTree(X).query(mesh.vertices, k=6)
    col = np.median(C[j], 1)
    seen = d[:, 0] <= reach
    col[~seen] = body
    return np.clip(col, 0, 255).astype(np.uint8), seen


def snap(new, partners, direction, max_move):
    """Move the new part's mesh along -direction until it touches its partners (closes the gap), at most max_move."""
    import trimesh
    S = np.concatenate([trimesh.sample.sample_surface(p, 4000, seed=0)[0] for p in partners])
    from scipy.spatial import cKDTree
    tree = cKDTree(S)
    V = trimesh.sample.sample_surface(new, 4000, seed=1)[0]
    best, gap0 = 0.0, float(tree.query(V)[0].min())
    for s in np.linspace(0, max_move, 41)[1:]:
        g = float(tree.query(V - direction * s)[0].min())
        if g < gap0 * .25 or g < 1e-3:
            best = s
            break
    return best
