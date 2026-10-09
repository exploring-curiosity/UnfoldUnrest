"""Offline 3D of one object from its continuity family (ledger O-111): built on request, never at ingest, never stored in
the index (late materialization, as the viewer's masks and thumbnails).

  frames    up to N of the family's sighting frames (its most different views first), straightened to a 90 deg pinhole
  geometry  upstream LingBot-Map on those frames, in time order: per frame camera-to-world pose, depth and confidence
  object    each frame's own cut (the ingest's mask) carried onto the straightened frame and the model's grid
  align     each frame's object points rigidly re-aligned to the others (trimmed ICP: Besl & McKay 1992, Chetverikov
            2002), the object-level fusion of Co-Fusion / MaskFusion, so camera drift between distant views does not
            smear the object
  surface   TSDF fusion over the object's box (Curless & Levoy 1996) with free space carved where a ray passes in front
            of the observed surface outside the object's mask (a visual hull, Laurentini 1994); marching cubes
  change    what no rigid motion explains: each frame's residual to the others after alignment, over the depth noise
            the model shows on a static scene (O-107: ~2.6% of depth per frame) - near 1 for a rigid object
"""
from __future__ import annotations

import numpy as np

ERODE = 7                   # BuddyBuilder: masks on the 518 px grid are eroded by 3 px before their depth is used
DEPTH_NOISE = .026          # O-107: view-dependent depth error per frame, fraction of depth (static kitchen)
MAX_TURN, MAX_SHIFT = 20.0, .25   # an ICP correction is a drift fix: at most 20 deg and a quarter of the object's size


def pick_frames(sightings, n=24, prefer=()):
    """sightings: [(frame, id)]; the preferred frames (a view bank) first, then the rest evenly over time."""
    ks = sorted({k for k, _ in sightings})
    chosen = [k for k in dict.fromkeys(prefer) if k in set(ks)][:n]
    rest = [k for k in ks if k not in set(chosen)]
    need = n - len(chosen)
    if need > 0 and rest:
        chosen += [rest[i] for i in np.unique(np.linspace(0, len(rest) - 1, min(need, len(rest))).round().astype(int))]
    return sorted(chosen)


def rect_mask(cut160, straight, size=640):
    """A fisheye cut (160 x 160 cells) on the straightened frame (size x size)."""
    m = cut160[straight.cell_y, straight.cell_x]
    inside = (straight.mx >= 0) & (straight.mx <= size - 1) & (straight.my >= 0) & (straight.my <= size - 1)
    return m & inside


def to_grid(m, g):
    import cv2
    return cv2.resize(m.astype(np.float32), (g, g), interpolation=cv2.INTER_AREA) >= .5


def _rigid(P, Q):
    """Least-squares R, t with R P + t ~ Q (Kabsch)."""
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def icp(P, Q, iters=30, trim=.8):
    """Trimmed point-to-point ICP of P onto Q: (R, t, rms of the kept pairs)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(Q)
    R, t = np.eye(3), np.zeros(3)
    rms = np.inf
    for _ in range(iters):
        X = P @ R.T + t
        d, j = tree.query(X)
        keep = d <= np.quantile(d, trim)
        dR, dt = _rigid(X[keep], Q[j[keep]])
        R, t = dR @ R, dR @ t + dt
        new = float(np.sqrt(np.mean(d[keep] ** 2)))
        if abs(rms - new) < 1e-7:
            break
        rms = new
    return R, t, rms


def fuse_points(W, masks, conf, cols, conf_q=.3):
    """Each frame's object points (masks eroded by one cell: depth bleeds across an object's edge) and colours."""
    import cv2
    pts, rgb = [], []
    for s in range(len(W)):
        m = cv2.erode(masks[s].astype(np.uint8), np.ones((ERODE, ERODE), np.uint8)) > 0
        if m.sum() < 20:
            m = masks[s]
        c = conf[s][m]
        ok = c >= np.quantile(c, conf_q) if len(c) else np.zeros(0, bool)
        pts.append(W[s][m][ok]); rgb.append(cols[s][m][ok])
    return pts, rgb


def align(pts, depth_med, passes=2):
    """Re-align each frame's object points to the others (the largest frame fixed); returns the moved points, each
    frame's (R, t) and its residual over the expected depth noise."""
    order = np.argsort([-len(p) for p in pts])
    T = [(np.eye(3), np.zeros(3)) for _ in pts]
    cur = [p.copy() for p in pts]
    res = np.full(len(pts), np.nan)
    allp = np.concatenate([p for p in pts if len(p)])
    diag = float(np.linalg.norm(np.quantile(allp, .98, 0) - np.quantile(allp, .02, 0)))
    rejected = 0
    for _ in range(passes):
        for s in order[1:]:
            others = np.concatenate([cur[o] for o in range(len(pts)) if o != s and len(cur[o])])
            if len(pts[s]) < 30 or len(others) < 30:
                continue
            R, t, rms = icp(pts[s], others)
            # a drift fix only: a large turn or shift is ICP sliding on a partial view (O-111: one frame took 79 deg)
            ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            shift = np.linalg.norm(pts[s].mean(0) @ R.T + t - pts[s].mean(0))
            if ang > MAX_TURN or shift > MAX_SHIFT * diag:
                rejected += 1
                continue
            T[s] = (R, t); cur[s] = pts[s] @ R.T + t
    for s in range(len(pts)):
        others = [cur[o] for o in range(len(pts)) if o != s and len(cur[o])]
        if len(cur[s]) < 30 or not others:
            continue
        from scipy.spatial import cKDTree
        d, _ = cKDTree(np.concatenate(others)).query(cur[s])
        res[s] = float(np.sqrt(np.mean(np.minimum(d, np.quantile(d, .9)) ** 2))) / (DEPTH_NOISE * depth_med[s])
    align.rejected = rejected
    return cur, T, res


def tsdf(pose, depth, masks, T, lo, hi, res=96, hfov=None, trunc=None):
    """Truncated signed distance over the box [lo, hi]: object pixels add their surface, pixels outside the object
    carve the space in front of what they see. T: each frame's rigid correction (applied to its camera)."""
    from . import mapcheck as MC
    g = depth.shape[-1]
    Rc, C = MC.cameras(np.asarray(pose, np.float64))
    fx, fy = MC.focals(np.asarray(pose, np.float64), g, hfov)
    ax = [np.linspace(lo[i], hi[i], res) for i in range(3)]
    X = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    vs = float(np.max((hi - lo) / (res - 1)))
    trunc = trunc or 3 * vs
    D = np.zeros(len(X), np.float32); Wt = np.zeros(len(X), np.float32)
    free = np.zeros(len(X), np.int16); behind = np.zeros(len(X), np.int16)
    import cv2
    core = [cv2.erode(masks[s].astype(np.uint8), np.ones((ERODE, ERODE), np.uint8)) > 0 for s in range(len(depth))]
    for s in range(len(depth)):
        R, t = T[s]                                          # object points of frame s were moved by (R, t): move
        Rw, Cw = R @ Rc[s], R @ C[s] + t                     # its camera the same way
        Xc = (X - Cw) @ Rw                                   # world -> camera (Rw is camera-to-world)
        z = Xc[:, 2]
        front = z > 1e-3
        u = np.where(front, Xc[:, 0] / np.maximum(z, 1e-3) * fx[s] + g / 2, -1)
        v = np.where(front, Xc[:, 1] / np.maximum(z, 1e-3) * fy[s] + g / 2, -1)
        iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
        ok = front & (iu >= 0) & (iu < g) & (iv >= 0) & (iv < g)
        d = np.zeros(len(X)); m = np.zeros(len(X), bool)
        d[ok] = depth[s][iv[ok], iu[ok]]; m[ok] = masks[s][iv[ok], iu[ok]]
        mc = np.zeros(len(X), bool); mc[ok] = core[s][iv[ok], iu[ok]]
        sdf = d - z
        on = ok & mc & (sdf > -trunc)                        # surface only from the mask's core (depth bleeds at edges)
        D[on] = (D[on] * Wt[on] + np.clip(sdf[on] / trunc, -1, 1)) / (Wt[on] + 1); Wt[on] += 1
        free[ok & ~m & (sdf > trunc)] += 1                   # the ray passes this voxel and sees something else
        behind[ok & m & (sdf <= -trunc)] += 1                # hidden behind the object's own surface
    # a solid: surface where it was seen, inside where only hidden behind the object (never seen through), outside
    # where seen through at least twice and more often than seen on (the visual hull's carving), else outside
    vol = np.where(Wt > 0, D, np.where((behind >= 1) & (free < 2), -1.0, 1.0))
    vol[(free >= 2) & (free > Wt)] = 1.0
    return vol.reshape(res, res, res).astype(np.float32), ax, vs


def mesh_from(vol, ax, keep_largest=True):
    import trimesh
    from skimage import measure
    sp = [a[1] - a[0] for a in ax]
    v, f, _, _ = measure.marching_cubes(vol, 0.0, spacing=sp)
    v += np.array([a[0] for a in ax])
    m = trimesh.Trimesh(v, f, process=True)
    if keep_largest and len(m.faces):
        parts = m.split(only_watertight=False)
        if len(parts):
            m = max(parts, key=lambda p: len(p.faces))
    return m


UNSEEN = np.array([201, 207, 218], np.uint8)       # the viewer's line grey: surface no camera saw (the hull's guess)


def colour(mesh, pts, rgb, near, k=8):
    """Vertex colours from the observed points within `near`; vertices farther from every observation are the visual
    hull's inference and take a neutral grey. Returns (colours, observed mask)."""
    from scipy.spatial import cKDTree
    P = np.concatenate(pts); Cc = np.concatenate(rgb)
    d, j = cKDTree(P).query(mesh.vertices, k=k)
    col = np.median(Cc[j], 1).astype(np.uint8)
    seen = d[:, 0] <= near
    col[~seen] = UNSEEN
    return col, seen


def seen_from_views(mesh, pose, masks, cols, T, used, near):
    """Which surface a camera saw, for a mesh that is not built from the views (ShapeR's whole shape, O-121): a vertex is
    seen when, in some kept view, it is the first surface along its ray (within `near` of the mesh's own z-buffer at
    its cell, plus the depth change a face slanted to the ray makes across one cell) and falls inside the object's
    mask there; its colour is the mean of those views' pixels. Everything else
    is the generator's inference and takes UNSEEN. Unlike colour(), this does not ask the surface to lie on the
    observed points, so an offset surface (ShapeR's -2..-4% depth bias, O-119c) is still tagged by what was in view.
    Returns (colours, seen mask)."""
    from . import mapcheck as MC
    g = masks.shape[-1]
    V = np.asarray(mesh.vertices, np.float64)
    N = np.asarray(mesh.vertex_normals, np.float64)
    Rc, C = MC.cameras(np.asarray(pose, np.float64))
    fx, fy = MC.focals(np.asarray(pose, np.float64), g)
    acc = np.zeros((len(V), 3)); cnt = np.zeros(len(V))
    for s in used:
        if not masks[s].any():
            continue
        R, t = T[s]
        Rw, Cw = R @ Rc[s], R @ C[s] + t
        _, zf = render(mesh, Rw, Cw, fx[s], fy[s], g)
        Xc = (V - Cw) @ Rw
        z = Xc[:, 2]
        ok = z > 1e-3
        u = np.full(len(V), -1); v = np.full(len(V), -1)
        u[ok] = np.floor(Xc[ok, 0] / z[ok] * fx[s] + g / 2).astype(int)
        v[ok] = np.floor(Xc[ok, 1] / z[ok] * fy[s] + g / 2).astype(int)
        inn = ok & (u >= 0) & (u < g) & (v >= 0) & (v < g)
        i = np.flatnonzero(inn)
        # the z-buffer holds a cell's nearest sample; on a face slanted by theta to the ray the depth changes by
        # z tan(theta) / f across one cell, so the tolerance grows with it (tan capped at 4: past ~76 deg is grazing)
        ray = Xc[i] / np.linalg.norm(Xc[i], axis=1, keepdims=True)
        cos = np.abs(np.einsum("ij,ij->i", N[i] @ Rw, ray))
        tan = np.minimum(np.sqrt(np.maximum(1 - cos ** 2, 0)) / np.maximum(cos, 1e-6), 4.0)
        tol = near + z[i] * tan / min(fx[s], fy[s])
        hit = masks[s][v[i], u[i]] & (zf[v[i], u[i]] > 0) & (z[i] <= zf[v[i], u[i]] + tol)
        i = i[hit]
        acc[i] += cols[s][v[i], u[i]]; cnt[i] += 1
    seen = cnt > 0
    col = np.tile(UNSEEN, (len(V), 1))
    col[seen] = np.clip(acc[seen] / cnt[seen, None], 0, 255).astype(np.uint8)
    return col, seen


def shape_change(cur, groups, min_pts=60):
    """Shape change between presences over the scatter within one: frames are grouped by presence (the id they show);
    within = median trimmed residual of a frame to the other frames of its own presence, between = after one rigid ICP
    of a presence onto each other presence. ~1 for a rigid object; above 1 when its shape differed between presences
    (an F-like ratio in which the object is its own control: density, coverage and depth noise cancel)."""
    from scipy.spatial import cKDTree

    def rms(P, Q):
        d, _ = cKDTree(Q).query(P)
        d = np.minimum(d, np.quantile(d, .9))
        return float(np.sqrt(np.mean(d ** 2)))
    gs = {}
    for s, gid in enumerate(groups):
        if len(cur[s]) >= 30:
            gs.setdefault(gid, []).append(s)
    within, between = [], []
    for gid, ss in gs.items():
        for s in ss:
            oth = [cur[o] for o in ss if o != s]
            if oth and sum(map(len, oth)) >= min_pts:
                within.append(rms(cur[s], np.concatenate(oth)))
    keys = [g for g in gs if sum(len(cur[s]) for s in gs[g]) >= min_pts]
    for x in range(len(keys)):
        for y in range(len(keys)):
            if x == y:
                continue
            P = np.concatenate([cur[s] for s in gs[keys[x]]]); Q = np.concatenate([cur[s] for s in gs[keys[y]]])
            R, t, _ = icp(P, Q)
            between.append(rms(P @ R.T + t, Q))
    if not within or not between:
        return dict(ratio=None, presences=len(keys), within=None, between=None)
    w, b = float(np.median(within)), float(np.median(between))
    return dict(ratio=round(b / max(w, 1e-9), 2), presences=len(keys), within=round(w, 5), between=round(b, 5))


def support_plane(W, masks, T, up_hint, obj=None, ring=(2, 14), thr=.01, iters=1500, max_tilt=60.0, seed=0):
    """The surface the object stands on, from video alone: a RANSAC plane (Fischler & Bolles 1981) through the scene
    points in a ring around the object's mask in every view (each moved by its frame's drift fix T), kept only within
    max_tilt of the cameras' up and, given the object's points, only if it lies UNDER the object: 98% of them above it
    and its lowest 2% within max(15% of its size, 2 cm) of it (the largest plane around the bread bag was another
    surface with 93% of the bag below it, O-111). Its normal is gravity's up for an object resting on a counter or
    floor; the cameras' own up is not (a head looking down tilts it by the pitch: the air fryer settled 80 deg off).
    Returns dict(n (unit, toward up_hint), d with n.x + d = 0, inliers, points, tilt_deg, gap) or None."""
    import cv2
    P = []
    for s in range(len(W)):
        m = masks[s].astype(np.uint8)
        if m.sum() < 20:
            continue
        outer = cv2.dilate(m, np.ones((2 * ring[1] + 1,) * 2, np.uint8)) > 0
        inner = cv2.dilate(m, np.ones((2 * ring[0] + 1,) * 2, np.uint8)) > 0
        R, t = T[s]
        P.append(W[s][outer & ~inner] @ R.T + t)
    if not P:
        return None
    P = np.concatenate(P)
    P = P[np.isfinite(P).all(1)]
    if len(P) < 100:
        return None
    rng = np.random.default_rng(seed)
    up_hint = up_hint / np.linalg.norm(up_hint)
    if obj is not None:
        obj = obj[np.isfinite(obj).all(1)]
        gap = max(.15 * float(np.linalg.norm(np.quantile(obj, .98, 0) - np.quantile(obj, .02, 0))), .02)

    def under(n, off):
        if obj is None:
            return True
        lo = np.quantile(obj @ n - off, .02)
        return -thr <= lo <= gap
    best = None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n @ up_hint < 0:
            n = -n
        if np.degrees(np.arccos(np.clip(n @ up_hint, -1, 1))) > max_tilt:
            continue
        k = int((np.abs(P @ n - n @ a) < thr).sum())
        if (best is None or k > best[0]) and under(n, n @ a):
            best = (k, n, a)
    if best is None:
        return None
    on = np.abs(P @ best[1] - best[1] @ best[2]) < thr                 # least squares on the inliers
    Q = P[on]; c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][-1]
    if n @ up_hint < 0:
        n = -n
    return dict(n=n, d=float(-n @ c), inliers=int(on.sum()), points=int(len(P)),
                tilt_deg=round(float(np.degrees(np.arccos(np.clip(n @ up_hint, -1, 1)))), 1),
                gap=None if obj is None else round(float(np.quantile(obj @ n - n @ c, .02)), 4))


def reconstruct(pose, depth, conf, masks, cols, groups=None, train=None, res=96):
    """O-111's fusion as one function: the views in `train` (all if None) shape the object; returns dict(mesh in LingBot's
    world frame, per view T (drift fix; identity for views not used), used views, dropped views, cur / rgb points,
    vol, ax, vs, plane, cam_up, change (shape change between presences), res (residual over noise per view))."""
    from . import mapcheck as MC
    n = len(pose)
    train = np.ones(n, bool) if train is None else np.asarray(train, bool)
    W = MC.lift(pose, depth)
    pts, rgb = fuse_points(W, masks, conf, cols)
    for s in np.flatnonzero(~train):
        pts[s] = pts[s][:0]
    dmed = np.array([float(np.median(depth[s][masks[s]])) if masks[s].any() else np.nan for s in range(n)])
    cur, T, rr = align(pts, dmed)
    # a view the model posed wrongly does not shape the object: residual past median + 3 robust sigmas (Hampel)
    med = np.nanmedian(rr); mad = 1.4826 * np.nanmedian(np.abs(rr - med))
    use = train & np.isfinite(rr) & (rr <= med + 3 * max(mad, 1e-9))
    dropped = np.flatnonzero(train & np.isfinite(rr) & ~use)
    sel = np.flatnonzero(use)
    change = shape_change([cur[s] for s in sel], [groups[s] for s in sel]) if groups is not None else None
    P = np.concatenate([cur[s] for s in sel if len(cur[s])])
    lo, hi = np.quantile(P, .02, 0), np.quantile(P, .98, 0)
    lo, hi = lo - .15 * (hi - lo), hi + .15 * (hi - lo)
    Ts = [T[s] for s in sel]
    vol, ax, vs = tsdf(pose[sel], depth[sel], masks[sel], Ts, lo, hi, res=res)
    Rc, _ = MC.cameras(np.asarray(pose, np.float64))
    cam_up = -np.mean([T[s][0] @ Rc[s][:, 1] for s in sel], 0); cam_up /= np.linalg.norm(cam_up)
    plane = support_plane(W[sel], masks[sel], Ts, cam_up, obj=P, thr=max(2 * vs, .005))
    if plane is not None:                                    # the visual hull's fill below the counter is the counter
        Xg = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
        vol[(Xg @ plane["n"] + plane["d"]) < .5 * vs] = 1.0
    mesh = mesh_from(vol, ax)
    return dict(mesh=mesh, T=T, used=sel, dropped=dropped, cur=cur, rgb=rgb, vol=vol, ax=ax, vs=vs, plane=plane,
                cam_up=cam_up, change=change, res=rr, P=P, W=W)


def render(mesh, Rw, Cw, fx, fy, g, per_cell=6):
    """The mesh seen by one camera (camera-to-world Rw, centre Cw) on the g grid: (mask, z-depth) by a z-buffer over
    dense surface samples (enough per covered cell, then a 3 x 3 closing for the sampling gaps)."""
    import cv2
    import trimesh
    Xc0 = (np.asarray(mesh.vertices) - Cw) @ Rw
    z0 = Xc0[:, 2]
    if not (z0 > 1e-3).any():
        return np.zeros((g, g), bool), np.zeros((g, g))
    # the surface's area on screen sets the samples: ~per_cell per covered cell
    span = np.ptp(Xc0[z0 > 1e-3, :2] / z0[z0 > 1e-3, None], 0) * [fx, fy]
    n = int(np.clip(per_cell * span[0] * span[1] * 2, 2e4, 2e6))
    X, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    X = np.concatenate([X, np.asarray(mesh.vertices)])
    Xc = (X - Cw) @ Rw
    z = Xc[:, 2]
    ok = z > 1e-3
    u = np.floor(Xc[ok, 0] / z[ok] * fx + g / 2).astype(int); v = np.floor(Xc[ok, 1] / z[ok] * fy + g / 2).astype(int)
    z = z[ok]
    inn = (u >= 0) & (u < g) & (v >= 0) & (v < g)
    zb = np.full(g * g, np.inf)
    np.minimum.at(zb, v[inn] * g + u[inn], z[inn])
    zb = zb.reshape(g, g)
    hit = np.isfinite(zb)
    mask = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    zf = np.where(hit, zb, 0.0)
    if (mask & ~hit).any():                                  # fill the closed cells' depth from their neighbours
        k = np.ones((3, 3), np.float32)
        num = cv2.filter2D(zf.astype(np.float32), -1, k); den = cv2.filter2D(hit.astype(np.float32), -1, k)
        zf = np.where(hit, zf, np.where(den > 0, num / np.maximum(den, 1e-9), 0.0))
    return mask, zf


def view_score(mesh, pose_s, depth_s, mask_s, T_s, hfov=None):
    """One view: mask IoU of the rendered mesh against the observed mask, and the median relative depth error on the
    cells inside both."""
    from . import mapcheck as MC
    g = depth_s.shape[-1]
    Rc, C = MC.cameras(np.asarray(pose_s[None], np.float64))
    fx, fy = MC.focals(np.asarray(pose_s[None], np.float64), g, hfov)
    R, t = T_s
    rm, rz = render(mesh, R @ Rc[0], R @ C[0] + t, fx[0], fy[0], g)
    inter, union = (rm & mask_s).sum(), (rm | mask_s).sum()
    both = rm & mask_s & (depth_s > 0)
    derr = float(np.median(np.abs(rz[both] - depth_s[both]) / depth_s[both])) if both.sum() >= 10 else None
    # where the misses are: rendered outside the outline (too fat / misplaced) vs outline not covered (missing)
    sgn = float(np.median((rz[both] - depth_s[both]) / depth_s[both])) if both.sum() >= 10 else None
    return dict(iou=float(inter / union) if union else None, depth_err=derr, depth_bias=sgn, pixels=int(mask_s.sum()),
                precision=float(inter / rm.sum()) if rm.sum() else None, recall=float(inter / mask_s.sum()) if mask_s.sum() else None,
                rendered=int(rm.sum()))


def register_view(pts_s, model_pts):
    """A held-out view's drift fix: its object points onto the model (trimmed ICP, the same guard as align)."""
    if len(pts_s) < 30 or len(model_pts) < 30:
        return (np.eye(3), np.zeros(3)), False
    R, t, _ = icp(pts_s, model_pts)
    diag = float(np.linalg.norm(np.quantile(model_pts, .98, 0) - np.quantile(model_pts, .02, 0)))
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    shift = np.linalg.norm(pts_s.mean(0) @ R.T + t - pts_s.mean(0))
    if ang > MAX_TURN or shift > MAX_SHIFT * diag:
        return (np.eye(3), np.zeros(3)), False
    return (R, t), True


def holdout(pose, depth, conf, masks, cols, groups=None, folds=4, res=96):
    """O-112: leave-k-out quality. Views are dealt to folds in time order; each fold is scored by a model built without
    it (its drift fix found by registering its own object points to that model, as a new view would be). Returns
    per-view rows and the medians."""
    n = len(pose)
    rows = []
    for f in range(folds):
        test = np.zeros(n, bool); test[f::folds] = True
        if test.all() or not test.any():
            continue
        try:
            r = reconstruct(pose, depth, conf, masks, cols, groups, train=~test, res=res)
        except Exception as e:                                # a fold the fusion cannot build is a failed fold
            rows += [dict(view=int(s), fold=f, iou=0.0, depth_err=None, registered=False, error=str(e)) for s in np.flatnonzero(test)]
            continue
        model_pts = np.concatenate([r["cur"][s] for s in r["used"] if len(r["cur"][s])])
        pts_all, _ = fuse_points(r["W"], masks, conf, cols)
        for s in np.flatnonzero(test):
            if not masks[s].any():
                continue
            Ts, ok = register_view(pts_all[s], model_pts)
            sc = view_score(r["mesh"], pose[s], depth[s], masks[s], Ts)
            rows.append(dict(view=int(s), fold=f, registered=ok, **sc))
    iou = [x["iou"] for x in rows if x["iou"] is not None]
    de = [x["depth_err"] for x in rows if x.get("depth_err") is not None]
    return dict(views=rows, iou_median=float(np.median(iou)) if iou else None, iou_mean=float(np.mean(iou)) if iou else None,
                depth_err_median=float(np.median(de)) if de else None, scored=len(rows))


def guided_filter(I, p, r, eps):
    """The colour guided filter (He, Sun & Tang 2010, eqs. 19-21): p filtered with the edges of the colour image I."""
    import cv2
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)
    mI, mp = box(I), box(p)
    cov = box(I * p[..., None]) - mI * mp[..., None]
    var = np.empty(I.shape[:2] + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            var[..., i, j] = var[..., j, i] = box(I[..., i] * I[..., j]) - mI[..., i] * mI[..., j]
    var += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(var, cov[..., None])[..., 0]
    b = mp - (a * mI).sum(-1)
    return (box(a) * I).sum(-1) + box(b)


def refine_mask(cut640, rect, g, band=10, max_change=.4):
    """The ingest's coarse cut (4 px cells on the fisheye frame) snapped to the image: GrabCut (Rother, Kolmogorov &
    Blake 2004) inside a band of +- band px around the cut only (deeper inside is the object, farther out is not), so
    it can move the outline but never leave the band. The cut is kept when the refined area changes by more than
    max_change or GrabCut fails. On a blocky cut of a textured square: IoU .931 -> 1.0 (guided filter, r 16: .987)."""
    import cv2
    if cut640.sum() < 50:
        return to_grid(cut640, g)
    c = cut640.astype(np.uint8)
    k = np.ones((2 * band + 1,) * 2, np.uint8)
    inner, outer = cv2.erode(c, k) > 0, cv2.dilate(c, k) > 0
    m = np.full(c.shape, cv2.GC_BGD, np.uint8)
    m[outer] = cv2.GC_PR_BGD; m[cut640] = cv2.GC_PR_FGD; m[inner] = cv2.GC_FGD
    if not inner.any():                                       # too thin for a sure inside: trust the cut
        return to_grid(cut640, g)
    try:
        bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(rect, cv2.COLOR_RGB2BGR), m, None, bg, fg, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return to_grid(cut640, g)
    out = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
    if abs(int(out.sum()) - int(cut640.sum())) > max_change * cut640.sum():
        return to_grid(cut640, g)
    return to_grid(out, g)