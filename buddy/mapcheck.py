"""Map fidelity without ground truth (O-98): is a stream's depth + camera self-consistent?

The measure of Luo et al., "Consistent Video Depth Estimation" (SIGGRAPH 2020), and the multi-view geometric
consistency of Schonberger et al. (ECCV 2016): lift frame i's depth with i's camera, project the points into frame j
with j's camera, and ask
  geometric:    does j's own depth agree where they land?  |z_in_j - depth_j| / depth_j
  photometric:  do they land on the same content?  |RGB_i - RGB_j at the landing cell|, against the identity warp
                (j's cell at the same position); a ratio below 1 means the geometry moves pixels onto their content.
Pairs at several gaps separate frame-to-frame jitter (gap 1) from drift (gap 25). Disagreement also counts
occlusion and moving hands, the same for every setting compared on the same frames.

Everything lives on the depth grid the model saw (g x g, square). The camera of a frame is a pinhole with either the
model's own field of view (pose_enc[7:9], what its depth and pose are consistent with) or a calibrated one (hfov in
degrees, square), and its pose is read camera-to-world (O-80) unless c2w=False.
"""
from __future__ import annotations

import numpy as np


def cameras(pose, c2w=True):
    """pose_enc (S, 9) -> camera-to-world rotations (S, 3, 3) and centres (S, 3)."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(np.asarray(pose[:, 3:7], np.float64)).as_matrix()
    t = np.asarray(pose[:, :3], np.float64)
    if c2w:
        return R, t
    Rc = R.transpose(0, 2, 1)
    return Rc, -np.einsum("nij,nj->ni", Rc, t)


def focals(pose, g, hfov=None):
    """(fx, fy) per frame on the g grid: the model's own (fov_w = pose[:, 8], fov_h = pose[:, 7]) or a fixed hfov."""
    if hfov is not None:
        f = np.full(len(pose), (g / 2) / np.tan(np.radians(hfov) / 2))
        return f, f.copy()
    return (g / 2) / np.tan(pose[:, 8] / 2), (g / 2) / np.tan(pose[:, 7] / 2)


def sample(a, x, y):
    """Bilinear sample of a (g, g[, k]) at continuous cell coordinates (cell centres at .5); x, y inside [.5, g - .5]."""
    x0, y0 = np.floor(x - .5).astype(np.int64), np.floor(y - .5).astype(np.int64)
    wx, wy = x - .5 - x0, y - .5 - y0
    if a.ndim == 3:
        wx, wy = wx[:, None], wy[:, None]
    return ((1 - wy) * ((1 - wx) * a[y0, x0] + wx * a[y0, x0 + 1]) + wy * ((1 - wx) * a[y0 + 1, x0] + wx * a[y0 + 1, x0 + 1]))


def pair(i, j, depth, keep, Rc, C, fx, fy, img=None):
    """Errors of frame i's kept cells landing in frame j: (relative depth errors, photometric |dRGB|, identity
    |dRGB|, share of i's kept cells that land inside j in front of it). j's depth is sampled bilinearly in inverse
    depth (exact on planes, where inverse depth is linear in the image), its colour bilinearly."""
    g = depth.shape[-1]
    ys, xs = np.nonzero(keep[i])
    z = depth[i, ys, xs].astype(np.float64)
    X = np.stack([(xs + .5 - g / 2) / fx[i] * z, (ys + .5 - g / 2) / fy[i] * z, z], 1) @ Rc[i].T + C[i]
    c = (X - C[j]) @ Rc[j]                                   # Rc is camera-to-world: camera coords = Rc^T (X - C)
    front = c[:, 2] > 1e-6
    zc = np.where(front, c[:, 2], 1.0)
    u, v = c[:, 0] / zc * fx[j] + g / 2, c[:, 1] / zc * fy[j] + g / 2
    ok = front & (u >= .5) & (u <= g - .5 - 1e-9) & (v >= .5) & (v <= g - .5 - 1e-9)
    inv = 1.0 / np.maximum(depth[j].astype(np.float64), 1e-9)
    dj = 1.0 / np.maximum(sample(inv, u[ok], v[ok]), 1e-12)
    geo = np.abs(c[ok, 2] - dj) / dj
    ph = idn = None
    if img is not None:
        a = img[i, ys[ok], xs[ok]]
        ph = np.abs(a - sample(img[j], u[ok], v[ok])).mean(1)
        idn = np.abs(a - img[j, ys[ok], xs[ok]]).mean(1)
    return geo, ph, idn, ok.mean() if len(ok) else 0.0


def fidelity(pose, depth, conf, gaps=(1, 2, 5, 10, 25), hfov=None, c2w=True, img=None, conf_q=.4, every=1):
    """Per gap: median relative depth error, share within 5% and 2%, photometric ratio (mean over cells of the
    landed |dRGB| over the identity |dRGB|), share landing in view, pairs. Frame i's cells kept: confidence above
    its frame's conf_q quantile and positive depth."""
    g = depth.shape[-1]
    Rc, C = cameras(np.asarray(pose, np.float64), c2w)
    fx, fy = focals(np.asarray(pose, np.float64), g, hfov)
    depth = np.asarray(depth, np.float32)
    conf = np.asarray(conf, np.float32)
    keep = (conf >= np.quantile(conf.reshape(len(conf), -1), conf_q, axis=1)[:, None, None]) & (depth > 0)
    out = {}
    for gap in gaps:
        G, P, I, V = [], [], [], []
        for i in range(0, len(depth) - gap, every):
            geo, ph, idn, inview = pair(i, i + gap, depth, keep, Rc, C, fx, fy, img)
            G.append(geo); V.append(inview)
            if ph is not None:
                P.append(ph); I.append(idn)
        if not G:
            continue
        G = np.concatenate(G)
        row = dict(pairs=len(V), median=round(float(np.median(G)), 4), within5=round(float((G < .05).mean()), 3),
                   within2=round(float((G < .02).mean()), 3), in_view=round(float(np.mean(V)), 3))
        if P:
            P, I = np.concatenate(P), np.concatenate(I)
            row["photo_ratio"] = round(float(P.mean() / max(I.mean(), 1e-9)), 3)
        out[gap] = row
    return out


def lift(pose, depth, c2w=True, hfov=None):
    """World points (S, g, g, 3) of every cell from depth + camera (the product's lifting, on the model's grid)."""
    g = depth.shape[-1]
    Rc, C = cameras(np.asarray(pose, np.float64), c2w)
    fx, fy = focals(np.asarray(pose, np.float64), g, hfov)
    c = np.arange(g) + .5 - g / 2
    u, v = np.meshgrid(c, c)
    cam = np.stack([u[None] / fx[:, None, None], v[None] / fy[:, None, None], np.ones((len(depth), g, g))], -1)
    cam = cam * np.asarray(depth, np.float64)[..., None]
    return np.einsum("sij,shwj->shwi", Rc, cam) + C[:, None, None]


def thickness(W, keep, pose, gaps=(1, 5, 25), every=1, c2w=True):
    """How thick the fused cloud is, whatever made the points (depth + camera, or a point head): for frame i's kept
    points that frame j's camera sees, the distance to the local surface through j's 6 nearest kept points
    (point-to-plane, so the grid's spacing does not count) over the point's distance from camera i. A crisp map
    scores near 0; a surface smeared across frames scores its spread. Per gap: median, share under 1% and 2%."""
    from scipy.spatial import cKDTree
    g = W.shape[1]
    Rc, C = cameras(np.asarray(pose, np.float64), c2w)
    fx, fy = focals(np.asarray(pose, np.float64), g)
    out = {}
    for gap in gaps:
        R = []
        for i in range(0, len(W) - gap, every):
            j = i + gap
            Xi, Xj = W[i][keep[i]], W[j][keep[j]]
            if not len(Xi) or not len(Xj):
                continue
            c = (Xi - C[j]) @ Rc[j]
            zc = np.where(c[:, 2] > 1e-6, c[:, 2], 1.0)
            u, v = c[:, 0] / zc * fx[j] + g / 2, c[:, 1] / zc * fy[j] + g / 2
            seen = (c[:, 2] > 1e-6) & (u >= 0) & (u < g) & (v >= 0) & (v < g)
            if not seen.any():
                continue
            if len(Xj) < 6:
                continue
            _, nb = cKDTree(Xj).query(Xi[seen], 6)        # point-to-plane: the local surface through j's 6 nearest
            Q = Xj[nb]; m = Q.mean(1); Q = Q - m[:, None]
            nrm = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))[1][:, :, 0]
            d = np.abs(np.einsum("ni,ni->n", Xi[seen] - m, nrm))
            R.append(d / np.maximum(np.linalg.norm(Xi[seen] - C[i], axis=1), 1e-9))
        if R:
            R = np.concatenate(R)
            out[gap] = dict(median=round(float(np.median(R)), 4), within1=round(float((R < .01).mean()), 3),
                            within2=round(float((R < .02).mean()), 3))
    return out
