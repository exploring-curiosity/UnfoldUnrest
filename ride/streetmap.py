"""The street map from LingBot's cameras and depth.

ground   the camera is fixed to the bike, so the road is fixed in the camera's frame: one RANSAC plane through the
         road just ahead in every frame (camera coordinates, pooled), and per frame the camera's height above it.
         A single world plane does not work: the poses drift (over 30 s it ends up rolled ~20 deg against the cameras
         and above the last ones). Checked by eye: the plane's horizon runs through the road's vanishing point
scale    metres from the camera's height above its own road: the median height is set to CAM_H (a handlebar / chest
         mount on a bike); video gives no scale of its own
plan     a top-down frame on the ground: x along the ride's main direction, y across it (left positive), origin at
         the start; every world point maps to (x, y) metres and a height above the road
ortho    a top-down photo of the street: each frame's pixels cast along their rays onto its road (as objects are, so
         map and objects agree), kept where the frame's depth agrees that the pixel is road, within REACH m; averaged
         over frames, nearer views weighted more
rays     anything standing on the road is placed where the camera ray through its lowest pixel meets the ground
         (flat-ground localisation), which needs no depth at the object itself
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from buddy import mapcheck as MC

from . import G

CAM_H = 1.1          # m: assumed camera height above the road (handlebar / chest mount)
REACH = 12.0         # m: beyond this the road is seen too obliquely for the map
SIDE = 14.0          # m: the map reaches this far either side of the ride (standing cars sit up to ~12 m out)


class Street:
    def __init__(self, pose, depth, conf, size):
        self.pose, self.depth, self.conf = pose, depth, conf
        self.W, self.H = size                                    # the frames' pixels (detections live there)
        self.R, self.C = MC.cameras(pose)
        self.fx, self.fy = MC.focals(pose, G)
        self.nk, self.dk, self.n, self.d = self.ground()
        h = np.einsum("ij,ij->i", self.C, self.nk) + self.dk
        self.scale = CAM_H / float(np.median(h))                  # metres per model unit
        self.heights = h * self.scale
        # plan axes: x along the ride, y across (left), z up
        P = self.C - np.outer(self.C @ self.n + self.d, self.n)
        D = P - P[0]
        u, s, vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
        ex = vt[0] - (vt[0] @ self.n) * self.n; ex /= np.linalg.norm(ex)
        if (P[-1] - P[0]) @ ex < 0:
            ex = -ex
        ey = np.cross(self.n, ex)
        self.ex, self.ey, self.o = ex, ey, P[0]

    # -- geometry -------------------------------------------------------------------------------------------
    def lift(self, k, us, vs):
        """Grid cells (us, vs on the G grid) of frame k -> world points from its depth."""
        z = self.depth[k][vs, us].astype(np.float64)
        X = np.stack([(us + .5 - G / 2) / self.fx[k] * z, (vs + .5 - G / 2) / self.fy[k] * z, z], 1)
        return X @ self.R[k].T + self.C[k]

    def road_cam(self, k, us, vs):
        """Grid cells of frame k -> points in its camera's own frame (x right, y down, z forward), model units."""
        z = self.depth[k][vs, us].astype(np.float64)
        return np.stack([(us + .5 - G / 2) / self.fx[k] * z, (vs + .5 - G / 2) / self.fy[k] * z, z], 1)

    def mount(self, iters=800, seed=0):
        """The camera is fixed to the bike, so the road sits still in the camera's frame: one RANSAC plane through the
        middle of every frame's lower part (each frame's points divided by their median depth, so all weigh the same;
        the corners, where the wide lens bends most, left out), refit on the inliers -> its up normal in the camera's
        frame. Then per frame the camera's height above that plane (median over its inliers, model units)."""
        rng = np.random.default_rng(seed)
        vs, us = np.mgrid[int(G * .55):int(G * .95):4, int(G * .25):int(G * .75):4]
        us, vs = us.ravel(), vs.ravel()
        Pk = [self.road_cam(k, us, vs) for k in range(len(self.depth))]
        P = np.concatenate([q / np.median(q[:, 2]) for q in Pk[::2]])
        thr, best = .02, (0, None)
        for _ in range(iters):
            a, b, c = P[rng.choice(len(P), 3, replace=False)]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-12:
                continue
            n /= np.linalg.norm(n)
            n = n if n[1] < 0 else -n
            if -n[1] < np.cos(np.radians(45)):
                continue
            k = int((np.abs((P - a) @ n) < thr).sum())
            if k > best[0]:
                best = (k, (n, a))
        n, a = best[1]
        Q = P[np.abs((P - a) @ n) < thr]; c = Q.mean(0)
        n = np.linalg.svd(Q - c, full_matrices=False)[2][-1]
        n = n if n[1] < 0 else -n
        h = np.array([np.median(-(q @ n)[np.abs(-(q @ n) / np.median(-(q @ n)) - 1) < .15]) for q in Pk])
        return n, h

    def ground(self, smooth=3):
        """-> per-frame road planes in the world (normals (S, 3), offsets (S,)) and the plan's floor (n, d)."""
        self.n_cam, h = self.mount()
        h = np.array([np.median(h[max(0, k - smooth):k + smooth + 1]) for k in range(len(h))])
        nk = self.R @ self.n_cam
        dk = h - np.einsum("ij,ij->i", self.C, nk)
        n = nk.mean(0); n /= np.linalg.norm(n)
        return nk, dk, n, float(np.median(h - self.C @ n))

    def plan(self, X):
        """World points -> (x, y, z) metres in the plan (z = height above the road)."""
        X = np.atleast_2d(X)
        D = X - self.o
        return np.stack([D @ self.ex, D @ self.ey, X @ self.n + self.d], 1) * self.scale

    def cam_plan(self):
        return self.plan(self.C)

    def ray_ground(self, k, x, y):
        """Frame pixels (x, y) of frame k -> world points where their rays meet the ground (nan if they do not)."""
        x, y = np.atleast_1d(x).astype(np.float64), np.atleast_1d(y).astype(np.float64)
        u, v = x * G / self.W, y * G / self.H
        dirs = np.stack([(u - G / 2) / self.fx[k], (v - G / 2) / self.fy[k], np.ones_like(u)], 1) @ self.R[k].T
        den = dirs @ self.nk[k]
        t = -(self.C[k] @ self.nk[k] + self.dk[k]) / np.where(np.abs(den) > 1e-9, den, np.nan)
        X = self.C[k] + t[:, None] * dirs
        X[~(t > 0)] = np.nan
        return X

    def height_at(self, k, x, y, r):
        """Height above the road (m) of the point on frame k's ray through pixel (x, y) at r m from the camera along the
        road: how tall an object standing r m away is when (x, y) is its top."""
        u, v = np.atleast_1d(x) * G / self.W, np.atleast_1d(y) * G / self.H
        dirs = np.stack([(u - G / 2) / self.fx[k], (v - G / 2) / self.fy[k], np.ones_like(u)], 1) @ self.R[k].T
        up = dirs @ self.nk[k]
        flat = np.linalg.norm(dirs - np.outer(up, self.nk[k]), axis=1)
        return self.heights[k] + up / flat * np.atleast_1d(r)

    # -- the top-down photo ----------------------------------------------------------------------------------
    def ortho(self, frames, res=.05, pad=4.0, every=1, tol=.05):
        """-> (image uint8 (h, w, 3), extent dict(x0, x1, y0, y1, res)) of the road seen from above. Each grid pixel's ray
        is cast onto its frame's road (as objects are placed); it counts as road only where the frame's depth agrees
        (the depth point lies within tol of the range from the ray's hit), so walls and planters stay out."""
        cam = self.cam_plan()
        x0, x1 = cam[:, 0].min() - pad, cam[:, 0].max() + pad
        y0, y1 = cam[:, 1].min() - SIDE, cam[:, 1].max() + SIDE
        w, h = int((x1 - x0) / res), int((y1 - y0) / res)
        acc = np.zeros((h * w, 3)); wt = np.zeros(h * w)
        vs, us = np.mgrid[0:G, 0:G]
        us, vs = us.ravel(), vs.ravel()
        xs, ys = (us + .5) * self.W / G, (vs + .5) * self.H / G
        for k in range(0, len(frames), every):
            img = cv2.resize(frames[k], (G, G), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float64)
            Xg = self.ray_ground(k, xs, ys)
            Xd = self.lift(k, us, vs)
            rg = np.linalg.norm(Xg - self.C[k], axis=1) * self.scale
            off = np.linalg.norm(Xd - Xg, axis=1) * self.scale
            keep = np.isfinite(rg) & (rg < REACH) & (off < tol * rg + .1)
            p = self.plan(Xg[keep])
            ix = ((p[:, 0] - x0) / res).astype(int); iy = ((y1 - p[:, 1]) / res).astype(int)
            ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            idx = iy[ok] * w + ix[ok]
            wk = 1.0 / (1.0 + rg[keep][ok]) ** 2
            np.add.at(acc, idx, img[keep][ok] * wk[:, None]); np.add.at(wt, idx, wk)
        out = np.zeros((h * w, 3), np.uint8)
        m = wt > 0
        out[m] = np.clip(acc[m] / wt[m, None], 0, 255)
        img = out.reshape(h, w, 3)
        # close the pinholes between splats (only where there is road nearby)
        filled = cv2.dilate(img, np.ones((5, 5), np.uint8))
        hole = (m.reshape(h, w) == 0)
        img[hole] = filled[hole]
        return img, dict(x0=float(x0), x1=float(x1), y0=float(y0), y1=float(y1), res=res)

    def tall(self, extent, res=.25, hmin=.9, hmax=3.0, max_d=15.0, stride=2, min_frames=3):
        """What stands taller than hmin above the road, from the depth alone (no detector): -> (occupied (h, w) bool,
        tallest height (h, w) m, frames seen (h, w)) on a res-m grid over `extent`. A cell is occupied when points
        between hmin and hmax m above their frame's road land in it from at least min_frames frames. Depth
        discontinuities (pixels smeared between a near edge and the background) are dropped first."""
        x0, y1 = extent["x0"], extent["y1"]
        w, h = int((extent["x1"] - x0) / res) + 1, int((y1 - extent["y0"]) / res) + 1
        seen = np.zeros(h * w, np.int32); top = np.zeros(h * w)
        vs, us = np.mgrid[1:G - 1:stride, 1:G - 1:stride]
        us, vs = us.ravel(), vs.ravel()
        cam = self.cam_plan()
        for k in range(len(self.depth)):
            z = self.depth[k].astype(np.float32)
            gx = np.abs(z[vs, us + 1] - z[vs, us - 1]); gy = np.abs(z[vs + 1, us] - z[vs - 1, us])
            c = self.conf[k][vs, us]
            ok = (np.maximum(gx, gy) / np.maximum(z[vs, us], 1e-6) < .08) & (c >= np.quantile(c, .3))
            X = self.lift(k, us[ok], vs[ok])
            p = self.plan(X)
            p[:, 2] = (X @ self.nk[k] + self.dk[k]) * self.scale
            d = np.linalg.norm(p[:, :2] - cam[k, :2], axis=1)
            keep = (p[:, 2] > hmin) & (p[:, 2] < hmax) & (d < max_d)
            ix = ((p[keep, 0] - x0) / res).astype(int); iy = ((y1 - p[keep, 1]) / res).astype(int)
            inb = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            idx = np.unique(iy[inb] * w + ix[inb])
            seen[idx] += 1
            np.maximum.at(top, iy[inb] * w + ix[inb], p[keep][inb, 2])
        seen, top = seen.reshape(h, w), top.reshape(h, w)
        return seen >= min_frames, top, seen

    def cloud(self, frames, every=3, stride=3, max_d=REACH):
        """A coloured point cloud of the street in plan metres (for the 3D view)."""
        vs, us = np.mgrid[0:G:stride, 0:G:stride]
        us, vs = us.ravel(), vs.ravel()
        cam = self.cam_plan()
        P, Cc = [], []
        for k in range(0, len(frames), every):
            img = cv2.resize(frames[k], (G, G), interpolation=cv2.INTER_AREA)[vs, us]
            c = self.conf[k][vs, us]
            X = self.lift(k, us, vs)
            p = self.plan(X)
            p[:, 2] = (X @ self.nk[k] + self.dk[k]) * self.scale          # height above the frame's own road
            dist = np.linalg.norm(p[:, :2] - cam[k, :2], axis=1)
            keep = (dist < max_d) & (c >= np.quantile(c, .3)) & (p[:, 2] > -.5) & (p[:, 2] < 12)
            P.append(p[keep]); Cc.append(img[keep])
        return np.concatenate(P), np.concatenate(Cc)


def build(out):
    from . import frames as F, geometry as GE
    out = Path(out)
    meta, fr = F.load(out)
    st = Street(*GE.load(out), meta["size"])
    return st, meta, fr


if __name__ == "__main__":
    import sys
    st, meta, fr = build(sys.argv[1])
    cam = st.cam_plan()
    print(f"scale {st.scale:.2f} m/unit; ride {np.linalg.norm(np.diff(cam[:, :2], axis=0), axis=1).sum():.1f} m in "
          f"{meta['times'][-1]:.1f} s; camera height spread {np.percentile(st.heights, [5, 50, 95]).round(2)} m")
    img, ext = st.ortho(fr)
    cv2.imwrite(str(Path(sys.argv[1]) / "ortho.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("ortho", img.shape, ext)
