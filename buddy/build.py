"""The assembly from the stages' outputs: rest states -> parts -> labels -> contacts -> build steps -> part meshes.

python -m buddy.build work/<name>   (after frames, segment, hands, geometry)"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

import warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul")

from . import assemble as A, frames as F, geometry as G, hands as Hd, object3d as O3, segment as S, states as St


def rest_states(fr, cuts, hand):
    share = hand.reshape(len(hand), -1).mean(1)
    rests = []
    for r in St.rest_runs(share, cuts):
        R = St.Rest(frames=r)
        for k in r:
            R.objects[k] = St.object_cuts(fr[k], cuts[k][0], cuts[k][1], hand[k])
        if sum(len(R.objects[k]) for k in r) >= len(r):       # a state that shows things (not the bare cloth)
            rests.append(R)
    return rests


def part_mesh(geo, fr, rests, moves, j, hand, up, res=112, log=print):
    """A part's mesh in the finished build's frame from every rest state that shows it as it sits there: the finished
    state's frames as they are, an earlier state's frames with their cameras moved by that state's group ICP."""
    P, D, C, M, I, src, OK = [], [], [], [], [], [], []
    for i, (R, t, g) in sorted(moves.items()):
        if j not in g:
            continue
        r = rests[i]
        for k in r.frames:
            m = r.labels[k].get(j)
            if m is None or m.sum() < 50:
                continue
            P.append(A.moved_pose(geo.pose[k], R, t) if i != len(rests) - 1 else geo.pose[k])
            others = [mm for jj, mm in r.labels[k].items() if jj != j]
            OK.append(np.any(others + [hand[k]], 0))
            D.append(geo.depth[k]); C.append(geo.conf[k]); M.append(m); I.append(fr[k]); src.append(int(k))
    if len(P) < 2:
        log(f"part {j}: seen in {len(P)} frames, no mesh"); return None
    r = O3.reconstruct(np.array(P), np.array(D), np.array(C), np.array(M), np.array(I).astype(np.float32), res=res)
    mesh = r["mesh"]
    col, seen = O3.colour(mesh, [r["cur"][s] for s in r["used"]], [r["rgb"][s] for s in r["used"]], near=3 * r["vs"])
    mesh.visual.vertex_colors = np.c_[col, np.full(len(col), 255, np.uint8)]
    log(f"part {j}: {len(P)} views ({len(r['used'])} used) from frames {sorted(set(src))[:3]}..., "
        f"{len(mesh.vertices)} vertices, {seen.mean():.0%} of the surface observed")
    from . import silfit as SF
    X = r["P"]
    size = float(np.linalg.norm(np.quantile(X, .9, 0) - np.quantile(X, .1, 0)))
    views = SF.Views(np.array(P), M, OK)
    kind, q, loss = SF.fit(views, X, size, SF.inits_from(X, up), log=log)
    log(f"part {j}: silhouette fit -> {kind} (loss {loss:.3f})")
    return dict(mesh=mesh, seen=seen, plane=r["plane"], cam_up=r["cam_up"], frames=src, P=r["P"], prim=(kind, q, loss),
                views=dict(poses=np.array(P), frames=src, masks=np.packbits(np.array(M), axis=-1)),
                rgb=np.concatenate([r["rgb"][s] for s in r["used"]]), pts=np.concatenate([r["cur"][s] for s in r["used"]]))


def run(out, log=print):
    out = Path(out)
    meta, fr = F.load(out)
    cuts = S.load(out)
    hand, _ = Hd.load(out)
    geo = A.Geo(*G.load(out))
    rests = rest_states(fr, cuts, hand)
    log("rest states: " + ", ".join(f"{r.frames[0]}-{r.frames[-1]}" for r in rests))
    parts = St.inventory(rests[0], fr)
    log(f"parts (laid out in frames {rests[0].frames[0]}-{rests[0].frames[-1]}): {len(parts)}")
    for R in rests:
        St.label(R, fr, parts)
        St.contacts(R, geo.depth)
    A.explain_final(geo, rests, fr, parts, log=log)
    St.contacts(rests[-1], geo.depth)
    for R in rests:
        log(f"  state {R.frames[0]}-{R.frames[-1]}: contacts " +
            ", ".join(f"{a}-{b}:{s:.2f}" for (a, b), s in sorted(R.contacts.items())))
    steps = St.steps(rests)
    log("contacts of the finished build: " + ", ".join(f"{a}-{b} (first held in state {i}, {s:.2f})" for i, (a, b), s in steps))
    moves = A.state_moves(geo, rests, log=log)
    up, _ = A.table_normal(geo, rests[-1])
    meshes = {j: part_mesh(geo, fr, rests, moves, j, hand, up, log=log) for j in range(len(parts))}
    with open(out / "assembly.pkl", "wb") as f:
        pickle.dump(dict(rests=[dict(frames=r.frames, contacts=r.contacts, labels=r.labels) for r in rests],
                         parts=[{k: v for k, v in p.items() if k in ("hist", "area", "elong")} for p in parts],
                         steps=steps, meshes=meshes), f)
    return rests, parts, steps, meshes


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
