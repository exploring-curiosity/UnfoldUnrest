# BuddyBuilder: video -> per-part 3D -> animated assembly guide (design + plan, 2026-10-09)

Hackathon scope (VAST Builders Challenge, demo ~4:30pm). Input: a handheld phone video of a build (LEGO first).
Output: an interactive three.js guide (step slider, part fly-in, orbit, parts list, the step's source clip).
Rules: every part rebuilt from the video alone (no LDraw/CAD library); Mac only (M5 Pro, MPS/CoreML); offline;
judged by eye; no subagents.

## Core idea: rest states
When the builder lets go, the scene is still and the moving phone sees it from several sides. Rest states give
(a) parts (the first rest state: parts laid out apart), (b) contacts per state, (c) the finished build's geometry.
A contact first held in rest state i dates the step that made it.

## Stages (each writes work/<name>/<stage>.npz; python -m buddy.<stage> work/<name>)
| stage | module | what |
|---|---|---|
| frames | frames.py | decode, 10 fps, centre square crop -> 518 px |
| cuts | segment.py | FastSAM-s (ElideDB ONNX, CoreML), masks from proto logits at 518 |
| motion | motion.py | DIS flow vs one homography: independently moving pixels (seed for hands) |
| hands | hands.py | skin chroma model learned from moving vs still pixels (MediaPipe failed: read the plate as a palm) |
| geometry | geometry.py | upstream LingBot-Map on MPS: pose, depth, conf per frame, one world frame |
| assembly | states.py, assemble.py, build.py | rest runs (hands < 10% of view), inventory parts, appearance labels (Lab hist + elongation, smallest cut first), finished state re-explained by 3D continuity (ICP of the last joined group), contacts (masks meet + depth agrees), steps, per-part TSDF + carving (object3d.py from ElideDB) |
| guide | guide.py | table frame (support plane), GLB per part, guide.json (steps, approach vector, clip times) |
| viewer | viewer/index.html | three.js 0.160 |

## Known limits / next
- Part meshes come from the finished state's frames only; hidden faces are hull guesses (grey).
- Appearance labels are fragile under lighting change; continuity does the heavy lifting where it can.
- Stretch: in-hand "show" frames for unseen faces; plane snapping for crisp edges; ShapeR completion.
