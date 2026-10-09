# Object tracker, space-time place and object 3D: a handoff for a new project (2026-10-09)

This document describes the object technology built inside ElideDB from 10-06 to 10-09, written so that a new project
in a separate session can reuse it without this repo's history. It covers:

1. the **object tracker**: a class-agnostic segmenter, continuous ids, and the object tree;
2. **space-time place and 3D mapping**: LingBot-Map, the fisheye lens, loop closure, the anchor same-spot test;
3. the **offline 3D rebuild** of one object: fusion, ShapeR's whole shape, and the physics demo.

**How to read the claims.** Every number is measured on this machine, unless marked otherwise.
- `[proven: ID]` names the ledger row in `docs/memory/ledger.md` that holds the setup and the raw result.
- `[assumed]` marks a claim that was never measured.
- All accuracy numbers come from one kitchen (HD-EPIC P01, the dev kitchen), on a few windows labelled by eye.

**Source repo:** `/Users/sudharshanramesh/Studies/MyProjects/ElideDB` (private; nothing here is committed or pushed
yet). The machine is an Apple M5 Pro with 48 GB, torch 2.7.1 on MPS, and ONNX Runtime 1.28 with CoreML.

---

## 1. The owner's rules for this technology

These are the owner's decisions, from 10-06 to 10-08. A new project on the same technology inherits them unless the
owner says otherwise.

**Identity**
- An **id is one continuous presence** of one object: from when it appears to when it leaves view. If it leaves and
  comes back, it gets a **new id**. "As long as it is in the frame consistently it has to maintain one id with the
  timestamps when it started to appear and when it end".
- An id must be in view **more than 1 s** to be persistent.
- **Same, look-alike and different are three relations:**
  - *same* is proven only by continuity (one id, or a continuity-backed link);
  - *look-alike* means "may be the same" (exact identicals stay look-alikes);
  - *different* means outside one object's own variation.
  - Colour and size alone are not enough. Cues must survive other angles and soft, deforming objects (bags).
- **The tree is a graph of continuations, not a linker.** "There is no linking and unlinking of the object ... it's
  just a graph representation of different continuations of objects." Each id keeps its closeness to earlier ids. A
  query picks the level; nothing is merged at ingest.
- **Views join an object only through continuity:** the same track, the same spot, or a carry path. Never through
  appearance alone, which would feed look-alike errors back into the object.
- **No actions in the tracker:** no pick-up, place, open, contact or hand concepts ("that's part of the video memory
  model").

**What the models may be**
- **A segmenter, never a detector.** Everything in view is split by its boundaries, class-agnostic, new objects
  included. Never a vocabulary-bound detector (YOLOE, YOLO-World, OWL, Grounding DINO). The owner tried YOLOE: it needs
  registered classes and misses new objects.
- **No training, no labels, no fitted weights.** Models are frozen. Every threshold is a quantile of the stream's own
  statistics. Labels only grade.
- **Video is the only input** ("I just have video as the input. no extra sensors.").

**Budgets**
- **Index-sized storage:** the object layer is an index, "like a hash map index or b+ trees", about 1% of the video
  or less. Masks and latents are **never stored**; they are re-derived from the video when someone looks (late
  materialization).
- **Ingest at least real time** beside the main ingest. The object layer's budget was about 0.5 s of compute per video
  second.
- **Cost before accuracy.** State a tool's cost per video second against the budget before running it. Never probe the
  accuracy of something that cannot fit (SAM 3 was stopped for this).
- **Weigh every parameter.** Before proposing any option, put accuracy, cost, storage, generalization and no-training
  in one table, and drop any option that fails one.

**3D**
- **3D is BETA.** "The 3d itself isnt any good." The objective is not a map. Space and time continuity is the tool that
  lets an object's other angles and deformed looks join it ("self-reinforce").

**How the owner works**
- **The owner judges by eye** in the object viewer (one port: `:8771`). HD-EPIC's object labels are not the verdict.
  Report cost (x real time, bytes per video minute) beside what the owner sees.
- **Research first, then one test.** No batches of variants, and no long runs to verify an idea: small windows first.
- **One model process at a time** on the GPU. Downloads need the owner's yes (name, source, size).

---

## 2. The whole system on one page

```
video (any fps / size / rotation)
  -> decode once, nearest-frame resample to a 5 fps grid, fit to 640 x 640 (sidefit: aspect kept, black pads)
  -> ObjectStream (python/elidedb/objstream.py), 4 stages on threads, all causal:
       flow     DIS optical flow both ways at 320 px; schedule FastSAM when the camera moved 20 px, or every 3 frames
       fastsam  FastSAM-s (ONNX, CoreML: CPU + Neural Engine) on the scheduled frames -> cuts (masks at 160 x 160)
       tracker  carry each id's mask by a per-pixel flow warp with a forward-backward check; match cuts to tracks
                (IoU >= .5, one-to-one); births; ghosts of ended ids; look-alike cues per sighting on a pool
       linker   each id once, when it ends (event-time watermark): kept rules, part rule, its row (colour, size,
                best view), its place memory, then an a-contrario link against EVERY earlier id
  -> objects.obt (OBT1): one row per kept id, edges to its closest earlier continuations (<= 4), ~23-35 B an id
  -> query side: ids in view at t; chains at closeness .9 (same object at the persistent level) + look-alikes;
     "find this object" (bottleneck closeness); continuity-only families; the viewer
  -> offline, on request: the 3D of one family (LingBot-Map + fusion + ShapeR), never stored in the index
```

---

## 3. The object tracker in detail

### 3.1 Segmenter (class-agnostic)
- **FastSAM-s** (an Ultralytics YOLOv8-seg model trained on SA-1B, so not bound to a vocabulary), exported to ONNX at
  `data/segment/o55/FastSAM-s.onnx` (45 MB).
  - It runs through ONNX Runtime with the CoreML provider (CPU + Neural Engine); NMS comes from `ultralytics.utils`.
  - The input is fixed at 640 x 640, which is why every frame is fitted first (`python/elidedb/sidefit.py`).
- **Cost:** 60.9 ms a frame p50 on the Neural Engine, masks included; 0.30 s per video second if run on every 5 fps
  frame [proven: O-55]. The motion-bounded schedule runs it only when needed (below).
- **A still-image segmenter re-cuts the scene every frame:** only 53-80% of segments continue into the next frame
  [proven: O-55]. So stability comes from the tracker, not the segmenter.
- **Cuts** (`objstream.cuts`, `one_cut`, `edge_share`):
  - confidence >= .1;
  - a cut with IoU > .5 to a better-scored kept one is the same cut;
  - each cut gets its share of boundary on the view's edge (the lens rim or the frame edge).

### 3.2 Tracking (what makes an id continuous)
Constants are at the top of `objstream.py`; each comes from a ledger row (O-55..O-67).
- **Schedule:** DIS flow at 320 px both ways (~0.5 ms). FastSAM runs when the median flow since the last segmented
  frame passes `TRIG` = 20 px, or every `GAP_MAX` = 3 frames.
- **Carry:** each track's mask is carried by a **per-pixel** flow warp. The carry holds while at least `CONS` = .5 of
  its warped pixels pass a per-pixel forward-backward check (`FB_FAIL` = 10 px at 320; Median Flow, Kalal 2010).
  - A single rigid motion per track ended every deforming bag [proven: O-66].
- **Match:** carried tracks against new cuts at IoU >= `MATCH` = .5, greedy one-to-one by IoU. A track may wait
  `WAIT` = 2 segmented frames unmatched. Unmatched cuts are births.
- **Kept rules** (`eligible`), all three required:
  - in view >= `MIN_SPAN` = 5 frames (1 s at 5 fps);
  - the **track's mean** FastSAM score >= `TRACK_CONF` = .4. Per-frame scores swing .10-.60 on a handled bag; scoring
    the track is the video-instance-segmentation practice [proven: O-66];
  - the median view-edge share < `VIEW_CUT` = .03. Walls, floor, ceiling, door strips and arms run along the lens rim.
- **Part rule:** a kept id that lies >= 80% inside a larger kept id, on >= 2 frames, is dropped as a part. This keeps
  one granularity per region.
- **The lit area** is wherever the running maximum of the grey frame exceeds 20. For a fisheye this is the lens disc;
  for a normal camera it is the whole frame. Nothing here is specific to a camera.
- **What failed for "noise"** [proven: O-55..O-66, by the owner's eye]: per-cut gates (boundary connectivity, SAM
  stability, boundedness) dropped real bags and background objects. Withholding ids leaves frames empty after head
  turns, which also reads as a failure. Track-level rules fixed it.
- **The owner's look:** "good" means clean outlines only where the segmenter cut on that frame, few layers, and stable
  ids. A mask is never drawn more than about 2 frames from where the segmenter cut it.

### 3.3 One row per id (what the index keeps)
For each kept id the linker writes:
- `t0`, `t1`: first and last segmented frame;
- the best view: the frame and box of the largest cut, used to re-derive the crop when someone looks;
- **colour on entry and on exit**: the mask's CIE-Lab mean and standard deviation over the first and second half of
  its sightings, 6 bytes each (one Gaussian, after Rubner's colour signatures);
- **size on entry and on exit**: 16 log2(pixels at 160 px), 1 byte each;
- its edges (3.4).

### 3.4 The edges: a-contrario closeness (`python/elidedb/objtree.py`, `link`)
A later id b is judged against every earlier id a that ended before b began.

**Cues**, each with its own null of "different object", measured on **pairs of ids on screen together** (provably
different), as the stream has seen them so far:
- **colour:** the diagonal 2-Wasserstein distance between a's exit Gaussian and b's entry Gaussian (Euclidean on the 6
  numbers); lower tail;
- **size:** |log2 exit size - log2 entry size|; lower tail;
- **position** (where present): a's mask carried as a ghost to b's birth frame, IoU; upper tail;
- **place** (3.6): where present, it replaces the ghost.

**Decision.** The cues are combined by **Fisher's method** into p. Then **NFA = p x the number of earlier ids**, the
expected number of different objects this close by chance. An edge is kept only if NFA <= `TAU` (~0.89). It is stored
as an evidence byte e = round(-10 log10 NFA), in decibans, and closeness = 1 - NFA.
- Each id keeps at most `K_EDGES` = 4 edges, best first; the first is its parent.
- The tree's persistent-object level is **closeness >= .9** (NFA <= .1).

**Exact and sub-quadratic.** NFA <= TAU needs the colour p under a bound set by the candidate count. So candidates come
from a radius query on a k-d forest of exit colours (`colour_bound`, `KDForest`, `StreamLinker`), and the result equals
the exhaustive n² search [proven: O-67b, O-126, tests/test_objtree.py]. Never let the build go back to all pairs: n² is
about 30 GB at 1 h.

**Lessons:**
- **Every cue's null must be the same statistic as its test.** A size null on raw areas was 2.5x overconfident
  [proven: O-67b].
- **Fisher lets one strong cue outvote the rest.** The ghost cue alone carried 13 of 29 wrong parents
  [proven: O-68d/e].

### 3.5 The same-object side: the look-alike veto (`python/elidedb/lookalike.py`, O-71)
The a-contrario test asks "rarer than different objects?". That passes two dark appliances that are more alike than
most different objects, yet differ more than any object ever differs from itself. The other side comes from
continuity, with no labels: one id's entry half against its own exit half is the variation of one object.
- **Six cues per sighting**, from the 640 px crop (box + 3 px), ~2 ms each:
  - Lab mean and std (2-Wasserstein);
  - a 4 x 6 x 6 joint Lab histogram (intersection);
  - a chroma-weighted hue histogram;
  - uniform LBP at radius 1 and 2 (chi-square);
  - Hu moments plus solidity, elongation and compactness;
  - log2 size.
- **Veto:** an edge is *different* if any cue exceeds the 1 - .01/6 quantile of the stream's own entry-vs-exit
  distances (the union-intersection test, Bonferroni; Fellegi-Sunter's m side).
- **Result:** parent precision went from .679 to .775 on dev and from .700 to .894 on the test window, with no right
  parent lost; it costs 0.083 s per video second [proven: O-71].
- **Dropped cues:** ORB is not a cue (at 640 px, 86% of ids never match even themselves), and silhouette shape is weak
  across angles [proven: O-70, O-68].

### 3.6 Place memory: where an object was last seen (`place.py`, `anchorplace.py`, `space3d.py`)
Identity needs space as well as time: an out-of-view object is where it was last seen until something moves it.
- **2D place** (`place.py`, O-73/O-74):
  - a's **surroundings** (ORB in its box grown by its own size, outside its cut, so the cue does not repeat colour or
    size) are registered into b's birth frame with a RANSAC homography;
  - the homography is kept only if it is meaningful a contrario (ORSA NFA < 1, Moisan-Moulon-Monasse 2012);
  - a's cut is then warped as a ghost into b's frame.
  - It is positive evidence only: no registration means no evidence, and place never vetoes (objects move).
  - Precision went to dev .823 and test .933, at 0.02-0.03 s per video second [proven: O-73, O-74].
- **Anchor same-spot test** (`anchorplace.py`, O-108/O-109), from two frames alone:
  - straighten both frames to a 90° pinhole (the lens model, 4.2);
  - SIFT anchors with the object's own cut left out;
  - an essential matrix by **MAGSAC++**. Below 1° of parallax (a head turn), the anchors' homography carries the
    object's ray. Otherwise the inliers are triangulated, and each frame's Depth Anything V2 Small disparity is aligned
    to them (1/z = aD + b), which moves a's object point into b;
  - the residual is compared against ids on screen with a.
  - It costs 0.21 s per video second with MAGSAC++ (0.68 with plain RANSAC). Depth Anything V2 Small runs in 16.6 ms a
    518 px frame [proven: O-108, O-109].
- **3D place from LingBot-Map** (`space3d.py`, O-78..O-82): see 4.3.
- **In production:** the 2D place where its surroundings register, otherwise the anchor or 3D place
  (`space3d.PlaceEither`). On the three labelled windows: 166 right / 25 wrong against 153 / 36 before [proven: O-82].

### 3.7 Continuous ingest and memory
- **One continuous stream:** every id stays a candidate for every later id, and state carries across pieces. Pieces are
  only transport: the same frames pushed as pieces or as one file give **byte-identical** rows and index
  [proven: O-123]. The continuous pass equals the batch pass on the 300 s by-eye truth: 102 of 104 truth ids, 0 mixed
  chains [proven: O-124].
- **Memory stays flat:** link memory that is written once and read rarely (last cuts, place contexts, anchor memory) is
  spilled to an append-only file with a RAM LRU (`spill.py`). RAM is flat at ~3.3 GB from minute 5 to 21 of a 22-min
  recording [proven: O-128].
- **Readers in separate processes:** in base v1 the object engine runs in its own process, fed frames over a pipe
  (`procreader.py`). In one process, every MPS call shares one command queue and the object engine waited behind the
  other models.
- **Speed:** base v1 reads the world model and the object tree together at ~3.5x real time on 30 min [proven: BV-11];
  the earlier joint stream ran at 2.65x [proven: O-123].

### 3.8 Storage: the OBT1 format (`objtree.py` header)
- **Columns:** varint t0 deltas, durations, view offset; view box as uint8; colour in and out as uint8 (out stored as a
  delta from in); size bytes; degree; edge back-deltas as varints; evidence bytes. Each column is raw or zlib, whichever
  is smaller. A JSON footer sits at the end, then the footer length and the magic `OBT1` (Parquet-style: one pass to
  write, one seek to read).
- **Size:** 23-35 B an id, 0.644 MB per video hour, **0.074% of the 640 px video** [proven: O-67b]. The cues and place
  registrations decide which edges exist and are not stored.
- **Store-wide map** (`objmap.py`, BV-12): every part's tree laid end to end as one tree.
  - Ingest edges stay sealed. Merge edges join parts on colour and size, judged against the earlier parts' rows.
  - It is append-only.
  - All of P01 (59,394 ids, ~5 h): 1.04 MB, built in 0.98 s [proven: BV-12].

### 3.9 Query side
- **Ids in view** over any time range: a binary search on the sorted t0 plus the maximum duration
  (`Tree.in_range`), so it is sublinear.
- **An object at the persistent level:** union-find components over edges at closeness >= .9
  (`objsearch.ObjectFilter`). Look-alikes are ids one edge away at any closeness.
- **Find this object:** every id joined to the chosen one, ranked by **bottleneck closeness** (the best path's
  weakest edge, i.e. the single-linkage level at which they join), with a widest-path search that stops after k
  results (`ObjectFilter.related`). At or above .9 it is the same object at the persistent level; below that it is a
  look-alike.
- **Families by continuity only** (`scripts/o108_families.py`): ids joined by the same track, or by an edge at >= .9
  whose continuity cue (place, anchor or ghost) is meaningful **on its own**, so colour and size cannot carry it.
  - Each family's view bank is picked farthest-first in units of one object's own variation.
  - On the 300 s window: 52 families over 14 real objects, **0 mixed** [proven: O-109b].
- **The viewer** (`scripts/object_viewer.py`, `:8771`, launch config `object-viewer`):
  - ids through time with outlines re-derived from the stored data;
  - an id's continuations;
  - find by image;
  - the Families tab;
  - the 3D panel (Build 3D / Open 3D, three.js).
  - Data lives under `data/objects_viewer_o67_held/<recording@window>/`.

---

## 4. Space-time place and 3D mapping

### 4.1 LingBot-Map (the streaming 3D model)
- **What it is:** Robbyant/lingbot-map (Apache-2.0, ECCV 2026, arXiv 2604.14141). A VGGT-style model (DINOv2 ViT-L
  plus 24 frame and global blocks, ~1.2 B parameters) that gives every streamed frame a camera pose, a depth map and a
  confidence.
- **Weights:** `data/lingbot/checkpoints` (lingbot-map.pt, 4.63 GB). The code is in `data/lingbot/lingbot-map`.
- **Convention:** the streamed `pose_enc` is **camera-to-world** (translation, quaternion xyzw, fov_h, fov_w). Reading
  it as world-to-camera produced two false failures and a wrong RCA [proven: O-78..O-80]. **Check any geometry
  convention by reprojection first:** lift one frame's depth into the next frame and see that it agrees.
- **Upstream on MPS** (`scripts/lingbot_upstream.py`), with these changes, all exact:
  - the RoPE complex128 table stays on the CPU and each slice goes to MPS as complex64;
  - `PYTORCH_ENABLE_MPS_FALLBACK=1` for the bicubic resize;
  - attention runs in query chunks;
  - the key cache is bf16;
  - the GPU is synced after each layer.
  - Cost: 2.4-2.7 s a 518 px frame, 34 GB [proven: O-102..O-107].
- **The MLX port** (anmolduainter/lingbot-map-mlx, fp16; `scripts/lingbot_stream.py`, with ring-KV and compile
  patches in `o77_ringkv.py` and `o77_fast.py`):
  - real time only at 378 px (3.3 fps, 4 frames a call); 8/4-bit quantization saves memory, not time
    [proven: O-77];
  - **it is not faithful**: depth self-consistency is worse than upstream's on the same frames (.009 / .025 / .049 vs
    .006 / .010 / .014 at gaps 1 / 5 / 10) [proven: O-100];
  - every LingBot number from O-77 to O-99 came through the port.
- **Neither checkpoint has point-head weights:** use depth plus camera. Upstream's viewer draws only depth_conf > 1.5.

### 4.2 The lens (HD-EPIC specific: Aria Gen 1 RGB)
- **Model:** a circular **equidistant** fisheye, 110° field of view. On 640 px copies, f = 335 px; the image circle has
  radius 360 px (`fisheye.py`, `space3d.CIRCLE`). The focal length was checked by the plumb-line rule.
- **Why rectify:** feed-forward reconstructors assume a pinhole. Read raw, the fisheye looks like a 73° pinhole and the
  model under-rotates by 24%. Rectified to a 90° pinhole, rotation is right (1.006 of the fisheye-match rotation)
  [proven: O-103].
- **For a new project with a normal camera**, `fisheye.py`, `CIRCLE` and the straightening in `anchorplace.Straight`
  become the identity or that camera's model. This is the main camera-specific code.

### 4.3 What a global map from video alone can and cannot do
- **Fidelity measure without ground truth** (`mapcheck.py`; Luo et al. SIGGRAPH 2020, Schonberger ECCV 2016): lift
  frame i's depth with its camera into frame j and compare.
  - Geometric: does j's own depth agree?
  - Photometric: do the points land on the same content, against the identity warp?
  - Point-to-plane thickness of a flat surface.
  - It has synthetic tests, and the measure has no floor problem (synthetic clutter scores 0.000) [proven: O-104].
- **The kitchen's limit is depth under head turns.** There is little parallax, so each view's depth is its own guess.
  The same surface disagrees by 3% at 0.2 s and 9% at 5 s. No pose fixes it; averaging reaches ~4% at best, against a
  3% bar. Upstream's own demos walk forward and score 0.6-1.4% [proven: O-102..O-107].
- **Loop closure** (`loopmap.py`, after VGGT-Long):
  - Sim(3) registrations between revisited views, kept only if meaningful a contrario. The p-value of a match is its
    rank among b's matched rays, because the solid-angle and ray-pair nulls passed random match sets in tests;
  - a Levenberg-Marquardt pose graph with Cauchy down-weighting.
  - Drift of a static object over 60 s fell from .54 to .24, and the coefficient of variation of static distances from
    .167 to .101. It costs ~0.2 s per video second [proven: O-87].
- **Owner's verdict on the fused map (10-08 10:24): not accurate.** Per-frame depths fused with the stream's poses give
  smeared volumes, not surfaces. **Measure fidelity before showing any map.**
- **Published evidence** (`docs/research/2026-10-08-space-time-object-place.md`):
  - video-only SLAM breaks on egocentric motion (LaMAria, ICCV 2025; EPIC Fields: ORB-SLAM tracks 44% of frames;
    COLMAP works offline at ~23x real time);
  - accurate real-time object maps all use a pose sensor plus depth (Khronos, Lost & Found, OSNOM);
  - the newest video-only HD-EPIC reconstruction is offline (arXiv 2603.22450, ~1.6 s per video second on an
    RTX 5090).
- **What does work from video alone:** relative, anchor-based geometry (R4DSG 2026; relative bundle adjustment, RSS
  2009). An object's place is stored relative to stable anchors seen in the same frame, so the frame's scale and pose
  errors cancel. That is why the anchor test (3.6) replaced the global 3D place [assumed that the error cancels; the
  anchor-relative spread over time is not yet measured].

### 4.4 What 3D is for (owner, 10-08 ~12:20)
"My objective is not to build a complete 3d map. but to know the space and time relations of an object to solve for
the look-alikes issue and deformables." 3D must deliver two tests:
1. **same place across a change of view:** the object did not move, so a sighting from another side or after a
   deformation is the same object;
2. **a carry path:** the object moved in hand from A to B. **Not built.** The contact wall is parked; see 6.

---

## 5. Offline 3D rebuild of one object (BETA)

This is built on request from the viewer, never at ingest, and never stored in the index. Entry points:
`scripts/object3d_build.py` and `POST /api/build3d` on the viewer.

**Pipeline** (`python/elidedb/object3d.py`):
1. **Frames:** up to 24 of the family's sightings, the most different views first, straightened to a 90° pinhole.
2. **Geometry:** upstream LingBot-Map on those frames, in time order: pose, depth and confidence per frame.
3. **Object:** each frame's own cut (the ingest's mask), carried onto the straightened frame and the model's grid.
4. **Align:** trimmed ICP of each frame's object points to the others, so camera drift between distant views does not
   smear the object (object-level fusion as in Co-Fusion / MaskFusion).
5. **Surface:** TSDF fusion (Curless & Levoy), with free space carved outside the mask (visual hull), then marching
   cubes.
6. **Up direction:** a RANSAC support plane **under** the object. The cameras' up is wrong by 48-52° when the head
   looks down [proven: O-111].
7. **Whole shape (unseen faces):** ShapeR (Meta, CVPR 2026, arXiv 2601.11514, **CC BY-NC 4.0**, 4.18 GB at
   `data/shaper/ShapeR`), ported to MPS in `python/elidedb/shaper_mac.py`:
   - torchsparse convolutions re-implemented from its own Python, SDPA attention, no compile;
   - **text is never given** (the null token).
   - Cost: 47-50 s an object; Chamfer x100 on its own objects 0.88-2.65 [proven: O-118].
   - Each surface point is tagged observed or inferred, and never-seen faces are hatched.

**Cost:** 70-76 s an object, of which 14-15 s is model loading; peak ~10-15 GB [proven: O-121].

**Quality:** on held-out views, ShapeR's IoU against fusion's is fryer .761 vs .758, bag .764 vs .760, board .617 vs
.672. Held-out views cannot score the unseen faces [proven: O-121].

**Physics** (`softsim.py`, MuJoCo 3.3.1 flex):
- The cube sag is validated against ρgH²/2E within 2% [proven: O-111].
- **Stiffness is not identifiable from this video.** The presets (soft 10 kPa, firm 30 kPa, rigid) are assumptions,
  labelled in the UI.
- The owner judged MuJoCo too basic, so `--press` is opt-in.
- The state of the art (PhysTwin, EgoPhys) identifies spring-mass parameters from the handling. Not built. An object
  never seen deforming gets a bound, not a material.

**Owner, 10-08 17:20:** BETA, "close to a minute is fine", no more quality or time work unless asked.

---

## 6. What failed or is parked (do not retry without a new mechanism)

| tried | outcome | ledger |
|---|---|---|
| DINOv3 masked-crop cosine for identity | weak (AUC .76; .52 on look-alikes); "cosine never works" | O-3..O-6, O-15 |
| EdgeTAM / SAM 2-family video tracking + OWLv2 boxes for discovery | better OWTA (.167 -> .315), but OWLv2 is a vocabulary detector and dense tracking cost 4-6.5 min a video minute | O-12..O-27 |
| Getting through hand contact (SAMURAI, clean memory, SAM 2.1-B+, re-detection, last box) | 1-11 of 72 recovered; parked as "the contact wall" | O-22p..O-26p |
| SAM 3 | 1.5-3 s per video second (7-20x the budget): stopped before any accuracy number | O-56 |
| Per-cut noise gates (connectivity, stability, boundedness) | dropped real bags and background objects | O-57..O-63 |
| 3D place everywhere instead of 2D-else-3D | worse in clutter | O-81 |
| A tree on the loop-closed map | missed its precision bar by .006 | O-88 |
| A place gate (two test families) for NFA growth | no gain: the cues limit, not the test count | O-97 |
| Cannot-link between ids seen together | precision .952, but the persistent-object bar failed; one drifting id merges two objects | O-94, O-95 |
| The fused global 3D map | smeared, not accurate (owner verdict) | O-98..O-107 |
| The MLX LingBot port as a geometry source | unfaithful (drifts ~3x upstream) | O-100 |
| The object tree as a search scope or ranker (ElideDB) | scope covers only 44-57% of right answers; rank P@1 .338 vs words .358 (not significant) | BV-1, BV-12 |

---

## 7. Pitfalls learned the hard way

- **Check geometry conventions by reprojection** before any downstream number (camera-to-world vs world-to-camera).
- **Validate any registration null on random matches first.** O-86's volume null and two other nulls passed random sets.
- **Check stored intermediates against production** before trusting a study built on them. A debug dump rebuilt masks
  without the box crop, which voided a day of cue tables.
- **Gravity is not the camera's up** on head-worn video. Use the support plane under the object.
- **FastSAM's ONNX input is fixed at 640 x 640.** Fit frames (aspect kept, pads) before the segmenter, and map boxes
  back (`sidefit.side_map`).
- **Phone video is rotated by metadata.** Decode the first frame's `rotation` and rotate the frames.
- **One MPS command queue per process:** readers that share a process wait on each other. Use processes.
- **Do not time runs while the owner uses the Mac:** a game slowed FastSAM 5x.
- **Closeness rounds to 1.0** in float64 above ~160 decibans. Copy evidence bytes, not closeness
  (`objtree.write(edge_bytes=True)`).
- **The NFA count grows with history.** The same evidence that is meaningful early in a stream stops being so late.
  This is a property of the test, and re-judging sealed data with a larger count makes identity fade as the store
  grows (BV-12 design note).

---

## 8. What to copy into a new project

**Code** (from `python/elidedb/`):

| module | role | depends on |
|---|---|---|
| `objstream.py` | the engine (4 stages, rules, linker) | objtree, lookalike, place, space3d, anchorplace (optional), spill, sidefit; cv2, onnxruntime, ultralytics (NMS), torch |
| `objtree.py` | OBT1 format, a-contrario `link`, nulls, k-d forest, `StreamLinker`, `Tree` lookups | numpy, scipy |
| `lookalike.py` | the six same-object cues and the veto | cv2, scikit-image |
| `place.py` | 2D place memory (ORB, ORSA) | cv2 |
| `anchorplace.py` | anchor same-spot test (SIFT, MAGSAC++, Depth Anything V2 Small) | cv2, torch, transformers, fisheye |
| `space3d.py` | LingBot cameras to world points; `Place3DCue`, `PlaceEither` | fisheye |
| `fisheye.py` | equidistant lens, rectify to pinhole | cv2 |
| `loopmap.py` | loop closure for a streamed map | scipy, cv2 |
| `mapcheck.py` | map fidelity without ground truth | numpy |
| `object3d.py` | offline object 3D: ICP, TSDF + carving, support plane | trimesh, scikit-image |
| `shaper_mac.py` | ShapeR on MPS | torch, ShapeR's code |
| `softsim.py` | MuJoCo press (opt-in demo) | mujoco 3.3.1 |
| `spill.py` | flat-RAM link memory | stdlib |
| `sidefit.py` | fit any frame to 640 x 640 | cv2 |
| `objsearch.py` | query side: ids in range, chains, find this object | objtree |
| `objmap.py` | many trees as one (append-only) | objtree |

**Scripts:** `objects_stream.py` (CLI for the engine), `object_viewer.py` (`:8771`), `o108_families.py`,
`object3d_build.py`, `lingbot_upstream.py`, `spacetime_o78_run.py`, `spacetime_view.py`.

**Tests:** `test_objtree.py`, `test_lookalike.py`, `test_place.py`, `test_anchorplace.py`, `test_loopmap.py`,
`test_mapcheck.py`, `test_obj3dqa.py`, `test_softsim.py`, `test_objsearch.py`, `test_objmap.py`,
`test_objstream_guard.py`.

**Weights** (all already on this Mac; licences matter for any product):

| model | where | size | licence |
|---|---|---|---|
| FastSAM-s ONNX | `data/segment/o55/FastSAM-s.onnx` | 45 MB | AGPL-3.0 (Ultralytics) [assumed: check before any product use] |
| Depth Anything V2 Small | HF cache `depth-anything/Depth-Anything-V2-Small-hf` | ~100 MB | Apache-2.0 |
| LingBot-Map | `data/lingbot/checkpoints` | 4.63 GB | Apache-2.0 |
| ShapeR | `data/shaper/ShapeR` | 4.18 GB | CC BY-NC 4.0 (non-commercial) |
| EdgeTAM, OWLv2 (history only, not in the engine) | HF cache | 56 MB, 620 MB | — |

**Python stack:** pyenv 3.11.6 with numpy 2.2.5, opencv 5.0, onnxruntime 1.28, torch 2.7.1, transformers 4.57.6,
scikit-image 0.26, scipy 1.15.3, trimesh 5.0, mujoco 3.3.1, ultralytics 8.4.113, PyAV 18.0.

**What is specific to this data:**
- the fisheye lens (4.2);
- the 5 fps grid;
- the 640 px side frame;
- the dev windows used for labels (P01-20240203-121517@120-420 and @0-473, 130505@552-582, 184214@794-824,
  121042@166-196).

Everything else (the rules, the nulls, the format) is camera- and domain-general by construction: it has no classes,
no labels and no kitchen premises.

---

## 9. Open items the owner had not decided

- A **wide-baseline matcher** for the anchor test (XFeat or LightGlue; a download needs the owner's yes). SIFT fails
  on a step forward or a big turn, and 40 of 89 true returns never reach a place test because of the colour gate
  [proven: O-109].
- The **carry path** (continuity through the hand): parked behind the contact wall.
- **Device poses** (Aria MPS) as allowed input: the owner's rule is video only.
- Keeping models resident to cut the 3D build's 14-15 s of loading.
- Exact identicals stay look-alikes until a space-time memory can tell them apart.

## 10. Where the full history is

| what | where |
|---|---|
| ledger rows O-1..O-128 and BV-1..BV-12 | `docs/memory/ledger.md` |
| object persistence research | `docs/research/2026-10-06-object-persistence.md` |
| the external index | `docs/research/2026-10-07-external-object-index.md` |
| look-alikes, place memory, LingBot (§6b, §6c) | `docs/research/2026-10-07-lookalikes.md` |
| space-time place from video only (§6, §7) | `docs/research/2026-10-08-space-time-object-place.md` |
| unseen faces and physics, state of the art | `docs/research/2026-10-08-unseen-faces-and-physics-sota.md` |
| 3D quality plan | `docs/research/2026-10-08-object-3d-quality.md` |
| the store-wide map | `docs/research/2026-10-09-object-map-index-query.md` |
