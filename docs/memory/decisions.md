# Decisions

2026-10-09 - No parts library: reconstruct every part from the video
Chose reconstructing every part from video alone over LDraw/CAD library matching (offered as library-first with fallback). The user wants it fully general; keeps the ElideDB no-vocabulary rule. LEGO's known dimensions may still be used to grade accuracy, not as input.

2026-10-09 - Guide output is an interactive three.js web viewer
Chose an interactive 3D web viewer (step scrubbing, part fly-in animation, orbit, per-step parts list) over a rendered MP4 or static PDF pages; a video can be rendered from the viewer later.

2026-10-09 - Mac only compute
Everything runs on the M5 Pro (MPS/CoreML); no rented CUDA GPU. CUDA-only methods must be ported (as ShapeR was) or replaced.

2026-10-09 - Light filming routine allowed
The builder shows each new piece for 1-2 s before placing it and circles the model at the start (loose parts) and the end. The pipeline must still degrade gracefully when the routine is skipped.

2026-10-09 - Hands by learned skin chroma, not MediaPipe
MediaPipe hand landmarker read the studded round plate as a palm and missed the close-up partial hands; a per-video (a*,b*) histogram ratio of moving (flow residual) vs still pixels finds hands cleanly and keeps cream/grey out.

2026-10-09 - Finished state explained by 3D continuity before appearance
Look-alike parts fuse into one cut in the finished build (flag + bar); the last joined group (ICP onto finished-state points) claims its pixels first, the remainder goes to new parts by appearance.

2026-10-09 - Daylighting pivot: VAST-first, no fallbacks
Owner: use first-person biking video from the VAST index, no PIE fallback, no fallback paths in the design; the theme is to use VAST (VSS search, YOLO sidecars, Cosmos Reason/Embed, W&B inference, VastDB). Local models only where VAST has no equivalent (the 3D map).

2026-10-09 - Ground model for the bike ride: fixed camera mount
Chose one road plane fixed in the camera's frame (pooled RANSAC over all frames, central lower region) + per-frame camera height over a global world plane or per-frame planes: LingBot poses drift (global plane rolled ~20 deg) and single-frame depth planes had the wrong pitch; the pooled plane's horizon matches the road's vanishing point by eye.

2026-10-09 - "Standing" not "parked" in the ride viewer
A car that does not move during one pass may be queued at a light (this ride runs beside a busy street with no curb parking). The UI says "standing"; repeated rides of the same street (VAST index) are the way to call a car parked.

2026-10-09 - Ride service: VM pulls, Mac serves (quick Cloudflare tunnel)
The Mac cannot reach the VM (internal DNS fails, public origin 403/1010), so the VM always initiates: chunked clip upload (Cloudflare caps requests at 100 MB) and hash-checked pull sync into a mirror with the Mac's runs/ layout. Quick tunnel chosen by the owner (no account; URL changes per restart). Pipeline runs as a subprocess per run so GPU memory is freed.

2026-10-09 - Daylighting = approach-side sightlines, not a 20 ft ring
Owner correction: daylighting is about occlusion of oncoming traffic by parked vehicles and street furniture. CA AB 413 / NACTO: no stopping within 20 ft on the vehicle approach side of a crosswalk so a pedestrian at the curb and an approaching driver can see each other. Model: per crosswalk and approach direction, sight triangle from the curb waiting point up the approach lane; occluders = anything > ~0.9 m in it (3D height map + named detections).

2026-10-09 - Occluders from detections, not the depth height map
LingBot depth on this footage flattens ~1 m objects (hedge planters, bollards come out ~0.4 m), so a height-map occluder test misses them. Occluders are detected standing vehicles + furniture with a footprint from the box's bottom corners cast onto the road and a height from the box's top edge at that range.

2026-10-09 - Viewer as a one-screen "sightline cockpit"
Owner asked for a much cooler viewer and approved the cockpit design: the 3D street fills the screen and a director camera tells the story (fly-in, chase, slowed crossing inspection from behind the oncoming traffic, with STOP / SEEN distances painted on the lane). The camera video becomes a PiP, and the detailed list moves below the fold. Chose this over a split-panel dashboard because the demo story (a hedge hides the curb until 5 m, while stopping needs 9 m) reads best in one view.

2026-10-09 - W&B by CoreWeave: a vision model checks the findings; Weave traces every run
The hackathon's W&B offer is serverless inference. Owner picked a vision check plus Weave traces over a W&B metrics dashboard. The check targets the false findings the frames exposed (cars on another roadway, misplaced occluders): one Qwen3.8-27B call per exposed crossing end, with two outlined frames. A missing key fails the run (no silent skip).

2026-10-09 - Daylighting measured from the rider's view, read off the video
Owner: the biker is the subject. Per crossing, each side (a 3 m x 3.5 m patch where someone about to cross stands, 1 m tall) is cast into every frame on the way in and compared with the rebuilt depth (Street.sees): seen or covered by something nearer. Verdict at the stopping distance (1.5 s + 3 m/s^2): hidden <50%, partly <90%, in view >=90%. Area, not a single point (a point flipped with 1 m of placement). Judged only within 10.5 m (open road ahead reads 94-96% seen at 10 m, ~70% at 13 m) and not while turning >35 deg. Replaces the pedestrian waiting-point sightlines and the "car in the 20 ft" claims.

2026-10-09 - Camera roll forced to zero in Street.mount
The pooled road plane leaned ~9 deg; drawn on the frames, the model's horizon tilted and its verticals leaned while poles stand upright. A handlebar camera does not roll against the road, so the fitted normal's sideways part is dropped.
