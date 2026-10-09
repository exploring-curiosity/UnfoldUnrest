"""Review: a vision model on W&B Inference checks each exposed crossing end against the frames.

The 3D check finds, per crossing end, how far before the crossing the rider first sees it; the end is exposed when that
is later than the stopping distance. The rebuild can be wrong (a car on another roadway placed beside the bike path, a
hedge measured too short), so before an exposed end becomes a finding the model is shown two frames with the end's
watch area outlined: one at stopping distance, one where the 3D check says the area comes into view. It answers
whether the area is visible at stopping distance, what hides it, whether that stands between the rider and the area,
and whether the claim holds. Its answer goes into ride.json next to the 3D numbers (end["review"]); a crossing is
confirmed when one of its exposed ends is.

Needs WANDB_API_KEY and WANDB_PROJECT (<team>/<project>) in the environment (pipeline.py loads .env).
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import weave
from PIL import Image, ImageDraw

BASE_URL = "https://api.inference.wandb.ai/v1"
MODEL = "Qwen/Qwen3.8-27B"
WIDTH = 1024            # px: frames are sent at this width

SYSTEM = ("You check claims made by a 3D street analysis against camera frames from a bicycle. Judge only from what "
          "the images show, not from the claim. Answer with one JSON object and nothing else.")

ASK = """The camera rides toward a crossing at {speed} km/h; the rider needs about {need} m to stop.
The yellow outline marks the {side} end of the crossing, on the rider's {side}: where a person, another rider or a
car could come into the crossing.

Image 1: the rider is {d1} m before the crossing, the last moment to start stopping.
Image 2: the rider is {d2} m before the crossing.

The 3D analysis claims that in image 1 the outlined area is hidden from the rider{by}, and that {clear}.

Answer with this JSON:
{{"crossing_ahead": true or false (is there a marked crossing at the outline?),
 "area_visible_in_image_1": "yes" or "partly" or "no",
 "blocker": "what hides the area in image 1, in a few words" or null,
 "blocker_kind": "parked vehicle" or "vehicle waiting in traffic" or "moving vehicle" or "street furniture or planting" or "building or wall" or "nothing",
 "blocker_between": true or false (does it stand between the rider and the outlined area?),
 "blocker_named_right": true or false (is it what the analysis named?), or null if the analysis named nothing,
 "claim": "confirmed" if the outlined area is hidden or mostly hidden in image 1, "rejected" if it is in clear view, "unclear" if the images cannot tell,
 "reason": "one sentence a street designer can read"}}"""


def client():
    import openai
    return openai.OpenAI(base_url=BASE_URL, api_key=os.environ["WANDB_API_KEY"], project=os.environ["WANDB_PROJECT"])


def outlined(path, poly):
    """The frame with the watch area drawn on it, as a data URL."""
    im = Image.open(path).convert("RGB")
    s = WIDTH / im.width
    im = im.resize((WIDTH, round(im.height * s)))
    if poly:                                               # the area's corners (a box: on the road and at TARGET_H)
        hull = cv2.convexHull(np.asarray(poly, np.float32) * s)[:, 0]
        ImageDraw.Draw(im).polygon([tuple(q) for q in hull.tolist()], outline=(255, 210, 0), width=5)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def answer(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(t)


@weave.op
def check_end(cw_id: int, end: dict, by: str, frames_dir: str) -> dict:
    fr = {f["k"]: f for f in end["frames"]}
    near = min((f for f in end["frames"] if f["counted"]), key=lambda f: f["d"])
    k1, k2 = end["stop_k"], end["clear_k"] if end["clear_k"] is not None else near["k"]
    ask = ASK.format(speed=end["speed_kmh"], need=end["needed_m"], side=end["side"], d1=fr[k1]["d"], d2=fr[k2]["d"],
                     by=f" by {by}" if by else "",
                     clear=f"it only comes fully into view {end['clear_from_m']} m before the crossing" if end["clear_from_m"]
                     else "it never comes fully into view on the way in")
    msg = [{"role": "system", "content": SYSTEM},
           {"role": "user", "content": [{"type": "text", "text": ask},
                                        {"type": "image_url", "image_url": {"url": outlined(Path(frames_dir) / f"{k1:05d}.jpg", fr[k1]["poly"])}},
                                        {"type": "image_url", "image_url": {"url": outlined(Path(frames_dir) / f"{k2:05d}.jpg", fr[k2]["poly"])}}]}]
    r = client().chat.completions.create(model=MODEL, messages=msg, temperature=0, response_format={"type": "json_object"})
    out = answer(r.choices[0].message.content)
    return dict(out, model=MODEL, frames=[k1, k2])


def things(labels):
    if not labels:
        return ""
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


@weave.op
def run(out: str) -> dict:
    out = Path(out)
    p = out / "app" / "ride.json"
    data = json.loads(p.read_text())
    byid = {o["id"]: o for o in data["objects"]}
    checked = confirmed = rejected = 0
    for cw in [o for o in data["objects"] if o["group"] == "crosswalk"]:
        for end in cw["ends"]:
            if not end["exposed"]:
                continue
            by = things(list(dict.fromkeys(f"a {byid[b['id']]['label']}" for b in end["blockers"])))
            end["review"] = check_end(cw["id"], end, by, str(out / "frames"))
            checked += 1
        cw["confirmed"] = any(e.get("review", {}).get("claim") == "confirmed" for e in cw["ends"])
        confirmed += cw["confirmed"]
        rejected += any(e.get("review", {}).get("claim") == "rejected" for e in cw["ends"]) and not cw["confirmed"]
    data["stats"].update(checked=checked, confirmed=confirmed, rejected=rejected, review_model=MODEL)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, p)
    print(f"review: {checked} exposed ends checked by {MODEL}; {confirmed} crossings confirmed, {rejected} rejected", flush=True)
    return data["stats"]
