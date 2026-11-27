"""Arm C's marks: one Claude call on the first scene frame returns two pixels, and two rings are
drawn at those pixels on every scene frame.

SHARED MODULE. The demonstrations (scripts/mark_demos.py, scripts/render_marked.py) and the scored
trials (arm C's and C-ACT's inference) call exactly these functions, with the same model, prompt,
image encoding, retry rule and ring drawing. Nothing in here may look at layouts.json: the demos
get no help the trials do not get.

Images are RGB uint8 arrays, 480 x 640 (LeRobot's cameras and video decoder both give RGB).
Credentials as arm A: ANTHROPIC_API_KEY in the environment or ~/.config/who-moves-the-arm/.env.
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np

MODEL = "claude-opus-5-5"
PROMPT_FILE = Path(__file__).parent / "mark_prompt_v1.md"
JPEG_QUALITY = 85
MAX_ATTEMPTS = 3        # an API error or an invalid answer is retried; after 3, no marks
W, H = 640, 480

# Rings, fixed here so the demos and the trials draw them identically.
GRASP_RGB = (255, 0, 255)      # magenta: the object to pick up
RELEASE_RGB = (0, 110, 255)    # blue: the container to put it in
GRASP_RADIUS = 22              # px at 640 x 480
RELEASE_RADIUS = 34
THICKNESS = 3
LINE_TYPE = cv2.LINE_8         # not anti-aliased: the same pixels every time

KEY_FILE = Path.home() / ".config" / "who-moves-the-arm" / ".env"  # outside the repo

TOOL = {
    "name": "mark",
    "description": "Mark the two points in the image. Pixels: u is the column (0-639, left to right), "
                   "v the row (0-479, top to bottom).",
    "input_schema": {"type": "object", "properties": {
        "grasp_u": {"type": "number"}, "grasp_v": {"type": "number"},
        "release_u": {"type": "number"}, "release_v": {"type": "number"}},
        "required": ["grasp_u", "grasp_v", "release_u", "release_v"]},
}


class NoMarks(RuntimeError):
    """No valid marks after MAX_ATTEMPTS. A demo without marks is dropped; a trial without marks
    is scored as a failure."""


def load_key_file(path=KEY_FILE):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def client():
    import anthropic
    load_key_file()
    return anthropic.Anthropic()


def jpeg_b64(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def _valid(m):
    try:
        vals = [float(m[k]) for k in ("grasp_u", "grasp_v", "release_u", "release_v")]
    except (KeyError, TypeError, ValueError):
        return None
    gu, gv, ru, rv = vals
    if not all(np.isfinite(vals)) or not (0 <= gu < W and 0 <= ru < W and 0 <= gv < H and 0 <= rv < H):
        return None
    return {"grasp": [gu, gv], "release": [ru, rv]}


def get_marks(rgb, instruction, api=None):
    """Ask Claude for the grasp and release pixels on this frame. Returns
    {"grasp": [u, v], "release": [u, v], "attempts": n, "latency_s": s, "usage": {...},
     "raw": [...answers...], "model": MODEL, "prompt": "mark_prompt_v1.md"}; raises NoMarks."""
    assert rgb.shape == (H, W, 3), f"expected a {H}x{W} RGB frame, got {rgb.shape}"
    api = api or client()
    system = PROMPT_FILE.read_text()
    content = [{"type": "text", "text": f"Instruction: {instruction}"},
               {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64(rgb)}}]
    raw, usage = [], {"input_tokens": 0, "output_tokens": 0}
    t0 = time.time()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            # Opus 5.5 does not accept a forced tool_choice; the prompt asks for the tool instead
            r = api.messages.create(model=MODEL, max_tokens=2048, system=system, tools=[TOOL],
                                    messages=[{"role": "user", "content": content}])
        except Exception as e:  # network or API error: counts as an attempt
            raw.append({"error": repr(e)[:300]})
            time.sleep(2)
            continue
        usage["input_tokens"] += r.usage.input_tokens
        usage["output_tokens"] += r.usage.output_tokens
        m = next((b.input for b in r.content if b.type == "tool_use"), None)
        raw.append(m if m is not None else
                   {"no_tool_call": " ".join(getattr(b, "text", "") for b in r.content)[:300], "stop": r.stop_reason})
        marks = _valid(m) if m is not None else None
        if marks:
            return {**marks, "attempts": attempt, "latency_s": round(time.time() - t0, 2), "usage": usage,
                    "raw": raw, "model": MODEL, "prompt": PROMPT_FILE.name}
    raise NoMarks(f"no valid marks after {MAX_ATTEMPTS} attempts: {raw}")


def draw(rgb, marks):
    """The frame with both rings drawn on a copy. Same call for every frame of a demo or trial."""
    out = np.ascontiguousarray(rgb).copy()
    gu, gv = (int(round(x)) for x in marks["grasp"])
    ru, rv = (int(round(x)) for x in marks["release"])
    cv2.circle(out, (ru, rv), RELEASE_RADIUS, RELEASE_RGB, THICKNESS, LINE_TYPE)
    cv2.circle(out, (gu, gv), GRASP_RADIUS, GRASP_RGB, THICKNESS, LINE_TYPE)
    return out
