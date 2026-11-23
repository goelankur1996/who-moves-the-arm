"""Arm A: Claude drives the SO-101 through five tools.

Usage (arm at rest, leader not needed):
  python -m arm_a.agent --instruction "put the orange cube in the pink cup" \
      --trial-id pilot_01 --layout-id P01 --condition E1

Credentials: `ant auth login` (no key), or ANTHROPIC_API_KEY in the environment or ~/.config/who-moves-the-arm/.env. Writes runs/<trial-id>/: log.jsonl, images, video.mp4, summary.json,
and appends one row to the results CSV after you score the trial.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
from pathlib import Path

import threading

import anthropic
import cv2

from arm_a.rig import Recorder, Rig

MODEL = "claude-opus-5-5"
PRICE_PER_MTOK = {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0}
TIME_LIMIT_S = 120.0
MAX_TOOL_CALLS = 25
HERE = Path(__file__).parent
KEY_FILE = Path.home() / ".config" / "who-moves-the-arm" / ".env"  # outside the repo, never committed


def load_key_file(path=KEY_FILE):
    """Fill missing environment variables from KEY=value lines in a local file."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

TOOLS = [
    {"name": "look", "description": "Both camera images (scene, wrist) and the arm state.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "table_point", "description": "Convert a scene-image pixel on the table surface to table coordinates in mm.",
     "input_schema": {"type": "object", "properties": {
         "u": {"type": "number", "description": "pixel column, 0-639"},
         "v": {"type": "number", "description": "pixel row, 0-479"}}, "required": ["u", "v"]}},
    {"name": "move_to", "description": "Move the fingertip in a straight line to (x, y, height) mm, height above the table.",
     "input_schema": {"type": "object", "properties": {
         "x": {"type": "number"}, "y": {"type": "number"}, "height": {"type": "number"},
         "yaw_deg": {"type": "number", "description": "gripper rotation about vertical, default 0"}},
         "required": ["x", "y", "height"]}},
    {"name": "gripper", "description": "Open (fingertips about 80 mm apart) or close the gripper. After closing, reports whether something is held.",
     "input_schema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}},
    {"name": "home", "description": "Move to a safe hover position above the middle of the table.",
     "input_schema": {"type": "object", "properties": {}}},
]


class LiveView:
    """Rerun window with both cameras at 5 Hz, half size, JPEG-compressed, plus a text log of every step.
    It only reads the latest camera frames already in memory; it never touches the motor bus,
    so it cannot slow the arm's control."""

    def __init__(self, scene, wrist, hz=5.0):
        import rerun as rr
        self.rr, self.scene, self.wrist, self.hz = rr, scene, wrist, hz
        rr.init("who-moves-the-arm: arm A")
        rr.spawn(memory_limit="10%")
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        rr = self.rr
        while not self._stop:
            t = time.time()
            for name, cam in (("scene", self.scene), ("wrist", self.wrist)):
                img, _ = cam.latest()
                small = cv2.cvtColor(cv2.resize(img, (320, 240)), cv2.COLOR_BGR2RGB)
                rr.log(f"camera/{name}", rr.Image(small).compress(jpeg_quality=70), static=True)
            time.sleep(max(0.0, 1 / self.hz - (time.time() - t)))

    def note(self, text):
        self.rr.log("agent/log", self.rr.TextLog(text))

    def close(self):
        self._stop = True
        self.thread.join(timeout=2)


def mark_cache_point(messages):
    """Prompt caching for the whole conversation: one cache breakpoint on the newest user turn.
    Every earlier image and tool result is then read from cache at a twentieth of the input price
    instead of being dropped."""
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                if isinstance(b, dict):
                    b.pop("cache_control", None)
    last = messages[-1]["content"]
    if isinstance(last, list) and last and isinstance(last[-1], dict):
        last[-1]["cache_control"] = {"type": "ephemeral"}


def jpeg_b64(img, quality=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def image_block(img):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64(img)}}


def cost_usd(u):
    return (u["input"] * PRICE_PER_MTOK["input"] + u["output"] * PRICE_PER_MTOK["output"]
            + u["cache_read"] * PRICE_PER_MTOK["cache_read"] + u["cache_write"] * PRICE_PER_MTOK["cache_write"]) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--trial-id", required=True)
    ap.add_argument("--layout-id", default="")
    ap.add_argument("--condition", default="")
    ap.add_argument("--run-order", default="")
    ap.add_argument("--urdf", default=os.path.expanduser("~/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"))
    ap.add_argument("--calib", default="calib/touch_calib_v2.json")
    ap.add_argument("--prompt", default=str(HERE / "system_prompt_v1.md"))
    ap.add_argument("--results", default="results/pilot_a.csv")
    ap.add_argument("--no-viewer", action="store_true", help="skip the live Rerun window")
    args = ap.parse_args()

    out = Path("runs") / args.trial_id
    (out / "images").mkdir(parents=True, exist_ok=False)
    logf = open(out / "log.jsonl", "a")

    def log(kind, **kw):
        logf.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **kw}) + "\n")
        logf.flush()

    system = open(args.prompt).read()
    load_key_file()  # an API key here or in the environment wins; otherwise the SDK uses the `ant auth login` profile
    client = anthropic.Anthropic()
    rig = Rig(args.urdf, args.calib)
    rec = Recorder(str(out / "video.mp4"), rig.scene, rig.wrist)
    view = None
    if not args.no_viewer:
        try:
            view = LiveView(rig.scene, rig.wrist)
            view.note(f"{args.trial_id}: {args.instruction}")
        except Exception as e:  # the trial does not depend on the viewer
            print(f"live view unavailable: {e}")
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    n_calls, n_images, api_latency, final_text = 0, 0, [], ""
    stop_reason = "model_finished"

    def save(img, tag):
        nonlocal n_images
        n_images += 1
        cv2.imwrite(str(out / "images" / f"{n_images:03d}_{tag}.jpg"), img)

    try:
        rig.gripper("open")
        scene, wrist, state = rig.look()
        save(scene, "scene"); save(wrist, "wrist")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": f"Instruction: {args.instruction}\nCurrent arm state: {json.dumps(state)}\nScene image, then wrist image:"},
            image_block(scene), image_block(wrist)]}]
        log("start", instruction=args.instruction, model=MODEL, prompt=args.prompt, state=state)
        t_start = time.time()

        while True:
            if time.time() - t_start > TIME_LIMIT_S:
                stop_reason = "time_limit"; break
            mark_cache_point(messages)
            t0 = time.time()
            resp = client.messages.create(
                model=MODEL, max_tokens=2048, tools=TOOLS, messages=messages,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}])
            api_latency.append(time.time() - t0)
            u = resp.usage
            usage["input"] += u.input_tokens; usage["output"] += u.output_tokens
            usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
            messages.append({"role": "assistant", "content": resp.content})
            texts = [b.text for b in resp.content if b.type == "text"]
            calls = [b for b in resp.content if b.type == "tool_use"]
            log("model", latency_s=round(api_latency[-1], 2), text=texts, calls=[{"name": c.name, "input": c.input} for c in calls])
            if view and texts:
                view.note("Claude: " + " ".join(texts)[:300])
            if texts:
                final_text = texts[-1]
            if not calls:
                break
            results = []
            for c in calls:
                n_calls += 1
                if n_calls > MAX_TOOL_CALLS:
                    stop_reason = "call_limit"; break
                t1 = time.time()
                if c.name == "look":
                    scene, wrist, state = rig.look()
                    save(scene, "scene"); save(wrist, "wrist")
                    content = [{"type": "text", "text": json.dumps(state) + "\nScene image, then wrist image:"},
                               image_block(scene), image_block(wrist)]
                    logged = state
                else:
                    try:
                        r = getattr(rig, c.name)(**c.input)
                    except Exception as e:  # a bad argument should not kill the trial
                        r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    content, logged = json.dumps(r), r
                log("tool", name=c.name, input=c.input, result=logged, duration_s=round(time.time() - t1, 2))
                line = f"{c.name}({json.dumps(c.input)}) -> {json.dumps(logged)[:160]}"
                print("  " + line)
                if view:
                    view.note(line)
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": content})
            if stop_reason == "call_limit":
                break
            messages.append({"role": "user", "content": results})
    finally:
        duration = time.time() - t_start if "t_start" in locals() else 0.0
        try:
            rig.home()
        finally:
            rec.close()
            if view:
                view.note(f"stopped: {stop_reason}")
                view.close()
            rig.close()

    summary = {"trial_id": args.trial_id, "instruction": args.instruction, "stop_reason": stop_reason,
               "duration_s": round(duration, 1), "tool_calls": n_calls, "usage": usage,
               "api_cost_usd": round(cost_usd(usage), 4),
               "mean_decision_latency_s": round(sum(api_latency) / max(1, len(api_latency)), 2),
               "final_text": final_text}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))

    print("""
Score this trial (see SCORING.md):
  progress = highest stage reached: 0 nothing useful, 1 fingers at the object, 2 lifted it,
             3 carried it over the named cup, 4 it ended inside the named cup
  success  = 1 only if the named object is fully inside the named cup, gripper open, within 120 s
  failure tag = the FIRST thing that went wrong: perception / reach / grasp / transport / release / freeze
             (none if success)""")
    success = input("  success (1/0): ").strip()
    progress = input("  progress (0-4): ").strip()
    tag = input("  failure tag: ").strip()
    notes = input("  notes (anything odd; Enter to skip): ").strip()
    path = Path(args.results)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["trial_id", "layout_id", "condition", "arm", "run_order", "date", "scorer", "success",
                        "progress", "failure_tag", "duration_s", "tool_calls", "api_cost_usd",
                        "decision_latency_s", "rerun_reason", "notes"])
        w.writerow([args.trial_id, args.layout_id, args.condition, "A", args.run_order, time.strftime("%Y-%m-%d"),
                    "live", success, progress, tag, summary["duration_s"], n_calls, summary["api_cost_usd"],
                    summary["mean_decision_latency_s"], "", notes])
    log("scored", success=success, progress=progress, failure_tag=tag, notes=notes)
    print(f"saved {out} and a row in {path}")


if __name__ == "__main__":
    main()
