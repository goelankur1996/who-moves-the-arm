"""Arm C's marks for the demonstrations: Claude marks them all, then you approve or redo each one.

For each episode of ankurcw/wmta_demos_v2, the first scene frame (decoded from the dataset's own
video) and the episode's instruction go to arm_c.marks.get_marks, the same call a scored trial
makes on its first frame.

  --ask     asks Claude for every demo that has no answer yet or that you marked for redo
            (two at a time; about 3 minutes for 80). No window.
  (none)    the review window: the frame with Claude's rings, the instruction and close-ups of
            both rings, one demo after another, only demos waiting for a decision:
                a  approve      r  redo (asked again at the next --ask)      b  back      q  quit
            Nothing waits on Claude here. When the list is done it tells you what is left.
  --report  compares every answer with the layouts (for the record only) and writes
            results/marks_demos.csv.

Every answer is kept, the redone ones too: they are Claude's real error rate, which the trials
will not have a person to catch. The layouts are never used to place or check a mark.

    python scripts/mark_demos.py --ask --episodes 0,1   # try two
    python scripts/mark_demos.py                        # review them
    python scripts/mark_demos.py --ask                  # all the rest
    python scripts/mark_demos.py                        # review all; repeat --ask + review for redos

Writes results/marks_demos.json and results/marks_demos/frame0/*.png (the first frames, cached).
"""
import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arm_c import marks as M  # noqa: E402
from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402

REPO_ID = "ankurcw/wmta_demos_v2"
KEY = "observation.images.scene"
FPS = 30
REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "results" / "marks_demos.json"
OUT_CSV = REPO / "results" / "marks_demos.csv"
FRAME0 = REPO / "results" / "marks_demos" / "frame0"
RIM_CENTRES = {"pink_tumbler": (255.6, 313.0), "mug": (521.0, 250.5)}  # scripts/ghost.py RIMS
WIN = "arm C marks"


# ---------------------------------------------------------------- dataset

def episodes(root):
    """Episode metadata rows (episode_index, length, task, scene video file and first frame)."""
    rows = []
    for f in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        t = pq.read_table(f).to_pydict()
        for i, ep in enumerate(t["episode_index"]):
            rows.append({"episode": ep, "length": t["length"][i], "task": t["tasks"][i][0],
                         "video": root / "videos" / KEY / f"chunk-{t[f'videos/{KEY}/chunk_index'][i]:03d}"
                                  / f"file-{t[f'videos/{KEY}/file_index'][i]:03d}.mp4",
                         "first": round(t[f"videos/{KEY}/from_timestamp"][i] * FPS)})
    rows.sort(key=lambda r: r["episode"])
    assert [r["episode"] for r in rows] == list(range(len(rows)))
    return rows


def decode(video):
    """Every frame of a video file, in order, as RGB arrays."""
    with av.open(str(video)) as c:
        for fr in c.decode(c.streams.video[0]):
            yield fr.to_ndarray(format="rgb24")


def first_frames(rows):
    """{episode: first scene frame (RGB)}, decoded once and cached as lossless PNG."""
    FRAME0.mkdir(parents=True, exist_ok=True)
    path = {r["episode"]: FRAME0 / f"ep{r['episode']:03d}.png" for r in rows}
    missing = [r for r in rows if not path[r["episode"]].exists()]
    if missing:
        print(f"decoding the first frame of {len(missing)} episodes (once)...")
        for video in sorted({r["video"] for r in missing}):
            need = {r["first"]: r["episode"] for r in missing if r["video"] == video}
            for i, rgb in enumerate(decode(video)):
                if i in need:
                    cv2.imwrite(str(path[need[i]]), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if i >= max(need):
                    break
    return {e: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for e, p in path.items()}


def approved_marks(path=OUT_JSON):
    """{str(episode): the approved candidate} for every approved episode."""
    done = json.loads(Path(path).read_text())
    return {e: next(c for c in d["candidates"] if c.get("decision") == "approved")
            for e, d in done.items() if any(c.get("decision") == "approved" for c in d["candidates"])}


# ---------------------------------------------------------------- review window

def wrap(text, width):
    lines, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    return lines + [line]


def closeup(rgb, uv, half, size=150):
    u, v = (int(round(x)) for x in uv)
    pad = cv2.copyMakeBorder(rgb, half, half, half, half, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return cv2.resize(pad[v:v + 2 * half, u:u + 2 * half], (size, size), interpolation=cv2.INTER_NEAREST)


def view(rgb, ep, n, d, cand, status, n_approved, progress=""):
    """960 x 480: the frame (with rings) on the left, instruction, close-ups and keys on the right."""
    img = M.draw(rgb, cand) if cand and "grasp" in cand else rgb.copy()
    panel = np.full((480, 320, 3), 255, np.uint8)
    put = lambda t, y, s=0.5, c=(0, 0, 0), th=1: cv2.putText(panel, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, th, cv2.LINE_AA)
    put(f"{d['layout_id']}   (episode {ep}, {ep + 1}/{n})", 24, 0.5, (90, 90, 90))
    y = 60
    for line in wrap(d["instruction"], 22):
        put(line, y, 0.75, (0, 0, 0), 2)
        y += 30
    k = len(d["candidates"])
    if cand and "grasp" in cand:
        put(f"Claude's answer {k}" + ("" if k == 1 else f" ({k - 1} redone)"), y + 4, 0.45, (90, 90, 90))
        panel[y + 16:y + 166, 8:158] = closeup(img, cand["grasp"], 40)
        panel[y + 16:y + 166, 162:312] = closeup(img, cand["release"], 55)
        put("pick (magenta)", y + 182, 0.42, (90, 90, 90))
        cv2.putText(panel, "container (blue)", (166, y + 182), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    put(status, 400, 0.55, (0, 0, 200) if status else (0, 0, 0), 1)
    put("a approve   r redo   b back   q quit", 440, 0.48)
    put(f"{progress}    approved {n_approved}/{n}", 466, 0.45, (90, 90, 90))
    return cv2.cvtColor(np.hstack([img, panel]), cv2.COLOR_RGB2BGR)


def load(rows=None):
    """The answers file, with an entry for every episode."""
    done = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    if rows is not None:
        index = {int(r["episode_index"]): r for r in csv.DictReader(open(REPO / "results" / "demos_v2_index.csv"))}
        for r in rows:
            e = r["episode"]
            assert r["task"] == index[e]["instruction"], f"episode {e}: task differs from the index; stop"
            done.setdefault(str(e), {"layout_id": index[e]["layout_id"], "instruction": r["task"], "candidates": []})
    return done


def save(done):
    OUT_JSON.write_text(json.dumps(done, indent=1))


def state(d):
    """'approved', 'waiting' (an answer to decide on), 'error' (Claude gave no marks) or 'ask'."""
    c = d["candidates"]
    if any(x.get("decision") == "approved" for x in c):
        return "approved"
    if not c or c[-1].get("decision") == "redo":
        return "ask"
    return "error" if "error" in c[-1] else "waiting"


def ask(wanted):
    rows = episodes(HF_LEROBOT_HOME / REPO_ID)
    frames, done = first_frames(rows), load(rows)
    todo = [e for e in (wanted or range(len(rows))) if state(done[str(e)]) in ("ask", "error")]
    for e in todo:  # a failed answer is kept and asked again
        if done[str(e)]["candidates"] and "error" in done[str(e)]["candidates"][-1]:
            done[str(e)]["candidates"][-1]["decision"] = "redo"
    print(f"asking Claude for {len(todo)} demos")
    api = M.client()

    def one(e):
        try:
            c = M.get_marks(frames[e], done[str(e)]["instruction"], api)
        except M.NoMarks as err:
            c = {"error": str(err)[:800]}
        return e, {**c, "asked_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    with ThreadPoolExecutor(2) as pool:
        for k, (e, c) in enumerate(pool.map(one, todo), 1):
            done[str(e)]["candidates"].append(c)
            save(done)
            print(f"  {k}/{len(todo)}  {done[str(e)]['layout_id']}: "
                  + (f"marked ({c['latency_s']}s)" if "grasp" in c else "NO MARKS: " + c["error"][:200]))
    save(done)
    summary(done)


def summary(done):
    n = {s: sum(state(d) == s for d in done.values()) for s in ("approved", "waiting", "error", "ask")}
    print(f"approved {n['approved']}/{len(done)}, to review {n['waiting']}, "
          f"to ask Claude {n['ask'] + n['error']} (redo or no marks)")
    if n["waiting"]:
        print("next: python scripts/mark_demos.py   (review)")
    elif n["ask"] + n["error"]:
        print("next: python scripts/mark_demos.py --ask")
    else:
        print("all approved. next: python scripts/mark_demos.py --report")


def review():
    rows = episodes(HF_LEROBOT_HOME / REPO_ID)
    frames, done = first_frames(rows), load(rows)
    queue_ = [r["episode"] for r in rows if state(done[str(r["episode"])]) in ("waiting", "error")]
    if not queue_:
        summary(done)
        return
    n, k = len(rows), 0
    cv2.namedWindow(WIN)
    while 0 <= k < len(queue_):
        e = queue_[k]
        d = done[str(e)]
        cand = d["candidates"][-1]
        dec = cand.get("decision")
        status = ("no marks from Claude: press r" if "error" in cand else
                  "approved" if dec == "approved" else "marked for redo" if dec == "redo" else "")
        cv2.imshow(WIN, view(frames[e], e, n, d, cand if "grasp" in cand else None, status,
                             sum(state(x) == "approved" for x in done.values()), f"review {k + 1}/{len(queue_)}"))
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            k = max(0, k - 1)
        elif key == ord("a") and "grasp" in cand:
            cand["decision"] = "approved"
            save(done)
            k += 1
        elif key == ord("r"):
            cand["decision"] = "redo"
            save(done)
            k += 1
    cv2.destroyAllWindows()
    save(done)
    summary(done)


# ---------------------------------------------------------------- report

def score(mk, layout):
    """Errors against the layout, for the record only."""
    dist = {p["id"]: math.dist(mk["grasp"], p["pixel"]) for p in layout["pieces"]}
    rdist = {c: math.dist(mk["release"], xy) for c, xy in RIM_CENTRES.items()}
    return {"grasp_err_px": round(dist[layout["target"]], 1), "grasp_nearest": min(dist, key=dist.get),
            "grasp_right_object": min(dist, key=dist.get) == layout["target"],
            "release_err_px": round(rdist[layout["container"]], 1), "release_nearest": min(rdist, key=rdist.get),
            "release_right_container": min(rdist, key=rdist.get) == layout["container"]}


def report():
    done = json.loads(OUT_JSON.read_text())
    layouts = {L["layout_id"]: L for L in json.load(open(REPO / "layouts" / "layouts.json"))["layouts"]}
    fields = ["episode_index", "layout_id", "condition", "instruction", "answer", "decision", "grasp_u", "grasp_v",
              "release_u", "release_v", "attempts", "latency_s", "error", "grasp_err_px", "grasp_nearest",
              "grasp_right_object", "release_err_px", "release_nearest", "release_right_container"]
    rows = []
    for e in sorted(int(k) for k in done):
        d = done[str(e)]
        for i, c in enumerate(d["candidates"], 1):
            row = {"episode_index": e, "layout_id": d["layout_id"], "condition": layouts[d["layout_id"]]["condition"],
                   "instruction": d["instruction"], "answer": i, "decision": c.get("decision", ""),
                   "error": c.get("error", "")[:200]}
            if "grasp" in c:
                row.update({"grasp_u": c["grasp"][0], "grasp_v": c["grasp"][1], "release_u": c["release"][0],
                            "release_v": c["release"][1], "attempts": c["attempts"], "latency_s": c["latency_s"],
                            **score(c, layouts[d["layout_id"]])})
            rows.append(row)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    first = [r for r in rows if r["answer"] == 1]
    ok = [r for r in rows if "grasp_err_px" in r]
    print(f"{len(done)} demos, {len(rows)} answers from Claude ({len(rows) - len(first)} redos)")
    print(f"  first answer approved: {sum(r['decision'] == 'approved' for r in first)}/{len(first)}")
    print(f"  approved overall:      {len({r['episode_index'] for r in rows if r['decision'] == 'approved'})}/{len(done)}")
    if ok:
        med = lambda k: sorted(r[k] for r in ok)[len(ok) // 2]
        print(f"  all answers vs layouts: right object {sum(r['grasp_right_object'] for r in ok)}/{len(ok)}, "
              f"right container {sum(r['release_right_container'] for r in ok)}/{len(ok)}; "
              f"median error grasp {med('grasp_err_px')} px, release {med('release_err_px')} px")
    print(f"  written: {OUT_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", action="store_true", help="ask Claude for every demo without an answer to review")
    ap.add_argument("--episodes", help="with --ask: only these episode indices, comma-separated")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.ask:
        ask([int(x) for x in a.episodes.split(",")] if a.episodes else None)
    elif a.report:
        report()
    else:
        review()
