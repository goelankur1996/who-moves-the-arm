"""Check a recorded LeRobot dataset (v3 format) before training on it.

    python scripts/check_data.py ankurcw/wmta_demos_v2
    python scripts/check_data.py ankurcw/wmta_demos_v2 --frames _review/check_frames.jpg

Reads the dataset folder directly (data/, meta/, videos/); changes nothing.
Exit code 1 if any ERROR; WARN lines are worth a look but do not fail the check.

ERROR checks
  files     episode count and frame count agree across data, episode metadata and info.json;
            episodes numbered 0..N-1; frame_index runs 0..len-1; timestamp = frame_index / fps
  video     every camera has exactly one video frame per data row in every episode
  frozen    the leader commands a joint through 5+ units but the follower moves it under 1
            (dead or unplugged motor)
  tracking  follower lags the leader's command by more than 25 units (95th percentile) on a joint
  grasp     no grasp found (gripper never opens and then closes)
  task      the episode's instruction is not the one in layouts/layouts.json for its layout
            (layout = D-(episode+1); only with --layouts, on by default for wmta datasets)
WARN checks
  unused    a joint moves under 1 unit and was not commanded to move either (operator's choice)
  idle      the arm does not move for more than 1 s at the start
  length    episode length more than 3 MADs from the median
  pose      wrist flex or wrist roll at the grasp more than 3 MADs from the median
  rest      start or end pose away from rest (base 45-70 deg, other joints within 15 deg of
            calib/rest_pose.json)
"""
import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FROZEN_UNITS = 1.0        # follower moved less than this...
COMMANDED_UNITS = 5.0     # ...while the leader commanded at least this: a dead motor
TRACKING_UNITS = 25.0
IDLE_S = 1.0
MOVE_UNITS = 0.5          # per-frame change that counts as moving
REST_BASE = (45.0, 70.0)  # deg; the rest pose is a range for the base (27 Nov)
REST_OTHER_DEG = 15.0
MAD_Z = 3.0


def mad_z(x):
    x = np.asarray(x, float)
    m = np.nanmedian(x)
    d = np.nanmedian(np.abs(x - m)) * 1.4826
    return (x - m) / d if d > 0 else np.zeros_like(x)


def read_parquets(root, pattern):
    files = sorted(root.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {root / pattern}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def video_counts(root, eps, key):
    """Frames present in the video for each episode, from packet timestamps."""
    out = {}
    for (ci, fi), g in eps.groupby([f"videos/{key}/chunk_index", f"videos/{key}/file_index"]):
        path = root / f"videos/{key}/chunk-{int(ci):03d}/file-{int(fi):03d}.mp4"
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            pts = np.sort([float(p.pts * s.time_base) for p in c.demux(s) if p.pts is not None])
        for _, r in g.iterrows():
            a, b = r[f"videos/{key}/from_timestamp"], r[f"videos/{key}/to_timestamp"]
            out[int(r["episode_index"])] = int(((pts >= a - 1e-3) & (pts < b - 1e-3)).sum())
    return out


def find_grasp(grip):
    """First frame where the gripper closes after having opened; None if it never does."""
    lo, hi = np.percentile(grip, 5), np.percentile(grip, 95)
    if hi - lo < 5:
        return None
    opened = np.where(grip > lo + 0.7 * (hi - lo))[0]
    if not len(opened):
        return None
    closed = np.where((np.arange(len(grip)) > opened[0]) & (grip < lo + 0.3 * (hi - lo)))[0]
    return int(closed[0]) if len(closed) else None


def frame_at(root, ep_row, key, t):
    path = root / (f"videos/{key}/chunk-{int(ep_row[f'videos/{key}/chunk_index']):03d}/"
                   f"file-{int(ep_row[f'videos/{key}/file_index']):03d}.mp4")
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(t - 1.0, 0) / s.time_base), stream=s, backward=True)
        best = None
        for f in c.decode(s):
            best = f
            if f.time is not None and f.time >= t - 1e-3:
                break
        return best.to_ndarray(format="bgr24")


def contact_sheet(root, eps, index, out):
    import cv2
    key = "observation.images.scene"
    tiles = []
    for _, r in eps.iterrows():
        pair = []
        for t in (r[f"videos/{key}/from_timestamp"] + 0.02, r[f"videos/{key}/to_timestamp"] - 0.05):
            pair.append(cv2.resize(frame_at(root, r, key, t)[140:440, 140:620], (240, 150)))
        tile = np.hstack(pair)
        e = int(r["episode_index"])
        label = f"ep {e}" + (f" {index[e]['layout_id']}: {index[e]['instruction'][8:]}" if e in index else "")
        cv2.putText(tile, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(tile, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(160, 160, 160)))
    tiles += [np.full_like(tiles[0], 255)] * ((-len(tiles)) % 4)
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("--root", help="dataset folder (default: ~/.cache/huggingface/lerobot/<repo_id>)")
    ap.add_argument("--layouts", default="layouts/layouts.json",
                    help="check each episode's instruction against layout D-(episode+1); 'none' to skip")
    ap.add_argument("--rest", default="calib/rest_pose.json")
    ap.add_argument("--frames", help="also write a sheet of first/last scene frames per episode (JPEG)")
    args = ap.parse_args()
    root = Path(args.root) if args.root else Path.home() / ".cache/huggingface/lerobot" / args.repo_id

    info = json.load(open(root / "meta/info.json"))
    fps = info["fps"]
    df = read_parquets(root, "data/chunk-*/file-*.parquet")
    eps = read_parquets(root, "meta/episodes/chunk-*/file-*.parquet").sort_values("episode_index")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    task_name = {int(t): name for name, t in zip(tasks.index, tasks["task_index"])}
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    errors, warns = [], []
    err = lambda e, what, msg: errors.append((e, what, msg))   # noqa: E731
    warn = lambda e, what, msg: warns.append((e, what, msg))   # noqa: E731

    # files
    n = info["total_episodes"]
    if not (df.episode_index.nunique() == len(eps) == n):
        err(-1, "files", f"episodes: data {df.episode_index.nunique()}, metadata {len(eps)}, info.json {n}")
    if len(df) != info["total_frames"]:
        err(-1, "files", f"frames: data {len(df)}, info.json {info['total_frames']}")
    if sorted(eps.episode_index) != list(range(len(eps))):
        err(-1, "files", "episodes are not numbered 0..N-1")

    layouts = None
    if args.layouts != "none" and Path(args.layouts).exists():
        layouts = {L["layout_id"]: L for L in json.load(open(args.layouts))["layouts"]}
    rest = np.array(json.load(open(args.rest))["joints_deg"]) if Path(args.rest).exists() else None

    counts = {k: video_counts(root, eps, k) for k in video_keys}
    per_ep, index = [], {}
    for _, r in eps.iterrows():
        e = int(r["episode_index"])
        g = df[df.episode_index == e].sort_values("frame_index")
        n_rows = len(g)
        if n_rows != r["length"]:
            err(e, "files", f"{n_rows} data rows, metadata says {r['length']}")
        if not np.array_equal(g.frame_index.values, np.arange(n_rows)):
            err(e, "files", "frame_index is not 0..len-1")
        if not np.allclose(g.timestamp.values, np.arange(n_rows) / fps, atol=1e-4):
            err(e, "files", "timestamp is not frame_index / fps")
        for k in video_keys:
            if counts[k].get(e) != n_rows:
                err(e, "video", f"{k.split('.')[-1]} has {counts[k].get(e)} frames for {n_rows} rows")

        s = np.stack(g["observation.state"].values)
        a = np.stack(g["action"].values)
        rng, cmd = s.max(0) - s.min(0), a.max(0) - a.min(0)
        for j in np.where(rng < FROZEN_UNITS)[0]:
            if cmd[j] >= COMMANDED_UNITS:
                err(e, "frozen", f"{JOINTS[j]} moved {rng[j]:.2f} while commanded through {cmd[j]:.1f}")
            else:
                warn(e, "unused", f"{JOINTS[j]} moved {rng[j]:.2f}, commanded {cmd[j]:.2f}")
        lag = np.percentile(np.abs(s[1:] - a[:-1]), 95, axis=0)
        for j in np.where(lag > TRACKING_UNITS)[0]:
            err(e, "tracking", f"{JOINTS[j]} lags its command by {lag[j]:.1f} (95th pct)")
        k = find_grasp(s[:, 5])
        if k is None:
            err(e, "grasp", "no grasp found")

        moving = np.where(np.abs(np.diff(s, axis=0)).max(1) > MOVE_UNITS)[0]
        idle = moving[0] / fps if len(moving) else n_rows / fps
        if idle > IDLE_S:
            warn(e, "idle", f"no motion for the first {idle:.1f} s")
        if rest is not None:
            for when, q in (("start", s[0, :5]), ("end", s[-1, :5])):
                off = np.abs(q[1:] - rest[1:]).max()
                if not (REST_BASE[0] <= q[0] <= REST_BASE[1]) or off > REST_OTHER_DEG:
                    warn(e, "rest", f"{when}: base {q[0]:.0f} deg, other joints up to {off:.0f} deg from rest")

        instr = task_name[int(g.task_index.iloc[0])]
        if g.task_index.nunique() != 1:
            err(e, "task", "more than one task in the episode")
        if layouts is not None:
            lid = f"D-{e + 1:02d}"
            if lid not in layouts or layouts[lid]["instruction"] != instr:
                err(e, "task", f"'{instr}' does not match layout {lid}")
            index[e] = {"layout_id": lid, "instruction": instr}
        per_ep.append({"episode": e, "frames": n_rows, "seconds": n_rows / fps,
                       "wrist_flex": s[k, 3] if k is not None else np.nan,
                       "wrist_roll": s[k, 4] if k is not None else np.nan})

    p = pd.DataFrame(per_ep)
    for col, what in (("frames", "length"), ("wrist_flex", "pose"), ("wrist_roll", "pose")):
        z = mad_z(p[col])
        for i in np.where(np.abs(z) > MAD_Z)[0]:
            warn(int(p.episode[i]), what, f"{col} {p[col][i]:.1f} (median {np.nanmedian(p[col]):.1f})")

    # report
    print(f"{args.repo_id}: {len(eps)} episodes, {len(df)} frames, {fps} fps, cameras: "
          f"{', '.join(k.split('.')[-1] for k in video_keys)}; "
          f"episode length {p.seconds.min():.1f}-{p.seconds.max():.1f} s (median {p.seconds.median():.1f})")
    for label, items in (("ERROR", errors), ("WARN", warns)):
        for e, what, msg in sorted(items, key=lambda x: (x[1], x[0])):
            print(f"  {label:5} {what:8} {'all' if e < 0 else f'ep {e}':7} {msg}")
    if args.frames:
        contact_sheet(root, eps, index, args.frames)
        print(f"first/last scene frames: {args.frames}")
    print(f"{len(errors)} errors, {len(warns)} warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
