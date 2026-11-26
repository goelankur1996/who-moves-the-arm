"""Build the final demonstration dataset with episode N = layout D-(N+1), using LeRobot's own tools.

Sources (neither is modified):
  ankurcw/wmta_demos_v1       80 episodes, D-01..D-80 in order; 5 have damaged video (dropped frames)
  ankurcw/wmta_demos_v1_redo  the 5 re-recorded layouts, in the order listed in REDO below
Output:
  ankurcw/wmta_demos_v2       80 episodes; the 5 damaged ones replaced in place
  results/demos_v2_index.csv  episode -> layout, instruction, source, frames

Steps: split v1 into runs of good episodes and the redo set into single episodes
(lerobot.datasets.dataset_tools.split_dataset), merge them back in D order (merge_datasets),
then check every episode against its source before reporting success.

    python scripts/build_demo_dataset.py
"""
import csv
import json
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets, split_dataset
from lerobot.utils.constants import HF_LEROBOT_HOME

V1, REDO, OUT = "ankurcw/wmta_demos_v1", "ankurcw/wmta_demos_v1_redo", "ankurcw/wmta_demos_v2"
REDO_LAYOUTS = ["D-28", "D-31", "D-32", "D-33", "D-37"]  # redo episode j is REDO_LAYOUTS[j]
TMP = HF_LEROBOT_HOME / "ankurcw" / "_wmta_build_tmp"
VIDEO_KEYS = ["observation.images.scene", "observation.images.wrist"]
N = 80


def load(repo_id):
    return LeRobotDataset(repo_id, root=HF_LEROBOT_HOME / repo_id)


def read_frames(root):
    return pd.concat([pd.read_parquet(f) for f in sorted(Path(root).glob("data/chunk-*/file-*.parquet"))])


def read_episodes(root):
    return pd.concat([pd.read_parquet(f) for f in sorted(Path(root).glob("meta/episodes/chunk-*/file-*.parquet"))])


def video_frame_counts(root, eps, key):
    """Frames actually present in the video for each episode, from packet timestamps."""
    counts = {}
    for fi, g in eps.groupby(f"videos/{key}/file_index"):
        ci = int(g[f"videos/{key}/chunk_index"].iloc[0])
        path = Path(root) / f"videos/{key}/chunk-{ci:03d}/file-{int(fi):03d}.mp4"
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            pts = np.array(sorted(float(p.pts * s.time_base) for p in c.demux(s) if p.pts is not None))
        for _, r in g.iterrows():
            a, b = r[f"videos/{key}/from_timestamp"], r[f"videos/{key}/to_timestamp"]
            counts[int(r["episode_index"])] = int(((pts >= a - 1e-3) & (pts < b - 1e-3)).sum())
    return counts


def main():
    out_root = HF_LEROBOT_HOME / OUT
    assert not out_root.exists(), f"{out_root} exists; move it away first"
    assert not TMP.exists(), f"{TMP} exists from an earlier run; delete it first"

    # Plan: which source each final episode comes from.
    redo_of = {int(lid[2:]) - 1: j for j, lid in enumerate(REDO_LAYOUTS)}
    plan = [("redo", redo_of[i]) if i in redo_of else ("v1", i) for i in range(N)]
    pieces, run = [], []
    for i, (src, k) in enumerate(plan):
        if src == "v1":
            run.append(k)
        else:
            if run:
                pieces.append(("v1", run))
                run = []
            pieces.append(("redo", [k]))
    if run:
        pieces.append(("v1", run))
    print("pieces:", [(s, f"{e[0]}-{e[-1]}" if len(e) > 1 else str(e[0])) for s, e in pieces])

    v1, redo = load(V1), load(REDO)
    assert v1.meta.total_episodes == N and redo.meta.total_episodes == len(REDO_LAYOUTS)

    v1_splits = {f"p{n:02d}": eps for n, (s, eps) in enumerate(pieces) if s == "v1"}
    redo_splits = {f"p{n:02d}": eps for n, (s, eps) in enumerate(pieces) if s == "redo"}
    parts = {}
    parts.update(split_dataset(v1, v1_splits, output_dir=TMP / "v1"))
    parts.update(split_dataset(redo, redo_splits, output_dir=TMP / "redo"))
    ordered = [parts[f"p{n:02d}"] for n in range(len(pieces))]
    merge_datasets(ordered, OUT, output_dir=out_root)

    # ---------- checks ----------
    layouts = {L["layout_id"]: L for L in json.load(open("layouts/layouts.json"))["layouts"]}
    out_f, out_e = read_frames(out_root), read_episodes(out_root).sort_values("episode_index")
    tasks = pd.read_parquet(out_root / "meta/tasks.parquet")
    task_of = {int(t): name for name, t in zip(tasks.index, tasks["task_index"])}
    src_f = {"v1": read_frames(HF_LEROBOT_HOME / V1), "redo": read_frames(HF_LEROBOT_HOME / REDO)}
    counts = {k: video_frame_counts(out_root, out_e, k) for k in VIDEO_KEYS}
    assert len(out_e) == N, f"{len(out_e)} episodes in the output, expected {N}"

    rows, problems = [], []
    for i, (src, k) in enumerate(plan):
        lid = f"D-{i + 1:02d}"
        a = out_f[out_f.episode_index == i].sort_values("frame_index")
        b = src_f[src][src_f[src].episode_index == k].sort_values("frame_index")
        n = len(a)
        if n != len(b):
            problems.append(f"{lid}: {n} frames, source has {len(b)}")
        for col in ("action", "observation.state"):
            if not np.array_equal(np.stack(a[col].values), np.stack(b[col].values)):
                problems.append(f"{lid}: {col} differs from source")
        instr = task_of[int(a["task_index"].iloc[0])]
        if instr != layouts[lid]["instruction"]:
            problems.append(f"{lid}: task '{instr}', layout says '{layouts[lid]['instruction']}'")
        for key in VIDEO_KEYS:
            if counts[key][i] != n:
                problems.append(f"{lid}: {key} has {counts[key][i]} video frames for {n} rows")
        rows.append({"episode_index": i, "layout_id": lid, "condition": layouts[lid]["condition"],
                     "instruction": instr, "frames": n,
                     "source": f"{V1} episode {k}" if src == "v1" else f"{REDO} episode {k}"})

    if problems:
        print("CHECK FAILED; the output is kept for inspection, the sources are untouched:")
        print("\n".join("  " + p for p in problems))
        raise SystemExit(1)

    with open("results/demos_v2_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    shutil.rmtree(TMP)
    print(f"OK: {N} episodes in {out_root}, episode N = D-(N+1); {sum(r['frames'] for r in rows)} frames; "
          f"joint data identical to source; every video frame present; index in results/demos_v2_index.csv")


if __name__ == "__main__":
    main()
