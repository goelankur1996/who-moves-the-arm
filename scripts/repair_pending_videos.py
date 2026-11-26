"""Encode videos for episodes whose frames were saved as images but never encoded.

27 Nov 2026: LeRobot's batch video encoding crashed on the resumed dataset, leaving episodes
37-46 (D-38 to D-47) with complete joint data and their frames as PNG images, but no video and
no video entries in the episode metadata. This script finishes that job:

  1. backs up meta/ next to the dataset
  2. encodes each pending episode's PNGs with the dataset's own encoder settings (LeRobot's
     encode_video_frames, default camera encoder: AV1, CRF 30, as every other episode)
  3. checks every episode video has exactly as many frames as the episode has rows
  4. joins them into one new video file per camera (existing video files are not touched)
  5. writes the video columns into the episode metadata, in the same schema as the other files
  6. only then deletes the PNG folders it encoded

    python scripts/repair_pending_videos.py            # dry run: shows what it would do
    python scripts/repair_pending_videos.py --apply
"""
import argparse
import shutil
import tempfile
import time
from pathlib import Path

import av
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.configs.video import camera_encoder_defaults
from lerobot.datasets.video_utils import concatenate_video_files, encode_video_frames
from lerobot.utils.constants import HF_LEROBOT_HOME

REPO_ID = "ankurcw/wmta_demos_v1"
KEYS = ["observation.images.scene", "observation.images.wrist"]
FPS = 30


def n_frames(path):
    with av.open(str(path)) as c:
        return sum(1 for _ in c.demux(c.streams.video[0]) if _.size > 0)


def write_video_columns(path, table, schema, new_cols):
    """Rewrite an episode-metadata parquet file with the video columns added, using the column
    order and types of `schema` (a metadata file that already has video columns)."""
    arrays = [pa.array(new_cols[f.name], type=f.type) if f.name in new_cols
              else table.column(f.name).cast(f.type) for f in schema]
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = HF_LEROBOT_HOME / REPO_ID

    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    tables = {f: pq.read_table(f) for f in files}
    vcol = f"videos/{KEYS[0]}/file_index"
    done = [f for f, t in tables.items() if vcol in t.column_names]
    pending = [f for f, t in tables.items() if vcol not in t.column_names]
    if not pending:
        print("nothing to repair: every episode has its video entries")
        return
    assert len(pending) == 1 and pending[0] == files[-1], f"unexpected layout: pending files {pending}"
    target, ref = pending[0], tables[done[-1]]
    other = [n for n in ref.column_names if not n.startswith("videos/")]
    assert sorted(other) == sorted(tables[target].column_names), "metadata columns differ; stop"
    t = tables[target].to_pydict()
    eps, lengths = t["episode_index"], t["length"]
    print(f"pending: episodes {eps[0]}-{eps[-1]} in {target.relative_to(root)} ({sum(lengths)} frames)")

    plan = {}
    for key in KEYS:
        last = tables[done[-1]].to_pydict()
        chunk = last[f"videos/{key}/chunk_index"][-1]
        file_i = last[f"videos/{key}/file_index"][-1] + 1
        out = root / f"videos/{key}/chunk-{chunk:03d}/file-{file_i:03d}.mp4"
        reuse = out.exists()
        if reuse:  # written by an earlier run that stopped before the metadata step
            got = n_frames(out)
            assert got == sum(lengths), f"{out} exists with {got} frames, expected {sum(lengths)}; stop"
        for ep, n in zip(eps, lengths):
            pngs = len(list((root / f"images/{key}/episode-{ep:06d}").glob("*.png")))
            assert pngs == n, f"{key} episode {ep}: {pngs} images but {n} rows; stop"
        plan[key] = (chunk, file_i, out, reuse)
        print(f"  {key}: {len(eps)} episodes -> {out.relative_to(root)}"
              + (" (already written and complete; reused)" if reuse else ""))
    if not args.apply:
        print("dry run; add --apply to do it")
        return

    backup = root.parent / f"{root.name}_meta_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(root / "meta", backup)
    print(f"meta backed up to {backup}")

    tmp = tempfile.mkdtemp(dir=root)
    new_cols = {}
    try:
        for key, (chunk, file_i, out, reuse) in plan.items():
            parts, t0 = [], 0.0
            cols = {f"videos/{key}/{c}": [] for c in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")}
            for ep, n in zip(eps, lengths):
                if not reuse:
                    part = Path(tmp) / f"{key}_{ep:06d}.mp4"  # LeRobot's concatenate needs Path objects
                    encode_video_frames(root / f"images/{key}/episode-{ep:06d}", part, FPS,
                                        camera_encoder=camera_encoder_defaults(), overwrite=True)
                    got = n_frames(part)
                    assert got == n, f"{key} episode {ep}: video has {got} frames, expected {n}; stop"
                    parts.append(part)
                    print(f"  encoded {key} episode {ep}: {n} frames")
                cols[f"videos/{key}/chunk_index"].append(chunk)
                cols[f"videos/{key}/file_index"].append(file_i)
                cols[f"videos/{key}/from_timestamp"].append(t0)
                cols[f"videos/{key}/to_timestamp"].append(t0 + n / FPS)
                t0 += n / FPS
            if not reuse:
                out.parent.mkdir(parents=True, exist_ok=True)
                concatenate_video_files(parts, out)
            got = n_frames(out)
            assert got == sum(lengths), f"{out}: {got} frames after joining, expected {sum(lengths)}; stop"
            new_cols.update(cols)
            print(f"  {'checked' if reuse else 'wrote'} {out.relative_to(root)} ({got} frames)")

        write_video_columns(target, tables[target], ref.schema, new_cols)
        print(f"  episode metadata updated: {target.relative_to(root)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for key in KEYS:
        for ep in eps:
            shutil.rmtree(root / f"images/{key}/episode-{ep:06d}")
    for d in [root / "images" / k for k in KEYS] + [root / "images"]:
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    print("done: PNG folders for the repaired episodes removed")


if __name__ == "__main__":
    main()
