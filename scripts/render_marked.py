"""Step 2 of arm C's data: build ankurcw/wmta_demos_v2_marked, the demonstrations with the rings.

Copies ankurcw/wmta_demos_v2 and redraws only its scene video: every frame of episode N gets the
rings you approved for episode N (results/marks_demos.json), drawn by arm_c.marks.draw (the same call the
trials use), and the video is re-encoded with LeRobot's own encoder settings (AV1, CRF 30, as the
original). Joint data, instructions, the wrist video and all metadata are copied unchanged, so
episode N is still layout D-(N+1). The source dataset is not touched.

The image statistics in meta/stats.json are kept from v2: the rings cover well under 1% of the
pixels, so per-channel mean and std move by about a thousandth.

Checks: every episode has marks; the new video has exactly as many frames as the old one; pixels
away from the rings match the original to within encoding noise. Writes results/marks_audit.jpg,
20 random frames of the new dataset (for the G2 visual audit).

    python scripts/render_marked.py            # build
    python scripts/render_marked.py --push     # build if needed, then upload (public)
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_c import marks as M  # noqa: E402
from mark_demos import FPS, KEY, REPO, REPO_ID, approved_marks, decode, episodes  # noqa: E402
from lerobot.configs.video import camera_encoder_defaults  # noqa: E402
from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402

OUT_ID = "ankurcw/wmta_demos_v2_marked"
AUDIT = REPO / "results" / "marks_audit.jpg"


def ring_mask(mk):
    """Pixels within 4 px of either ring (where the drawing and its encoding blur can differ)."""
    m = np.zeros((M.H, M.W), np.uint8)
    for (u, v), r in ((mk["grasp"], M.GRASP_RADIUS), (mk["release"], M.RELEASE_RADIUS)):
        cv2.circle(m, (int(round(u)), int(round(v))), r, 255, M.THICKNESS + 8)
    return m > 0


def render_file(src, dst, rows, marks):
    """Decode src, draw each episode's rings on its frames, encode to dst. Returns the frame count."""
    spans = [(r["first"], r["first"] + r["length"], marks[str(r["episode"])]) for r in rows]
    enc = camera_encoder_defaults()
    dst.parent.mkdir(parents=True, exist_ok=True)
    n, k = 0, 0
    with av.open(str(dst), "w") as out:
        st = out.add_stream(enc.vcodec, FPS, options=enc.get_codec_options(None, as_strings=True))
        st.pix_fmt, st.width, st.height = enc.pix_fmt, M.W, M.H
        for i, rgb in enumerate(decode(src)):
            while k < len(spans) and i >= spans[k][1]:
                k += 1
            assert k < len(spans) and spans[k][0] <= i, f"{src.name}: frame {i} belongs to no episode; stop"
            for p in st.encode(av.VideoFrame.from_ndarray(M.draw(rgb, spans[k][2]), format="rgb24")):
                out.mux(p)
            n += 1
        for p in st.encode():
            out.mux(p)
    return n


def count(video):
    with av.open(str(video)) as c:
        return sum(1 for p in c.demux(c.streams.video[0]) if p.size > 0)


def frames_at(video, idxs):
    """{index: RGB frame} for the given frame indices, in one pass over the video."""
    idxs, got = set(idxs), {}
    for i, rgb in enumerate(decode(video)):
        if i in idxs:
            got[i] = rgb
        if len(got) == len(idxs):
            break
    return got


def build(src_root, dst_root, marks):
    rows = episodes(src_root)
    missing = [r["episode"] for r in rows if "grasp" not in marks.get(str(r["episode"]), {})]
    assert not missing, f"episodes without approved marks: {missing}. Finish scripts/mark_demos.py first; stop"
    tmp = dst_root.parent / (dst_root.name + "_building")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src_root, tmp, ignore=lambda d, names: names if Path(d) == src_root / "videos" / KEY else [])
    for video in sorted({r["video"] for r in rows}):
        rel = video.relative_to(src_root)
        n = render_file(video, tmp / rel, [r for r in rows if r["video"] == video], marks)
        assert count(tmp / rel) == count(video) == n, f"{rel}: frame count differs; stop"
        print(f"  {rel}: {n} frames redrawn")

    random.seed(0)
    sample = random.sample(rows, 5)  # away from the rings, the frames must match the original
    for video in sorted({r["video"] for r in sample}):
        rs = [r for r in sample if r["video"] == video]
        idx = [r["first"] + r["length"] // 2 for r in rs]
        old, new = frames_at(video, idx), frames_at(tmp / video.relative_to(src_root), idx)
        for r, i in zip(rs, idx):
            d = np.abs(old[i].astype(int) - new[i].astype(int))
            ring = ring_mask(marks[str(r["episode"])])
            off, on = d[~ring].mean(), d[ring].mean()
            print(f"  episode {r['episode']} frame {i}: mean change {off:.2f} away from the rings, {on:.1f} on them")
            assert off < 5 and on > 10, "the redrawn frame does not look like the original plus rings; stop"
    tmp.rename(dst_root)


def audit(dst_root, marks):
    rows = episodes(dst_root)
    random.seed(1)
    picks = sorted((r["video"], r["first"] + random.randrange(r["length"]), r["episode"]) for r in random.sample(rows, 20))
    got = {}
    for video in sorted({v for v, _, _ in picks}):
        got[video] = frames_at(video, [i for v, i, _ in picks if v == video])
    tiles = []
    for video, i, ep in picks:
        t = cv2.resize(got[video][i], (320, 240))
        cv2.putText(t, f"ep {ep} D-{ep + 1:02d} f{i}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(t)
    grid = np.vstack([np.hstack(tiles[j:j + 5]) for j in range(0, 20, 5)])
    cv2.imwrite(str(AUDIT), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  audit sheet: {AUDIT.relative_to(REPO)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    src_root, dst_root = HF_LEROBOT_HOME / REPO_ID, HF_LEROBOT_HOME / OUT_ID
    marks = approved_marks()
    if dst_root.exists():
        print(f"{OUT_ID} already built at {dst_root}")
    else:
        build(src_root, dst_root, marks)
        audit(dst_root, marks)
        print(f"OK: {OUT_ID} built at {dst_root}")
    if args.push:
        from lerobot.datasets import LeRobotDataset
        LeRobotDataset(OUT_ID, root=dst_root).push_to_hub(private=False)
        print(f"pushed: https://huggingface.co/datasets/{OUT_ID}")


if __name__ == "__main__":
    main()
