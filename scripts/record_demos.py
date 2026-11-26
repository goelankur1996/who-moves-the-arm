"""Record the demonstrations: place each demo layout with the ghost overlay, then teleoperate.

    python scripts/record_demos.py                  # continues at the first demo layout not yet saved
    python scripts/record_demos.py --start D-21

One window, one camera owner (this script), so placing and recording need no restarts.
The follower follows the leader the whole time, so placing the pieces is also the reset.

  PLACE   live scene with the layout's ghost. Leader at rest, pieces on their outlines.
          r or space: start recording (after a --countdown-s pause)    a: blend    n / p: next / previous layout    q: quit
  RECORD  do the task with the leader. Stops by itself after --episode-s seconds.
          e: end now (piece in, arm back near rest)    x: throw this take away
  KEEP?   s: save and go to the next layout    x: discard and redo this layout

Quit only from PLACE, with the arm at rest. The dataset is finalised on quit (also on Ctrl-C).
Each saved episode is logged in results/demos.csv with its layout id.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghost import LAYOUTS, RIMS, outlines  # noqa: E402

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.datasets import (  # noqa: E402
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors  # noqa: E402
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig  # noqa: E402
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR  # noqa: E402
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data  # noqa: E402

FPS = 30
REPO_ID = "ankurcw/wmta_demos_v1"
LOG = "results/demos.csv"
LOG_FIELDS = ["episode_index", "layout_id", "condition", "instruction", "frames", "duration_s",
              "ended_by", "slow_frames", "takes", "saved_at"]
WIN = "record demos"
RERUN_EVERY = 3  # log to Rerun on every 3rd step (10 Hz) so the 30 Hz loop keeps its pace
BLUE, RED, BLACK, WHITE = (255, 90, 0), (0, 0, 220), (0, 0, 0), (255, 255, 255)


class Rig:
    """Follower with both cameras, leader, and LeRobot's default (identity) processors."""

    def __init__(self, args):
        cams = {
            # MJPG first, then 640x480, as arm A's rig opens them (it got 640x480 from both).
            # LeRobot may warn that MJPG was not set; that warning is harmless.
            "scene": OpenCVCameraConfig(index_or_path=args.scene_cam, fps=FPS, width=640, height=480, fourcc="MJPG"),
            "wrist": OpenCVCameraConfig(index_or_path=args.wrist_cam, fps=FPS, width=640, height=480, fourcc="MJPG"),
        }
        self.robot = SO101Follower(SO101FollowerConfig(port=args.follower_port, id="follower", cameras=cams))
        self.leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id="leader"))
        self.t_proc, self.r_proc, self.o_proc = make_default_processors()
        self.rerun, self.n = args.rerun, 0
        if self.rerun:
            init_rerun(session_name="record_demos")

    def features(self):
        return combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=self.t_proc, initial_features=create_initial_features(action=self.robot.action_features),
                use_videos=True),
            aggregate_pipeline_dataset_features(
                pipeline=self.o_proc, initial_features=create_initial_features(observation=self.robot.observation_features),
                use_videos=True),
        )

    def step(self):
        """One teleop step, as in LeRobot's record_loop. Returns the processed observation and action.

        A camera frame older than 0.5 s makes LeRobot raise TimeoutError (it happens while an episode
        is being encoded). Wait and retry for up to ~2 s; self.stalled tells the caller it happened.
        """
        self.stalled = False
        for attempt in range(40):
            try:
                obs = self.robot.get_observation()
                break
            except TimeoutError:
                self.stalled = True
                time.sleep(0.05)
        else:
            raise TimeoutError("a camera has sent no new frame for about 2 s")
        obs_p = self.o_proc(obs)
        act_p = self.t_proc((self.leader.get_action(), obs))
        self.robot.send_action(self.r_proc((act_p, obs)))
        self.n += 1
        if self.rerun and self.n % RERUN_EVERY == 0:  # both cameras and the joints, JPEG, at 10 Hz
            log_rerun_data(observation=obs_p, action=act_p, compress_images=True)
        return obs, obs_p, act_p


def banner(img, text, colour=WHITE, fg=BLACK):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), colour, -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, fg, 1, cv2.LINE_AA)


def ghost_view(live_bgr, ref, cs, w):
    view = cv2.addWeighted(ref, w, live_bgr, 1 - w, 0)
    cv2.drawContours(view, cs, -1, BLUE, 1, cv2.LINE_AA)
    for u, v, a, b in RIMS.values():
        cv2.ellipse(view, (round(u), round(v)), (round(a), round(b)), 0, 0, 360, BLUE, 1, cv2.LINE_AA)
    return view


def show(view, big=True):
    cv2.imshow(WIN, cv2.resize(view, (1280, 960)) if big else view)
    return cv2.waitKey(1) & 0xFF


def saved_layouts(log):
    if not os.path.exists(log):
        return []
    with open(log) as f:
        return [r["layout_id"] for r in csv.DictReader(f)]


def log_episode(row, log):
    new = not os.path.exists(log)
    with open(log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def open_dataset(rig, repo_id, streaming):
    # Streaming (default): video is encoded while you record, so saving is near-instant. With the
    # default 30-frame queue it dropped frames on 27 Nov; a 300-frame queue (10 s) gives the encoder
    # room to catch up, and record() throws away any take in which a frame is still dropped.
    # Off: each episode is encoded after you press s (about a minute's wait per demo).
    # LeRobot's batch encoding is not used: it crashes on a resumed dataset in this version.
    if streaming:
        enc = dict(streaming_encoding=True, encoder_threads=2, encoder_queue_maxsize=300)
    else:
        enc = dict(streaming_encoding=False)
    root = HF_LEROBOT_HOME / repo_id
    info = root / "meta" / "info.json"
    if info.exists() and json.load(open(info)).get("total_episodes", 0) == 0:
        print(f"removing the empty dataset left by an earlier run: {root}")
        shutil.rmtree(root)
    if info.exists():
        print(f"resuming {root}")
        return LeRobotDataset.resume(repo_id, root=root, image_writer_threads=8, **enc)
    print(f"creating {root}")
    return LeRobotDataset.create(repo_id, FPS, root=root, features=rig.features(), robot_type=rig.robot.name,
                                 use_videos=True, image_writer_threads=8, **enc)


def place(rig, L, k, n, ref, cs):
    """PLACE phase. Returns 'r', 'q', 'n' or 'p'."""
    modes, mode = [0.5, 0.0, 1.0], 0
    while True:
        t0 = time.perf_counter()
        obs, _, _ = rig.step()
        view = ghost_view(cv2.cvtColor(obs["scene"], cv2.COLOR_RGB2BGR), ref, cs, modes[mode])
        banner(view, f"PLACE {L['layout_id']} {k + 1}/{n}: {L['instruction']}   r: record  q: quit")
        key = show(view)
        if key in (ord("r"), ord(" ")):
            return "r"
        if key in (ord("q"), 27):
            return "q"
        if key in (ord("n"), ord("p")):
            return chr(key)
        if key == ord("a"):
            mode = (mode + 1) % len(modes)
        precise_sleep(max(1 / FPS - (time.perf_counter() - t0), 0))


def countdown(rig, L, seconds):
    """A few seconds to get a hand on the leader; the arm follows it, nothing is recorded."""
    t_end = time.perf_counter() + seconds
    while (left := t_end - time.perf_counter()) > 0:
        t0 = time.perf_counter()
        obs, _, _ = rig.step()
        view = cv2.cvtColor(obs["scene"], cv2.COLOR_RGB2BGR)
        banner(view, f"GET READY {L['layout_id']}: recording starts in {left:.0f} s", (0, 200, 255))
        show(view, big=False)
        precise_sleep(max(1 / FPS - (time.perf_counter() - t0), 0))


def dropped_frames(dataset):
    """Frames the streaming video encoder has dropped in the current episode (0 when not streaming)."""
    enc = getattr(dataset.writer, "_streaming_encoder", None)
    return sum(getattr(enc, "_dropped_frames", {}).values()) if enc is not None else 0


def record(rig, dataset, L, episode_s):
    """RECORD phase. Returns (ended_by, frames, duration_s, slow_frames); ended_by is 'key', 'time' or 'discard'."""
    frames, slow, t_start = 0, 0, time.perf_counter()
    while True:
        t0 = time.perf_counter()
        elapsed = t0 - t_start
        if elapsed >= episode_s:
            return "time", frames, round(elapsed, 2), slow
        obs, obs_p, act_p = rig.step()
        if rig.stalled:  # a gap of 0.5 s or more inside the episode: this take is not usable
            print(f"{L['layout_id']}: a camera stalled during recording; take thrown away, redo it")
            return "discard", frames, round(time.perf_counter() - t_start, 2), slow
        frame = {**build_dataset_frame(dataset.features, obs_p, prefix=OBS_STR),
                 **build_dataset_frame(dataset.features, act_p, prefix=ACTION),
                 "task": L["instruction"]}
        dataset.add_frame(frame)  # the first add_frame starts the encoder and resets its drop count
        frames += 1
        if dropped_frames(dataset):  # the video would be shorter than the joint data
            print(f"{L['layout_id']}: the video encoder dropped a frame; take thrown away, redo it")
            return "discard", frames, round(time.perf_counter() - t_start, 2), slow
        key = 255
        if frames % 3 == 0:  # draw at 10 Hz to keep the loop at 30 Hz
            view = cv2.cvtColor(obs["scene"], cv2.COLOR_RGB2BGR)
            banner(view, f"REC {L['layout_id']} {elapsed:4.1f}/{episode_s:.0f} s: {L['instruction']}   e: end  x: discard",
                   RED, WHITE)
            key = show(view, big=False)
        if key == ord("e"):
            return "key", frames, round(time.perf_counter() - t_start, 2), slow
        if key == ord("x"):
            return "discard", frames, round(time.perf_counter() - t_start, 2), slow
        dt = time.perf_counter() - t0
        slow += dt > 1.5 / FPS
        precise_sleep(max(1 / FPS - dt, 0))


def keep(rig, L, info):
    """KEEP? phase; the arm still follows the leader. Returns 's' or 'x'."""
    while True:
        t0 = time.perf_counter()
        obs, _, _ = rig.step()
        view = cv2.cvtColor(obs["scene"], cv2.COLOR_RGB2BGR)
        banner(view, f"KEEP {L['layout_id']}? {info}   s: save  x: redo")
        key = show(view)
        if key in (ord("s"), ord("x")):
            return chr(key)
        precise_sleep(max(1 / FPS - (time.perf_counter() - t0), 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="demo layout id to start at, e.g. D-21 (default: first one not yet saved)")
    ap.add_argument("--only", help="comma-separated layout ids to record again, e.g. D-28,D-31; then stop")
    ap.add_argument("--episode-s", type=float, default=45.0)
    ap.add_argument("--countdown-s", type=float, default=2.0, help="pause after r before recording starts")
    ap.add_argument("--no-rerun", dest="rerun", action="store_false", help="do not open the Rerun viewer")
    ap.add_argument("--no-streaming", dest="streaming", action="store_false",
                    help="encode video after each episode (about a minute per save) instead of while recording")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--log", default=LOG, help="CSV that lists the saved episodes of --repo-id")
    ap.add_argument("--follower-port", default="/dev/tty.usbmodem5B141148491")
    ap.add_argument("--leader-port", default="/dev/tty.usbmodem5B3D0440441")
    ap.add_argument("--scene-cam", type=int, default=0)
    ap.add_argument("--wrist-cam", type=int, default=1)
    args = ap.parse_args()

    data = json.load(open(LAYOUTS))
    demos = [L for L in data["layouts"] if L["set"] == "demo"]
    base = cv2.imread(data["meta"]["base_image"])
    done = saved_layouts(args.log)
    if args.only:  # re-record just these layouts, in this order, even though they were saved before
        want = [x.strip() for x in args.only.split(",")]
        by_id = {L["layout_id"]: L for L in demos}
        demos = [by_id[x] for x in want]
        k = 0
    elif args.start:
        k = next(j for j, L in enumerate(demos) if L["layout_id"] == args.start)
    else:
        k = next((j for j, L in enumerate(demos) if L["layout_id"] not in done), len(demos))
    if k >= len(demos):
        print("all demo layouts are saved")
        return

    rig = Rig(args)
    rig.robot.connect()  # connect first, so a failed connect leaves no dataset behind
    rig.leader.connect()
    takes = 0
    try:
        dataset = open_dataset(rig, args.repo_id, args.streaming)
        if dataset.num_episodes != len(done):
            print(f"WARNING: the dataset has {dataset.num_episodes} episodes but {args.log} lists {len(done)}")
        with VideoEncodingManager(dataset):
            while k < len(demos):
                L = demos[k]
                ref = cv2.imread(L["image"])
                cs = outlines(ref, base)
                if L["layout_id"] in done:
                    print(f"note: {L['layout_id']} was saved before; saving it again adds a second episode")
                act = place(rig, L, k, len(demos), ref, cs)
                if act == "q":
                    break
                if act in ("n", "p"):
                    k = max(0, min(len(demos) - 1, k + (1 if act == "n" else -1)))
                    takes = 0
                    continue
                takes += 1
                countdown(rig, L, args.countdown_s)
                ended, frames, dur, slow = record(rig, dataset, L, args.episode_s)
                if ended != "discard":
                    info = f"{dur:.1f} s, {frames} frames" + (f", {slow} slow" if slow else "")
                    ended = ended if keep(rig, L, info) == "s" else "discard"
                if ended == "discard":
                    dataset.clear_episode_buffer()
                    print(f"{L['layout_id']}: take {takes} discarded")
                    continue
                dataset.save_episode()
                log_episode({"episode_index": dataset.num_episodes - 1, "layout_id": L["layout_id"],
                             "condition": L["condition"], "instruction": L["instruction"], "frames": frames,
                             "duration_s": dur, "ended_by": ended, "slow_frames": slow, "takes": takes,
                             "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, args.log)
                done.append(L["layout_id"])
                print(f"{L['layout_id']}: saved as episode {dataset.num_episodes - 1} ({dur:.1f} s, take {takes})")
                k += 1
                takes = 0
    finally:
        cv2.destroyAllWindows()
        if rig.robot.is_connected:
            rig.robot.disconnect()
        if rig.leader.is_connected:
            rig.leader.disconnect()
    print(f"done; episodes are in {HF_LEROBOT_HOME / args.repo_id}")


if __name__ == "__main__":
    main()
