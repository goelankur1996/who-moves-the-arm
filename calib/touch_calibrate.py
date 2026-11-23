"""9-point touch calibration: scene-camera pixels -> table coordinates in the robot frame.

1. Grabs one scene frame. Click the centre of each tape mark (any order; the
   click order is the mark number). u = undo last click, Enter = done.
2. Teleop: move the follower with the leader so the closed fingertips touch
   mark 1, gripper pointing straight down, then press Enter in the terminal.
   Repeat for every mark in click order.
3. Forward kinematics on the SO-101 URDF gives each touch point in the robot
   frame. A homography is fitted pixel -> (x, y) mm and checked leave-one-out.

Gate G0: mean leave-one-out error under 10 mm.

Usage:
  python calib/touch_calibrate.py --urdf ~/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
"""
import argparse
import json
import os
import threading
import time

import cv2
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def grab_frame(index, w=640, h=480):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(20):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"camera {index} returned no frame")
    return frame


def click_points(frame, n):
    pts = []
    win = "click mark centres (u = undo, Enter = done)"

    def draw():
        img = frame.copy()
        for i, (x, y) in enumerate(pts):
            cv2.drawMarker(img, (int(x), int(y)), (235, 99, 37), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(img, str(i + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 99, 37), 2)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((float(x), float(y)))
            print(f"  clicked {len(pts)}/{n}" + ("  -> now press Enter or Space in the image window" if len(pts) == n else ""))
            draw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    draw()
    print(f"\nClick the {n} mark centres in the image window. Keys (u, Enter) go to that window, not the terminal.")
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
            print(f"  undone, {len(pts)}/{n}")
            draw()
        if k in (13, 10, 32) and len(pts) == n:
            break
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    return pts


def touch_points(follower, leader, n):
    """Teleop the follower; each Enter records the follower's joint angles."""
    joints, pressed = [], threading.Event()

    def wait_enter():
        while True:
            input()
            pressed.set()

    threading.Thread(target=wait_enter, daemon=True).start()
    print(f"\nTouch mark 1 of {n} with the closed fingertips, gripper straight down, then press Enter.")
    while len(joints) < n:
        follower.send_action(leader.get_action())
        if pressed.is_set():
            pressed.clear()
            obs = follower.get_observation()
            joints.append([float(obs[f"{j}.pos"]) for j in ARM_JOINTS])
            print(f"  mark {len(joints)} recorded: " + ", ".join(f"{v:.1f}" for v in joints[-1]))
            if len(joints) < n:
                print(f"Touch mark {len(joints) + 1} of {n}, then press Enter.")
        time.sleep(1 / 30)
    print("\nAll marks recorded. Bring the arm to its rest pose, then press Enter.")
    while not pressed.is_set():
        follower.send_action(leader.get_action())
        time.sleep(1 / 30)
    return joints


def leave_one_out(img_pts, tab_pts):
    errs = []
    for i in range(len(img_pts)):
        keep = [j for j in range(len(img_pts)) if j != i]
        H, _ = cv2.findHomography(img_pts[keep], tab_pts[keep], 0)
        p = cv2.perspectiveTransform(img_pts[i:i + 1].reshape(1, 1, 2), H).reshape(2)
        errs.append(float(np.linalg.norm(p - tab_pts[i])))
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--follower-port", default="/dev/tty.usbmodem5B141148491")
    ap.add_argument("--leader-port", default="/dev/tty.usbmodem5B3D0440441")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--out", default="calib/touch_calib_v1.json")
    args = ap.parse_args()

    kin = RobotKinematics(os.path.expanduser(args.urdf), "gripper_frame_link", ARM_JOINTS)

    frame = grab_frame(args.camera)
    cv2.imwrite(args.out.replace(".json", "_frame.jpg"), frame)
    img_pts = click_points(frame, args.n)

    follower = SO101Follower(SO101FollowerConfig(
        port=args.follower_port, id="follower", disable_torque_on_disconnect=False, max_relative_target=10.0))
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id="leader"))
    follower.connect()
    leader.connect()
    try:
        joints = touch_points(follower, leader, args.n)
    finally:
        leader.disconnect()
        follower.disconnect()  # torque stays on: disable_torque_on_disconnect=False

    xyz = np.array([kin.forward_kinematics(np.array(q))[:3, 3] * 1000 for q in joints])  # mm
    img = np.array(img_pts, dtype=np.float64)
    tab = xyz[:, :2].astype(np.float64)
    H, _ = cv2.findHomography(img, tab, 0)
    errs = leave_one_out(img, tab)

    print("\nmark   pixel (u, v)      table x, y, z (mm)        leave-one-out error (mm)")
    for i, (p, t, e) in enumerate(zip(img_pts, xyz, errs)):
        print(f"{i + 1:>4}   ({p[0]:5.0f}, {p[1]:5.0f})   ({t[0]:7.1f}, {t[1]:7.1f}, {t[2]:6.1f})   {e:6.1f}")
    mean_err, z_spread = float(np.mean(errs)), float(np.ptp(xyz[:, 2]))
    print(f"\nmean error {mean_err:.1f} mm, max {max(errs):.1f} mm, fingertip height spread {z_spread:.1f} mm")
    print("G0 PASS" if mean_err < 10 else "G0 FAIL: arm A falls back to the taught grid")

    with open(args.out, "w") as f:
        json.dump({
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_index": args.camera, "frame_size": [frame.shape[1], frame.shape[0]],
            "image_points": img_pts, "joints_deg": joints, "joint_names": ARM_JOINTS,
            "table_xyz_mm": xyz.tolist(), "H_image_to_table_mm": H.tolist(),
            "leave_one_out_mm": errs, "mean_error_mm": mean_err, "z_spread_mm": z_spread,
            "g0_pass": mean_err < 10,
        }, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
