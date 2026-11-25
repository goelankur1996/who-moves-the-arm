"""Calibration v3: scene-camera pixels -> robot frame, measured at the grasp centre.

v2 recorded the fixed fingertip pressed on the table. Trials use the point midway
between the open fingertips, hovering, and the 25 Nov spot check showed the arm
landing 10-30 mm short of where v2 said. v3 measures the point the trials use:
a ring tied at the middle of a taut thread between the open fingertips.

Steps:
1. Grabs one scene frame. Click the 9 table crosses (click order = cross number).
   u = undo, Enter = done. Keys go to the image window.
2. The follower's gripper opens to the trial opening and stays there whatever the
   leader's gripper does. Tie the thread and ring, then type how far the bottom of
   the ring hangs below the fingertips (mm).
3. Teleop: for each cross in click order, bring the ring down so its bottom just
   touches the cross, then press Enter.
4. Box (skip with --no-box): type the box height (mm). For each cross: put the box on
   it, click the cross on the box top in the new frame, then touch that cross with
   the ring and press Enter.
5. Bring the arm to rest and press Enter.

Output (v2 kept): calib/touch_calib_v3.json with the table homography in the same
format arm A loads, the table height under the grasp centre, and a camera
projection fitted to all points (for a height-aware table_point later).

Usage:
  python calib/touch_calibrate_v3.py
"""
import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_a.kin import ARM_JOINTS, Kinematics  # noqa: E402
from arm_a.rig import GRASP_OFFSET_MM, GRIPPER_OPEN  # noqa: E402
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig  # noqa: E402

BLUE = (235, 99, 37)


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


def click_points(frame, n, title):
    pts = []
    win = f"{title} (u = undo, Enter = done)"

    def draw():
        img = frame.copy()
        for i, (x, y) in enumerate(pts):
            cv2.drawMarker(img, (int(x), int(y)), BLUE, cv2.MARKER_CROSS, 16, 2)
            cv2.putText(img, str(i + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((float(x), float(y)))
            print(f"  clicked {len(pts)}/{n}" + ("  -> press Enter in the image window" if len(pts) == n else ""))
            draw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    draw()
    print(f"\n{title}: click in the image window. Keys (u, Enter) go to that window, not the terminal.")
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


def ask_number(prompt):
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  type a number, e.g. 12")


class Teleop:
    """Follower follows the leader, except the gripper, which stays at the trial opening."""

    def __init__(self, follower, leader):
        self.f, self.l = follower, leader

    def step(self):
        a = self.l.get_action()
        a["gripper.pos"] = GRIPPER_OPEN
        self.f.send_action(a)

    def hold_gripper_open(self):
        obs = self.f.get_observation()
        a = {f"{j}.pos": float(obs[f"{j}.pos"]) for j in ARM_JOINTS}
        a["gripper.pos"] = GRIPPER_OPEN
        for _ in range(45):  # about 1.5 s at 30 Hz
            self.f.send_action(a)
            time.sleep(1 / 30)

    def until_enter(self, message):
        """Teleop until Enter; returns the follower's arm joints at that moment."""
        done = threading.Event()
        threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
        print(message)
        while not done.is_set():
            self.step()
            time.sleep(1 / 30)
        obs = self.f.get_observation()
        return [float(obs[f"{j}.pos"]) for j in ARM_JOINTS]


def fit_projection(xyz, uv):
    """3x4 camera projection (DLT, normalised) from robot-frame points (mm) to pixels."""
    xyz, uv = np.asarray(xyz, float), np.asarray(uv, float)
    m3, s3 = xyz.mean(0), np.sqrt(3) / np.mean(np.linalg.norm(xyz - xyz.mean(0), axis=1))
    m2, s2 = uv.mean(0), np.sqrt(2) / np.mean(np.linalg.norm(uv - uv.mean(0), axis=1))
    T3 = np.diag([s3, s3, s3, 1.0]); T3[:3, 3] = -s3 * m3
    T2 = np.diag([s2, s2, 1.0]); T2[:2, 2] = -s2 * m2
    X = (T3 @ np.c_[xyz, np.ones(len(xyz))].T).T
    x = (T2 @ np.c_[uv, np.ones(len(uv))].T).T
    rows = []
    for Xi, (u, v, _) in zip(X, x):
        rows.append(np.r_[Xi, np.zeros(4), -u * Xi])
        rows.append(np.r_[np.zeros(4), Xi, -v * Xi])
    _, _, Vt = np.linalg.svd(np.array(rows))
    P = np.linalg.inv(T2) @ Vt[-1].reshape(3, 4) @ T3
    return P / np.linalg.norm(P[2, :3])


def project(P, xyz):
    p = P @ np.r_[xyz, 1.0]
    return p[:2] / p[2]


def ray_at_height(P, u, v, z):
    """Robot-frame (x, y) where the camera ray through pixel (u, v) crosses height z (mm)."""
    A = np.array([[P[0, 0] - u * P[2, 0], P[0, 1] - u * P[2, 1]],
                  [P[1, 0] - v * P[2, 0], P[1, 1] - v * P[2, 1]]])
    b = -np.array([P[0, 2] * z + P[0, 3] - u * (P[2, 2] * z + P[2, 3]),
                   P[1, 2] * z + P[1, 3] - v * (P[2, 2] * z + P[2, 3])])
    return np.linalg.solve(A, b)


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
    ap.add_argument("--urdf", default=os.path.expanduser("~/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--follower-port", default="/dev/tty.usbmodem5B141148491")
    ap.add_argument("--leader-port", default="/dev/tty.usbmodem5B3D0440441")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--no-box", action="store_true")
    ap.add_argument("--out", default="calib/touch_calib_v3.json")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; pass --out with a new name")

    kin = Kinematics(args.urdf)
    stem = args.out.replace(".json", "")

    # 1. table crosses
    frame = grab_frame(args.camera)
    cv2.imwrite(f"{stem}_frame.jpg", frame)
    table_px = click_points(frame, args.n, f"Click the {args.n} table crosses")

    follower = SO101Follower(SO101FollowerConfig(
        port=args.follower_port, id="follower", disable_torque_on_disconnect=False, max_relative_target=10.0))
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id="leader"))
    follower.connect()
    leader.connect()
    tele = Teleop(follower, leader)
    box_px, box_q, box_h = [], [], None
    try:
        # 2. open the gripper, tie the thread
        tele.hold_gripper_open()
        input("\nGripper open. Tie the thread and ring between the fingertips, then press Enter.")
        drop = ask_number("How far does the bottom of the ring hang below the fingertips (mm)? ")

        # 3. table crosses with the ring
        table_q = []
        for k in range(1, args.n + 1):
            table_q.append(tele.until_enter(
                f"\nCross {k} of {args.n}: bring the ring down until its bottom just touches the cross, then press Enter."))
            print(f"  recorded cross {k}")

        # 4. box crosses
        if not args.no_box:
            box_h = ask_number("\nBox height (mm)? ")
            for k in range(1, args.n + 1):
                tele.until_enter(f"\nBox {k} of {args.n}: move the arm clear of the box, put the box on cross {k}, "
                                 f"then press Enter to take a picture.")
                f = grab_frame(args.camera)
                cv2.imwrite(f"{stem}_box{k}.jpg", f)
                box_px += click_points(f, 1, f"Click the cross on the box top (box {k})")
                box_q.append(tele.until_enter(
                    f"Touch the box-top cross with the bottom of the ring, then press Enter."))
                print(f"  recorded box {k}")

        tele.until_enter("\nAll done. Bring the arm to its rest pose, then press Enter.")
    finally:
        leader.disconnect()
        follower.disconnect()  # torque stays on: disable_torque_on_disconnect=False

    # the point that touched: grasp centre, minus the ring's hang (straight down)
    def ring_xyz(q):
        return kin.point(np.array(q), GRASP_OFFSET_MM) - np.array([0.0, 0.0, drop])

    t_xyz = np.array([ring_xyz(q) for q in table_q])
    t_px = np.array(table_px, float)
    H, _ = cv2.findHomography(t_px, t_xyz[:, :2], 0)
    errs = leave_one_out(t_px, t_xyz[:, :2])
    table_z = float(np.median(t_xyz[:, 2]))

    print("\ncross  pixel (u, v)      x, y, z (mm)                leave-one-out (mm)")
    for i, (p, t, e) in enumerate(zip(table_px, t_xyz, errs)):
        print(f"{i + 1:>5}  ({p[0]:5.0f}, {p[1]:5.0f})   ({t[0]:7.1f}, {t[1]:7.1f}, {t[2]:6.1f})   {e:6.1f}")
    mean_err, z_spread = float(np.mean(errs)), float(np.ptp(t_xyz[:, 2]))
    print(f"\ntable: mean leave-one-out {mean_err:.1f} mm, max {max(errs):.1f} mm, height spread {z_spread:.1f} mm")

    out = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "method": "ring at the grasp centre, hovering",
        "camera_index": args.camera, "frame_size": [frame.shape[1], frame.shape[0]],
        "ring_drop_mm": drop, "grasp_offset_mm": list(GRASP_OFFSET_MM), "joint_names": ARM_JOINTS,
        "image_points": table_px, "joints_deg": table_q, "table_xyz_mm": t_xyz.tolist(),
        "H_image_to_table_mm": H.tolist(), "leave_one_out_mm": errs, "mean_error_mm": mean_err,
        "z_spread_mm": z_spread, "table_z_mm": table_z, "g0_pass": mean_err < 10,
    }

    if box_q:
        b_xyz = np.array([ring_xyz(q) for q in box_q])
        b_px = np.array(box_px, float)
        arm_h = b_xyz[:, 2] - table_z
        print(f"\nbox: you measured {box_h:.1f} mm; the arm reads {arm_h.mean():.1f} mm "
              f"(spread {np.ptp(arm_h):.1f} mm)")
        P = fit_projection(np.r_[t_xyz, b_xyz], np.r_[t_px, b_px])
        reproj = [float(np.linalg.norm(project(P, X) - p)) for X, p in zip(np.r_[t_xyz, b_xyz], np.r_[t_px, b_px])]
        at_h = [float(np.linalg.norm(ray_at_height(P, u, v, z) - X[:2]))
                for (u, v), X, z in zip(b_px, b_xyz, b_xyz[:, 2])]
        print(f"camera fit: mean reprojection {np.mean(reproj):.1f} px; box crosses located from their pixel "
              f"and height: mean {np.mean(at_h):.1f} mm, max {max(at_h):.1f} mm")
        out.update({"box_height_mm": box_h, "box_image_points": box_px, "box_joints_deg": box_q,
                    "box_xyz_mm": b_xyz.tolist(), "P_robot_to_image": P.tolist(),
                    "reprojection_px": reproj, "box_xy_error_mm": at_h})

    print("G0 PASS" if mean_err < 10 else "G0 FAIL")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
