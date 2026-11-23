"""The physical rig for arm A: follower arm, two cameras, calibration, kinematics.

The five agent tools are methods here: look, table_point, move_to, gripper, home.
Coordinates: x forward from the arm base, y to the robot's left, mm. Heights (z) are
millimetres above the table.

Start with the arm in its rest pose: connecting briefly disables torque (LeRobot's
configure step). On close the arm returns to that rest pose and keeps torque on.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from arm_a.kin import ARM_JOINTS, PAN_ORIGIN_XY_MM, ROLL_NEUTRAL_DEG, Kinematics

# Limits the agent cannot override.
MIN_HEIGHT_MM = 3.0          # fingertip never commanded lower than this above the table
MAX_HEIGHT_MM = 200.0
MIN_RADIUS_MM, MAX_RADIUS_MM = 100.0, 360.0
MAX_IK_ERROR_MM = 5.0
STEP_MM = 10.0               # Cartesian waypoint spacing
JOINT_SPEED_DEG_S = 60.0
CONTROL_HZ = 30

BOWL_HEIGHT_MM = 76.0        # bowls are 7.6 cm tall
TRAVEL_H_EMPTY = 100.0       # fingertips clear the bowl rims by ~2.5 cm
TRAVEL_H_HOLDING = 125.0     # a held piece hangs up to ~15 mm below the fingertips: ~3 cm clearance
LONG_MOVE_MM = 30.0          # horizontal moves longer than this go up, across, then down

GRIPPER_OPEN = 45.0          # 0 = closed, 100 = fully open (LeRobot RANGE_0_100)
GRIPPER_CLOSED = 0.0
HOLD_THRESHOLD = 5.0         # closed gripper reading: 0.9 empty, 14.5 on the orange cube (23 Nov, taped fingers)
# Per-step limit on how far a goal may be from the present position. The arm joints keep LeRobot's 10;
# the gripper gets no practical limit, because squeeze force comes from the gap between goal and position,
# and a limit of 10 capped that gap (and the grip) on 23 Nov.
MAX_STEP = {"shoulder_pan": 10.0, "shoulder_lift": 10.0, "elbow_flex": 10.0, "wrist_flex": 10.0,
            "wrist_roll": 10.0, "gripper": 100.0}

logging.getLogger().setLevel(logging.ERROR)  # LeRobot logs a warning on every clamped step


class Camera:
    """Keeps the latest frame from one camera in a background thread."""

    def __init__(self, index: int, w=640, h=480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.frame, self.t = None, 0.0
        self._stop = False
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()
        t0 = time.time()
        while self.frame is None and time.time() - t0 < 10:
            time.sleep(0.05)
        if self.frame is None:
            raise RuntimeError(f"camera {index}: no frames")

    def _run(self):
        while not self._stop:
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self.frame, self.t = f, time.time()

    def latest(self):
        with self._lock:
            return self.frame.copy(), time.time() - self.t

    def close(self):
        self._stop = True
        time.sleep(0.1)
        self.cap.release()


class Recorder:
    """Writes scene and wrist side by side to an mp4 at 15 fps."""

    def __init__(self, path, scene: Camera, wrist: Camera, fps=15):
        self.scene, self.wrist, self.fps = scene, wrist, fps
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 480))
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop:
            t = time.time()
            s, _ = self.scene.latest()
            w, _ = self.wrist.latest()
            self.writer.write(np.hstack([cv2.resize(s, (640, 480)), cv2.resize(w, (640, 480))]))
            time.sleep(max(0.0, 1 / self.fps - (time.time() - t)))

    def close(self):
        self._stop = True
        self.thread.join(timeout=2)
        self.writer.release()


@dataclass
class Calibration:
    H: np.ndarray
    table_z: float

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(np.array(d["H_image_to_table_mm"]), float(np.median(np.array(d["table_xyz_mm"])[:, 2])))

    def pixel_to_table(self, u, v):
        p = self.H @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])


class Rig:
    def __init__(self, urdf, calib_path, follower_port="/dev/tty.usbmodem5B141148491",
                 scene_index=0, wrist_index=1, log=print):
        self.log = log
        self.kin = Kinematics(urdf)
        self.calib = Calibration.load(calib_path)
        self.scene = Camera(scene_index)
        self.wrist = Camera(wrist_index)
        self.arm = SO101Follower(SO101FollowerConfig(
            port=follower_port, id="follower", disable_torque_on_disconnect=False, max_relative_target=dict(MAX_STEP)))
        self.arm.connect()
        self.rest_q = self._read_q()
        self.grip_target = self._read_gripper()
        self.holding = False

    # ---------- low level ----------
    def _obs(self):
        return self.arm.get_observation()

    def _read_q(self):
        o = self._obs()
        return np.array([o[f"{j}.pos"] for j in ARM_JOINTS], float)

    def _read_gripper(self):
        return float(self._obs()["gripper.pos"])

    def _send(self, q, grip):
        a = {f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, q)}
        a["gripper.pos"] = float(grip)
        self.arm.send_action(a)

    def _drive_joints(self, q_target, settle_s=0.4, timeout_s=8.0):
        """Stream interpolated joint targets at CONTROL_HZ, then wait to settle."""
        q0 = self._read_q()
        span = float(np.max(np.abs(q_target - q0)))
        n = max(1, int(math.ceil(span / JOINT_SPEED_DEG_S * CONTROL_HZ)))
        for i in range(1, n + 1):
            self._send(q0 + (q_target - q0) * i / n, self.grip_target)
            time.sleep(1 / CONTROL_HZ)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            self._send(q_target, self.grip_target)
            time.sleep(1 / CONTROL_HZ)
            if np.max(np.abs(self._read_q() - q_target)) < 1.5 or time.time() - t0 > settle_s * 3:
                break
        time.sleep(settle_s)

    def tip(self):
        """Current fingertip (x, y, height above table) in mm and tilt from vertical in degrees."""
        T = self.kin.fk(self._read_q())
        return T[0, 3], T[1, 3], T[2, 3] - self.calib.table_z, self.kin.tilt_deg(T)

    # ---------- the five tools ----------
    def look(self):
        scene, s_age = self.scene.latest()
        wrist, w_age = self.wrist.latest()
        x, y, h, tilt = self.tip()
        g = self._read_gripper()
        state = {
            "fingertip_mm": {"x": round(x, 1), "y": round(y, 1), "height": round(h, 1)},
            "gripper_tilt_from_vertical_deg": round(tilt, 1),
            "gripper_opening_0_100": round(g, 1),
            "holding": self.holding,
            "frame_age_s": round(max(s_age, w_age), 2),
        }
        return scene, wrist, state

    def table_point(self, u, v):
        x, y = self.calib.pixel_to_table(u, v)
        r = math.hypot(x - PAN_ORIGIN_XY_MM[0], y - PAN_ORIGIN_XY_MM[1])
        return {"x": round(x, 1), "y": round(y, 1), "radius_from_base_mm": round(r, 1),
                "reachable": MIN_RADIUS_MM <= r <= MAX_RADIUS_MM}

    def move_to(self, x, y, height, yaw_deg=0.0):
        h = float(np.clip(height, MIN_HEIGHT_MM, MAX_HEIGHT_MM))
        r = math.hypot(x - PAN_ORIGIN_XY_MM[0], y - PAN_ORIGIN_XY_MM[1])
        if not MIN_RADIUS_MM <= r <= MAX_RADIUS_MM:
            return {"ok": False, "error": f"target is {r:.0f} mm from the base; allowed {MIN_RADIUS_MM:.0f}-{MAX_RADIUS_MM:.0f}"}
        tz = self.calib.table_z
        goal = np.array([x, y, h + tz])
        q = self._read_q()
        start = self.kin.fk(q)[:3, 3]
        # long horizontal moves go up to travel height, across, then down, so nothing sweeps into a bowl
        travel = (TRAVEL_H_HOLDING if self.holding else TRAVEL_H_EMPTY) + tz
        corners = [start]
        if np.hypot(*(goal[:2] - start[:2])) > LONG_MOVE_MM and min(start[2], goal[2]) < travel:
            corners += [np.array([start[0], start[1], max(start[2], travel)]),
                        np.array([goal[0], goal[1], max(goal[2], travel)])]
        corners.append(goal)
        yaw0 = q[4] - ROLL_NEUTRAL_DEG
        path = []
        for leg, (a, b) in enumerate(zip(corners[:-1], corners[1:])):
            n = max(1, int(math.ceil(np.linalg.norm(b - a) / STEP_MM)))
            for i in range(1, n + 1):
                wp = a + (b - a) * i / n
                yaw = yaw0 + (yaw_deg - yaw0) * i / n if leg == 0 else yaw_deg  # turn the wrist on the first leg
                q, err, _ = self.kin.ik(q, wp, yaw)
                if err > MAX_IK_ERROR_MM:
                    return {"ok": False, "error": f"no joint solution for a point on the way (error {err:.0f} mm); arm did not move"}
                path.append(q)
        q0 = self._read_q()
        for qi in path:  # follow the waypoints at the speed limit
            steps = max(1, int(math.ceil(float(np.max(np.abs(qi - q0))) / JOINT_SPEED_DEG_S * CONTROL_HZ)))
            for _ in range(steps):
                self._send(qi, self.grip_target)
                time.sleep(1 / CONTROL_HZ)
            q0 = qi
        self._drive_joints(path[-1])
        # correct for sag: aim past the target by the accumulated miss, up to three times
        aim = goal.copy()
        for _ in range(3):
            actual = self.kin.fk(self._read_q())[:3, 3]
            miss = goal - actual
            if np.linalg.norm(miss) < 2.0:
                break
            aim = aim + miss
            q_fix, err, _ = self.kin.ik(self._read_q(), aim, yaw_deg)
            if err > MAX_IK_ERROR_MM:
                break
            self._drive_joints(q_fix)
        x1, y1, h1, tilt = self.tip()
        result = {"ok": True, "reached_mm": {"x": round(x1, 1), "y": round(y1, 1), "height": round(h1, 1)},
                  "miss_mm": round(float(np.linalg.norm(np.array([x1, y1, h1]) - np.array([x, y, h]))), 1),
                  "gripper_tilt_from_vertical_deg": round(tilt, 1),
                  "height_clamped": h != height,
                  "went_via_travel_height": len(corners) > 2}
        if self.holding:  # did the piece slip out on the way?
            g = self._read_gripper()
            if g <= HOLD_THRESHOLD:
                self.holding = False
                result["dropped_on_the_way"] = True
            result["holding"] = self.holding
        return result

    def gripper(self, action, opening=None):
        if action == "open":
            self.grip_target = float(opening) if opening is not None else GRIPPER_OPEN
            self.holding = False
        elif action == "close":
            self.grip_target = GRIPPER_CLOSED
        else:
            return {"ok": False, "error": "action must be 'open' or 'close'"}
        q = self._read_q()
        t0 = time.time()
        while time.time() - t0 < 1.0:
            self._send(q, self.grip_target)
            time.sleep(1 / CONTROL_HZ)
        g = self._read_gripper()
        if action == "close":
            self.holding = g > HOLD_THRESHOLD
        return {"ok": True, "gripper_opening_0_100": round(g, 1), "holding": self.holding}

    def home(self):
        """Hover pose clear of the table: 200 mm out, centred, 120 mm up."""
        return self.move_to(200.0, 0.0, 120.0)

    # ---------- shutdown ----------
    def close(self):
        """Return to the rest pose recorded at start; torque stays on."""
        try:
            self.move_to(200.0, 0.0, 120.0)
            self._drive_joints(self.rest_q, settle_s=0.8)
        finally:
            self.arm.disconnect()
            self.scene.close()
            self.wrist.close()
