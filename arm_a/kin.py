"""Kinematics for arm A: fingertip position <-> joint angles on the SO-101 URDF.

Robot frame (URDF base): x forward from the base, y to the robot's left, z up, mm.
The SO-101 has 5 arm joints. wrist_roll is set directly from the requested yaw; the other
four place the fingertip and, where the joint limits allow, point the gripper straight down.
"""
from __future__ import annotations

import math

import numpy as np
import placo

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP_FRAME = "gripper_frame_link"
ROLL_NEUTRAL_DEG = -86.0  # wrist_roll used in the 23 Nov touch calibration (gripper square to the arm)
PAN_ORIGIN_XY_MM = (38.8, 0.0)  # shoulder_pan axis in the base frame (URDF)


class Kinematics:
    def __init__(self, urdf_path: str):
        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        for dof in ("wrist_roll", "gripper"):
            self.solver.mask_dof(dof)
        self.solver.enable_joint_limits(True)
        self.pos_task = self.solver.add_position_task(TIP_FRAME, np.zeros(3))
        self.pos_task.configure("tip_position", "soft", 1.0)
        self.axis_task = self.solver.add_axisalign_task(TIP_FRAME, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]))

    def _set(self, q_deg):
        for name, v in zip(ARM_JOINTS, q_deg):
            self.robot.set_joint(name, math.radians(v))
        self.robot.update_kinematics()

    def fk(self, q_deg) -> np.ndarray:
        """4x4 fingertip pose, translation in mm."""
        self._set(q_deg)
        T = np.array(self.robot.get_T_world_frame(TIP_FRAME), dtype=float)
        T[:3, 3] *= 1000.0
        return T

    @staticmethod
    def tilt_deg(T) -> float:
        """Angle between the gripper approach axis and straight down."""
        return float(math.degrees(math.acos(float(np.clip(-T[2, 2], -1.0, 1.0)))))

    def ik(self, q_start_deg, xyz_mm, yaw_deg=0.0, tol_mm=1.0):
        """Returns (q_deg, position_error_mm, tilt_deg). Tries hard to keep the gripper vertical,
        then relaxes that where the joint limits make vertical impossible at this point."""
        roll = float(np.clip(ROLL_NEUTRAL_DEG + yaw_deg, -175.0, 175.0))
        results = []
        for w in (1.0, 0.2, 0.05, 0.0):
            q, err, tilt = self._solve(q_start_deg, xyz_mm, roll, w)
            results.append((q, err, tilt))
            if err < tol_mm and tilt < 2.0:
                return q, err, tilt
        reached = [r for r in results if r[1] < tol_mm]
        return min(reached, key=lambda r: r[2]) if reached else min(results, key=lambda r: r[1])

    def _solve(self, q_start_deg, xyz_mm, roll_deg, axis_weight, iters=300, tol_mm=0.3):
        q0 = list(q_start_deg)
        q0[4] = roll_deg
        self._set(q0)
        self.pos_task.target_world = np.asarray(xyz_mm, float) / 1000.0
        self.axis_task.configure("down", "soft", max(axis_weight, 1e-9))
        for _ in range(iters):
            self.solver.solve(True)
            self.robot.update_kinematics()
            p = np.array(self.robot.get_T_world_frame(TIP_FRAME))[:3, 3] * 1000.0
            if np.linalg.norm(p - xyz_mm) < tol_mm:
                break
        q = [math.degrees(self.robot.get_joint(n)) for n in ARM_JOINTS]
        T = self.fk(q)
        return np.array(q), float(np.linalg.norm(T[:3, 3] - xyz_mm)), self.tilt_deg(T)
