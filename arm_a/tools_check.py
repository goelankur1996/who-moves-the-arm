"""Hardware check of arm A's five tools, one step at a time. No Claude, no API key.

Start with the follower at rest and powered; the leader is not needed. Each move waits for
Enter, so you can stop before anything you don't like (type q then Enter to finish early;
the arm then returns to rest with torque on).

  python -m arm_a.tools_check
"""
import json
import os
from pathlib import Path

import cv2

from arm_a.rig import Rig


def step(msg):
    return input(f"\n{msg}  [Enter = go, q = finish] ").strip().lower() != "q"


def main():
    out = Path("runs/tools_check")
    out.mkdir(parents=True, exist_ok=True)
    rig = Rig(os.path.expanduser("~/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"), "calib/touch_calib_v2.json")
    try:
        scene, wrist, state = rig.look()
        cv2.imwrite(str(out / "scene.jpg"), scene)
        cv2.imwrite(str(out / "wrist.jpg"), wrist)
        print("look():", json.dumps(state), "-> images in runs/tools_check/")
        print("rest pose joints:", [round(v, 1) for v in rig.rest_q])
        print("table_point(320, 300):", rig.table_point(320, 300))

        if not step("gripper open"): return
        print(rig.gripper("open"))
        if not step("gripper close (nothing in the jaws)"): return
        print(rig.gripper("close"), " <- empty reading")
        if not step("gripper open, then put the orange cube between the jaws and hold it there"): return
        print(rig.gripper("open"))
        if not step("gripper close on the cube"): return
        print(rig.gripper("close"), " <- reading with the cube")
        print(rig.gripper("open"))

        if not step("home(): hover at x 200, y 0, height 120"): return
        print(rig.home())
        if not step("move_to(250, 0, 60): forward and lower"): return
        print(rig.move_to(250, 0, 60))
        if not step("move_to(250, 0, 25): down to 25 mm above the table"): return
        print(rig.move_to(250, 0, 25))
        if not step("move_to(250, 0, 25, yaw_deg=45): rotate the gripper 45 degrees"): return
        print(rig.move_to(250, 0, 25, yaw_deg=45))
        if not step("move_to(250, 80, 60): to the robot's left"): return
        print(rig.move_to(250, 80, 60))
        if not step("move_to(250, -60, 60): to the robot's right"): return
        print(rig.move_to(250, -60, 60))
        if not step("home()"): return
        print(rig.home())
    finally:
        print("\nreturning to rest (torque stays on)")
        rig.close()


if __name__ == "__main__":
    main()
