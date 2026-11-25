"""Check whether the arm lands where the camera calibration says a mark is.

For each container circle (not used in calibration v3), the arm:
1. turns the gripper so both fingertips are at the same height (wrist roll chosen by IK),
2. comes down until both fingertips rest on the table; you fine-tune with jog commands,
3. waits while you measure how far the point midway between the fingertips is from
   the circle's centre, and records your numbers.

Run with the arm at rest, nothing on the table:
    python -m arm_a.check_spot --calib calib/touch_calib_v3_fit.json
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import arm_a.rig as rigmod
from arm_a.rig import GRASP_OFFSET_MM, OPEN_GAP_MM, Rig

URDF = os.path.expanduser("~/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
CALIB = "calib/touch_calib_v2.json"
OUT = "calib/spot_check"

# Centres of the two container circles in the 25 Nov snapshot (scene camera, 640x480).
# Not used in calibration v3, so they test it fairly.
MARKS = {"mug": (287.7, 253.1), "tumbler": (495.0, 240.5)}

FIXED_TIP = np.zeros(3)                       # the URDF fingertip frame sits on the fixed finger
MOVING_TIP = np.array([-OPEN_GAP_MM, 0.0, 0.0])  # the moving jaw's tip at the trial opening

rigmod.MIN_HEIGHT_MM = -10.0  # this check alone may press the fingertips onto the table


def tip_heights(rig, q):
    """Heights of the fixed and moving fingertips above the table (mm)."""
    tz = rig.calib.table_z
    return rig.kin.point(q, FIXED_TIP)[2] - tz, rig.kin.point(q, MOVING_TIP)[2] - tz


def level_yaw(rig, x, y, h):
    """Gripper yaw (deg) at which both fingertips sit at the same height, closest to 0."""
    target = np.array([x, y, h + rig.calib.table_z])
    q0 = rig._read_q()
    best = None
    for yaw in np.arange(-90.0, 90.1, 3.0):
        try:
            q, err, _ = rig.kin.ik_point(q0, target, yaw, GRASP_OFFSET_MM)
        except RuntimeError:  # the solver finds no pose for this yaw
            continue
        if err > 3.0:
            continue
        a, b = tip_heights(rig, q)
        score = abs(a - b) + 0.02 * abs(yaw)
        if best is None or score < best[0]:
            best = (score, float(yaw), a - b)
    if best is None:
        return 0.0, float("nan")
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=CALIB)
    args = ap.parse_args()
    out = os.path.join(OUT, os.path.splitext(os.path.basename(args.calib))[0])
    os.makedirs(out, exist_ok=True)
    print(f"calibration: {args.calib}")
    rig = Rig(URDF, args.calib)
    results = []
    try:
        time.sleep(1.0)  # let the cameras warm up
        rig.gripper("open")
        for name in MARKS:
            p = rig.table_point(*MARKS[name])
            x, y = p["x"], p["y"]
            yaw, diff = level_yaw(rig, x, y, 0.0)
            print(f"\n{name}: target ({x}, {y}); yaw {yaw:.0f} deg puts the fingertips level (difference {diff:+.1f} mm)")
            rig.move_to(x, y, 15.0, yaw)
            h = 2.0
            rig.move_to(x, y, h, yaw)
            while True:
                a, b = tip_heights(rig, rig._read_q())
                cmd = input(f"  fingertips at {a:.1f} / {b:.1f} mm by the arm's reckoning. Both resting on the table?\n"
                            "  Enter = yes, measure | d = down 1 mm | u = up 1 mm | y+ / y- = turn 3 deg: ").strip()
                if cmd == "":
                    break
                if cmd == "d":
                    h -= 1.0
                elif cmd == "u":
                    h += 1.0
                elif cmd in ("y+", "y-"):
                    yaw += 3.0 if cmd == "y+" else -3.0
                else:
                    print("  type d, u, y+, y- or just Enter")
                    continue
                rig.move_to(x, y, h, yaw)
            scene, wrist, state = rig.look()
            tag = f"{name}_table"
            cv2.imwrite(f"{out}/{tag}_scene.jpg", scene)
            cv2.imwrite(f"{out}/{tag}_wrist.jpg", wrist)
            print(f"  arm reads grasp centre {state['grasp_centre_mm']}, tilt {state['gripper_tilt_from_vertical_deg']} deg")
            print("  Measure the point midway between the fingertips against the circle's centre.")
            radial = input("  Toward the base (+) or away from it (-), mm: ").strip()
            side = input("  To the robot's left (+) or right (-), mm: ").strip()
            results.append({"mark": name, "target": [x, y], "yaw_deg": yaw, "commanded_height": h,
                            "arm_reads": state["grasp_centre_mm"], "toward_base_mm": radial, "left_mm": side})
            rig.move_to(x, y, 40.0, yaw)
    finally:
        with open(f"{out}/results.json", "w") as f:
            json.dump({"calib": args.calib, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "stops": results}, f, indent=2)
        print(f"\nsaved {out}/results.json. Going home, then to rest.")
        rig.close()


if __name__ == "__main__":
    main()
