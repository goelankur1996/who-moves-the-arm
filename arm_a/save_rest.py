"""Save the folded rest pose that every run starts from and returns to.

Fold the arm to rest by hand with the follower's power unplugged, plug the power back in, then run:
    python -m arm_a.save_rest
"""
import json
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from arm_a.kin import ARM_JOINTS
from arm_a.rig import REST_FILE

arm = SO101Follower(SO101FollowerConfig(port="/dev/tty.usbmodem5B141148491", id="follower",
                                        disable_torque_on_disconnect=False, max_relative_target=10.0))
arm.connect()
obs = arm.get_observation()
q = [round(float(obs[f"{j}.pos"]), 1) for j in ARM_JOINTS]
arm.disconnect()
json.dump({"saved": time.strftime("%Y-%m-%d %H:%M:%S"), "joint_names": ARM_JOINTS, "joints_deg": q},
          open(REST_FILE, "w"), indent=2)
print(f"saved {REST_FILE}: " + ", ".join(f"{j} {v}" for j, v in zip(ARM_JOINTS, q)))
