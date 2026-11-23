# Setup (MacBook)

## Python environment

LeRobot runs from `~/lerobot-env` (Python 3.12, created with uv, so it has no `pip`: use `uv pip`).

```
source ~/lerobot-env/bin/activate
```

## Kinematics (placo)

```
cd ~/lerobot && uv pip install -e ".[placo-dep]"
uv pip install "cmeel-urdfdom>=4,<5" "cmeel-tinyxml2>=10,<11"
python -c "import placo; print('placo ok')"
```

The second line is required. placo 0.9.15 is built against urdfdom 4 and tinyxml2 10, but the resolver installs newer versions, and `import placo` then fails with `Library not loaded: liburdfdom_sensor.4.0.dylib` or `libtinyxml2.10.dylib`. Re-run it after any reinstall of LeRobot's extras.

## SO-101 URDF

```
git clone --depth 1 --filter=blob:none --sparse https://github.com/TheRobotStudio/SO-ARM100 ~/SO-ARM100
cd ~/SO-ARM100 && git sparse-checkout set Simulation/SO101
```

URDF: `~/SO-ARM100/Simulation/SO101/so101_new_calib.urdf`, end-effector frame `gripper_frame_link`.

## Hardware on this Mac

- Follower: `/dev/tty.usbmodem5B141148491`, id `follower`
- Leader: `/dev/tty.usbmodem5B3D0440441`, id `leader`
- Cameras: scene = index 0 (MJPG), wrist = index 1
- LeRobot pinned at commit `8bf6056d`
