# Smoke test: Gazebo Harmonic + ROS 2 Jazzy

Checks that the sim setup works: one RGB-D camera and one 2D bounding-box camera look at a labelled box (a door stand-in, label `1`). Runs headless (no window).

```bash
bash sim/smoke_test/run_smoke.sh      # topics, one bounding box, ROS 2 image rates
bash sim/smoke_test/gpu_check.sh nvidia   # image rate with NVIDIA EGL forced
```

Expected: Gazebo topics `/rgbd/*` and `/boxes`, a box with `label: 1`, and data on `/rgbd/image`, `/rgbd/depth_image`, `/rgbd/points` in ROS 2.

Notes
- Needs `ros-jazzy-desktop` + `ros-jazzy-ros-gz` (Gazebo Sim 8 = Harmonic).
- Don't use `set -u` in scripts that source `/opt/ros/jazzy/setup.bash`.
- Deactivate Python venvs in ROS terminals (ROS 2 uses the system Python).
- On the hybrid laptop, headless EGL picks Mesa by default and warns `failed to create dri2 screen`. `export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json` forces the NVIDIA EGL driver.
