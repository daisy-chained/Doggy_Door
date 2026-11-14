# What makes the full patrol runs slow (Madelief 2026-11-14: "start with figuring out 3"). Helpers of sim/smoke_test/cost_check.py.
#   group(cmdline) -> which part of the run a process belongs to (gazebo / bridge / slam / nav2 / detector / door_classifier /
#                     door_tour / other)
#   usage(prev, now_ticks, dt) -> {group: cores}: CPU use per group between two samples ({pid: (group, ticks)}), ticks in clock
#                     ticks (utime + stime); a pid missing from prev (new process) is skipped.
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "smoke_test"))
import cost_check as cc  # noqa: E402

CLK = os.sysconf("SC_CLK_TCK")


def test_group():
    g = cc.group
    assert g("gz sim -s -r --headless-rendering world.sdf") == "gazebo"
    assert g("ruby /usr/bin/gz sim -s world.sdf") == "gazebo"
    assert g("/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge --ros-args") == "bridge"
    assert g("/opt/ros/jazzy/lib/ros_gz_image/image_bridge /robot/cam0") == "bridge"
    assert g("/opt/ros/jazzy/lib/slam_toolbox/async_slam_toolbox_node --ros-args") == "slam"
    for n in ("controller_server", "planner_server", "bt_navigator", "behavior_server", "map_server", "velocity_smoother",
              "lifecycle_manager", "smoother_server", "collision_monitor", "waypoint_follower", "amcl"):
        assert g(f"/opt/ros/jazzy/lib/nav2_x/{n} --ros-args") == "nav2", n
    assert g("/home/madi/git/Doggy_Door/train/.venv/bin/python /home/madi/git/Doggy_Door/train/detect_server.py runs/x") == "detector"
    assert g("python3 /home/madi/git/Doggy_Door/train/props_server.py runs/y") == "door_classifier"
    assert g("python3 sim/building/door_tour.py sim/building/out/bld --patrol") == "door_tour"
    assert g("/opt/ros/jazzy/lib/robot_state_publisher/robot_state_publisher") == "other"


def test_usage():
    prev = {1: ("gazebo", 100), 2: ("slam", 50), 3: ("gazebo", 0)}
    now = {1: ("gazebo", 100 + 2 * CLK), 2: ("slam", 50 + CLK), 3: ("gazebo", CLK), 4: ("nav2", 9999)}
    u = cc.usage(prev, now, 2.0)
    assert abs(u["gazebo"] - 1.5) < 1e-9      # (2 + 1) s CPU over 2 s
    assert abs(u["slam"] - 0.5) < 1e-9
    assert "nav2" not in u                     # new process: no earlier sample
