"""Start a building world headless with one robot in it (phase 2e, wheeled base first; Madelief 2026-11-09).

    source /opt/ros/jazzy/setup.bash
    ros2 launch sim/launch/robot.launch.py world:=sim/building/out/bld_train_000.sdf platform:=sim/configs/robots/wheeled_low.yaml \
        x:=1.0 y:=4.0 yaw:=0.0 [gui:=true]

What it starts:
- Gazebo (server only unless gui:=true), the world running;
- the robot from the platform YAML (doorsim.robot.robot_sdf), spawned at x / y / yaw on the floor;
- ros_gz_bridge for /cmd_vel, /odom, /tf, /clock, /scan, /rgbd/*, /joint_states (doorsim.robot.bridge_args);
- static tfs base_link -> camera_link -> camera_optical, base_link -> lidar_link;
- map -> odom = the spawn pose (ground-truth localisation: the diff drive's odometry starts at the spawn pose; AMCL / SLAM later).
"""
import pathlib, sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import robot  # noqa: E402


def _tf(parent, child, xyz, rpy):
    return Node(package="tf2_ros", executable="static_transform_publisher", name=f"tf_{parent}_{child}".replace("/", "_"),
                arguments=["--x", str(xyz[0]), "--y", str(xyz[1]), "--z", str(xyz[2]),
                           "--roll", str(rpy[0]), "--pitch", str(rpy[1]), "--yaw", str(rpy[2]),
                           "--frame-id", parent, "--child-frame-id", child],
                parameters=[{"use_sim_time": True}])


def _setup(context):
    arg = lambda k: LaunchConfiguration(k).perform(context)
    world = str(pathlib.Path(arg("world")).resolve())
    p = robot.load_platform(arg("platform"))
    name, x, y, yaw = arg("name"), float(arg("x")), float(arg("y")), float(arg("yaw"))
    sdf_file = pathlib.Path("/tmp") / f"{name}_{p['name']}.sdf"
    sdf_file.write_text(robot.robot_sdf(p, name))

    gz_args = ["-r", world] if arg("gui") == "true" else ["-r", "-s", world]
    actions = [
        ExecuteProcess(cmd=["gz", "sim", *gz_args], output="screen"),
        TimerAction(period=3.0, actions=[Node(
            package="ros_gz_sim", executable="create", output="screen",
            arguments=["-name", name, "-file", str(sdf_file), "-x", str(x), "-y", str(y), "-z", "0.0", "-Y", str(yaw)])]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             arguments=robot.bridge_args(p), parameters=[{"use_sim_time": True}]),
        _tf("map", "odom", [x, y, 0.0], [0.0, 0.0, yaw]),
    ]
    actions += [_tf(*t) for t in robot.static_tfs(p)]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world"),
        DeclareLaunchArgument("platform", default_value=str(SIM / "configs" / "robots" / "wheeled_low.yaml")),
        DeclareLaunchArgument("name", default_value="robot"),
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("gui", default_value="false"),
        OpaqueFunction(function=_setup),
    ])
