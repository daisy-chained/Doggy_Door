"""Start a building world headless with one robot in it (phase 2e, wheeled base first; Madelief 2026-11-09).

    source /opt/ros/jazzy/setup.bash
    ros2 launch sim/launch/robot.launch.py world:=sim/building/out/bld_train_000.sdf platform:=sim/configs/robots/wheeled_low.yaml \
        x:=1.0 y:=4.0 yaw:=0.0 [gui:=true]

What it starts:
- Gazebo (server only unless gui:=true), the world running;
- the robot from the platform YAML (doorsim.robot.robot_sdf), spawned at x / y / yaw on the floor;
- ros_gz_bridge for /cmd_vel, /odom, /tf, /clock, /scan (lidar platform) or /radar_raw (radar platform), /<name>/rgbd/*, /joint_states (doorsim.robot.bridge_args);
- radar platform: sim/building/radar_node.py turns /radar_raw into /scan (beam, range bins, noise);
- static tfs base_link -> camera_link -> camera_optical (+ ring cameras), base_link -> lidar_link / radar_link;
- map -> odom = identity (ground-truth localisation: Gazebo's OdometryPublisher gives the true world pose as odom -> base_link,
  so no wheel-slip drift; AMCL / SLAM later); with odom_noise:=true the true pose goes to /odom_truth instead and
  sim/building/odom_noise_node.py publishes a drifting /odom + odom -> base_link (SLAM has to correct map -> odom).
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
    noisy = arg("odom_noise").lower() == "true"
    sdf_file = pathlib.Path("/tmp") / f"{name}_{p['name']}.sdf"
    sdf_file.write_text(robot.robot_sdf(p, name, odom_truth=noisy))

    gz_args = ["-r", world] if arg("gui") == "true" else ["-r", "-s", world]
    actions = [
        ExecuteProcess(cmd=["gz", "sim", *gz_args], output="screen"),
        TimerAction(period=3.0, actions=[Node(
            package="ros_gz_sim", executable="create", output="screen",
            arguments=["-name", name, "-file", str(sdf_file), "-x", str(x), "-y", str(y), "-z", "0.0", "-Y", str(yaw)])]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             arguments=robot.bridge_args(p, name, odom_truth=noisy), parameters=[{"use_sim_time": True}]),
        *([] if arg("slam").lower() == "true" else
          [_tf("map", "odom", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])]),   # odom = true world pose (robot.py); with slam, slam_toolbox publishes it
    ]
    if "radar" in p:
        actions.append(ExecuteProcess(cmd=["python3", str(SIM / "building" / "radar_node.py"), "--platform", arg("platform")],
                                      output="screen"))
    if noisy:
        actions.append(ExecuteProcess(cmd=["python3", str(SIM / "building" / "odom_noise_node.py")], output="screen"))
    actions += [_tf(*t) for t in robot.static_tfs(p)]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world"),
        DeclareLaunchArgument("platform", default_value=str(SIM / "configs" / "robots" / "wheeled_low.yaml")),
        DeclareLaunchArgument("slam", default_value="false", description="true: no map -> odom identity (slam_toolbox publishes it)"),
        DeclareLaunchArgument("odom_noise", default_value="false", description="true: /odom + odom -> base_link drift (odom_noise_node.py)"),
        DeclareLaunchArgument("name", default_value="robot"),
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("gui", default_value="false"),
        OpaqueFunction(function=_setup),
    ])
