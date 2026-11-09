"""Nav2 for a robot in a generated building (phase 2e; start robot.launch.py first).

    source /opt/ros/jazzy/setup.bash
    ros2 launch sim/launch/nav.launch.py building:=sim/building/out/bld_train_000 platform:=sim/configs/robots/wheeled_low.yaml

- map_server with the generator's map (<building>.map.yaml) + its own lifecycle manager;
- nav2_bringup navigation_launch.py with params from the platform file (doorsim.robot.nav2_params on top of nav2_bringup's
  defaults: footprint, speeds, base_link, sim time);
- no AMCL: robot.launch.py publishes map -> odom at the spawn pose (ground-truth localisation; AMCL / SLAM later).
"""
import pathlib, sys

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import robot  # noqa: E402


def _setup(context):
    arg = lambda k: LaunchConfiguration(k).perform(context)
    building = pathlib.Path(arg("building")).resolve()
    p = robot.load_platform(arg("platform"))
    bringup = pathlib.Path(get_package_share_directory("nav2_bringup"))
    params = robot.nav2_params(p, yaml.safe_load((bringup / "params" / "nav2_params.yaml").read_text()))
    params_file = pathlib.Path("/tmp") / f"nav2_{p['name']}.yaml"
    params_file.write_text(yaml.safe_dump(params, sort_keys=False))
    map_yaml = str(building.parent / f"{building.name}.map.yaml")
    return [
        Node(package="nav2_map_server", executable="map_server", name="map_server", output="screen",
             parameters=[{"yaml_filename": map_yaml, "use_sim_time": True}]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_map", output="screen",
             parameters=[{"use_sim_time": True, "autostart": True, "node_names": ["map_server"]}]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(bringup / "launch" / "navigation_launch.py")),
            launch_arguments={"params_file": str(params_file), "use_sim_time": "true", "autostart": "true"}.items()),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("building", description="path without extension, e.g. sim/building/out/bld_train_000"),
        DeclareLaunchArgument("platform", default_value=str(SIM / "configs" / "robots" / "wheeled_low.yaml")),
        OpaqueFunction(function=_setup),
    ])
