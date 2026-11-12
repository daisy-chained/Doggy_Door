"""Nav2 for a robot in a generated building (phase 2e; start robot.launch.py first).

    source /opt/ros/jazzy/setup.bash
    ros2 launch sim/launch/nav.launch.py building:=sim/building/out/bld_train_000 platform:=sim/configs/robots/wheeled_low.yaml

- map_server with the generator's map (<building>.map.yaml) + its own lifecycle manager;
- nav2_bringup navigation_launch.py with params from the platform file (doorsim.robot.nav2_params on top of nav2_bringup's
  defaults: footprint, speeds, base_link, sim time);
- no AMCL: robot.launch.py publishes map -> odom = identity (odom is the true world pose); slam:=true -> slam_toolbox
  (online async) builds /map and publishes map -> odom instead of map_server (run 4; odom is still the true pose).
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
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(bringup / "launch" / "navigation_launch.py")),
        launch_arguments={"params_file": str(params_file), "use_sim_time": "true", "autostart": "true"}.items())
    if arg("slam").lower() == "true":                  # run 4 (Madelief 2026-11-11): slam_toolbox builds /map live, no given floor map
        st = pathlib.Path(get_package_share_directory("slam_toolbox"))
        sp = yaml.safe_load((st / "config" / "mapper_params_online_async.yaml").read_text())
        sp["slam_toolbox"]["ros__parameters"].update(
            {"use_sim_time": True, "base_frame": "base_link", "odom_frame": "odom", "map_frame": "map", "scan_topic": "/scan",
             "resolution": 0.05, "max_laser_range": float((p.get("lidar") or p["radar"])["range"][1]), "map_update_interval": 2.0,
             "minimum_travel_distance": 0.2, "minimum_travel_heading": 0.3,
             # map->odom is stamped scan time + transform_timeout; scans land every 0.5 s, so the default 0.2 left gaps where Nav2
             # failed goals ("extrapolation into the future", dev log 2026-11-12). 1.0 s covers two missed scans.
             "transform_timeout": 1.0})
        slam_file = pathlib.Path("/tmp") / f"slam_{p['name']}.yaml"
        slam_file.write_text(yaml.safe_dump(sp, sort_keys=False))
        return [IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(str(st / "launch" / "online_async_launch.py")),
                    launch_arguments={"slam_params_file": str(slam_file), "use_sim_time": "true"}.items()),
                nav2]
    return [
        Node(package="nav2_map_server", executable="map_server", name="map_server", output="screen",
             parameters=[{"yaml_filename": map_yaml, "use_sim_time": True}]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_map", output="screen",
             parameters=[{"use_sim_time": True, "autostart": True, "node_names": ["map_server"]}]),
        nav2,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("building", description="path without extension, e.g. sim/building/out/bld_train_000"),
        DeclareLaunchArgument("platform", default_value=str(SIM / "configs" / "robots" / "wheeled_low.yaml")),
        DeclareLaunchArgument("slam", default_value="false", description="true: slam_toolbox maps live instead of map_server"),
        OpaqueFunction(function=_setup),
    ])
