"""/radar_raw (raw gpu_lidar ring) -> /scan (radar-like: beam widening, 4 cm bins, noise) for Nav2 / slam_toolbox.

    python3 sim/building/radar_node.py --platform sim/configs/robots/wheeled_ring.yaml     (ROS Jazzy sourced, system python)
Same header / frame (radar_link), angles and range limits as the raw scan; ranges = doorsim.radar_scan.process."""
import argparse, math, pathlib, sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from doorsim import radar_scan, robot  # noqa: E402


class RadarNode(Node):
    def __init__(self, radar):
        super().__init__("radar_node", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.radar, self.rng = radar, np.random.default_rng(0)
        self.pub = self.create_publisher(LaserScan, "/scan", 10)
        self.create_subscription(LaserScan, "/radar_raw", self.on_scan, qos_profile_sensor_data)

    def on_scan(self, m):
        r = self.radar
        out = LaserScan()
        out.header = m.header
        out.angle_min, out.angle_max, out.angle_increment = m.angle_min, m.angle_max, m.angle_increment
        out.time_increment, out.scan_time = m.time_increment, m.scan_time
        out.range_min, out.range_max = m.range_min, m.range_max
        res = radar_scan.process(np.asarray(m.ranges, dtype=float), m.angle_increment, m.range_min, m.range_max,
                                 math.radians(r["beam_deg"]), r["range_res"], r["noise"], self.rng)
        out.ranges = res.astype(np.float32).tolist()
        self.pub.publish(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = RadarNode(robot.load_platform(a.platform)["radar"])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
