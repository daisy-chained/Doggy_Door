"""/odom_truth (Gazebo's true pose) -> drifting /odom + TF odom -> base_link (doorsim.odom_noise), for SLAM runs without perfect odometry.

    python3 sim/building/odom_noise_node.py [--seed N] [--trans-scale F] [--rot-scale F]     (ROS Jazzy sourced, system python)
Header stamp = the truth message stamp (use_sim_time); twist is copied from the truth."""
import argparse, math, pathlib, sys

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import TransformBroadcaster

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from doorsim import odom_noise  # noqa: E402


class OdomNoiseNode(Node):
    def __init__(self, params):
        super().__init__("odom_noise_node", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.drift = odom_noise.Drift(params)
        self.pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, "/odom_truth", self.on_odom, 10)

    def on_odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        x, y, a = self.drift.update((m.pose.pose.position.x, m.pose.pose.position.y, yaw))
        qz, qw = math.sin(a / 2), math.cos(a / 2)
        out = Odometry()
        out.header.stamp, out.header.frame_id, out.child_frame_id = m.header.stamp, "odom", "base_link"
        out.pose.pose.position.x, out.pose.pose.position.y = x, y
        out.pose.pose.orientation.z, out.pose.pose.orientation.w = qz, qw
        out.twist = m.twist
        self.pub.publish(out)
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = m.header.stamp, "odom", "base_link"
        t.transform.translation.x, t.transform.translation.y = x, y
        t.transform.rotation.z, t.transform.rotation.w = qz, qw
        self.tf.sendTransform(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--trans-scale", type=float)
    ap.add_argument("--rot-scale", type=float)
    a, ros_args = ap.parse_known_args()
    params = dict(odom_noise.DEFAULTS)
    for k in ("seed", "trans_scale", "rot_scale"):
        if getattr(a, k) is not None:
            params[k] = getattr(a, k)
    rclpy.init(args=ros_args)
    node = OdomNoiseNode(params)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
