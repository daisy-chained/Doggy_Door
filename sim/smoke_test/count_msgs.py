# Count messages per topic for N seconds (raw, no deserialisation) to see the real ROS 2 rate.
import sys, time, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
rclpy.init(); n = Node("counter"); c = {}
for t, ty in [("/rgbd/image", Image), ("/rgbd/depth_image", Image), ("/rgbd/points", PointCloud2)]:
    c[t] = 0
    n.create_subscription(ty, t, lambda m, t=t: c.__setitem__(t, c[t] + 1), 10, raw=True)
SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 10
end = time.time() + SECS
while time.time() < end: rclpy.spin_once(n, timeout_sec=0.1)
for t, k in c.items(): print(f"{t} {k/SECS:.1f} Hz")
