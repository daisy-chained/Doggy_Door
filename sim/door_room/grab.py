# Save one RGB frame from /rgbd/image as PNG (no numpy needed). usage: grab.py out.png
import sys, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from PIL import Image as PILImage
rclpy.init(); n = Node("grab"); got = []
n.create_subscription(Image, "/rgbd/image", lambda m: got.append(m), 10)
while not got: rclpy.spin_once(n, timeout_sec=0.5)
m = got[0]
mode = {"rgb8": "RGB", "bgr8": "RGB", "rgba8": "RGBA"}.get(m.encoding, "RGB")
PILImage.frombytes(mode, (m.width, m.height), bytes(m.data)).save(sys.argv[1])
print(f"saved {sys.argv[1]} ({m.width}x{m.height}, {m.encoding})")
