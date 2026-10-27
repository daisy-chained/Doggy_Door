#!/usr/bin/env bash
# Headless smoke test: Gazebo RGB-D + bounding-box camera, images bridged to ROS 2.
# (no set -u: ROS setup.bash uses unset vars)
source /opt/ros/jazzy/setup.bash
cd "$(dirname "$0")"

setsid gz sim -s -r --headless-rendering rgbd_smoke.sdf > gz.log 2>&1 &
GZ=$!
setsid ros2 run ros_gz_bridge parameter_bridge \
  /rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image \
  /rgbd/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked > bridge.log 2>&1 &
BR=$!
sleep 12

echo "== gz topics"; gz topic -l | grep -E "rgbd|boxes"
echo "== bounding box (one msg)"; timeout 10 gz topic -e -t /boxes -n 1 | head -30
echo "== ros2 rates (raw subscriber, 10 s)"; python3 count_msgs.py 10
echo "== image header"; timeout 8 ros2 topic echo --once --field width /rgbd/image 2>&1 | head -2

kill -- -$BR -$GZ 2>/dev/null; wait 2>/dev/null
echo "== gz log errors"; grep -i -E "error|fail" gz.log | head -10 || true
