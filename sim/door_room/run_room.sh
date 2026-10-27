#!/usr/bin/env bash
# Door room v0 check (headless): labels, one frame closed, open both doors, labels + frame again.
cd "$(dirname "$0")"; source /opt/ros/jazzy/setup.bash
OUT=${1:-out}; mkdir -p "$OUT"
setsid gz sim -s -r --headless-rendering door_room_v0.sdf > "$OUT/gz.log" 2>&1 & GZ=$!
setsid ros2 run ros_gz_bridge parameter_bridge /rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image > "$OUT/bridge.log" 2>&1 & BR=$!
sleep 10
boxes() { timeout 10 gz topic -e -t /boxes -n 1 | tr -d ' ' | grep -E "^(x|y|label):" | paste -sd' ' | sed 's/x:/\n x:/g' | paste - - | sed 's/^/ /'; }
echo "== closed: boxes (min x y, max x y, label)"; boxes
python3 grab.py "$OUT/closed.png"
gz topic -t /door_a/cmd_pos -m gz.msgs.Double -p 'data: -1.2'
gz topic -t /door_b/cmd_pos -m gz.msgs.Double -p 'data: -1.2'
sleep 6
echo "== open: boxes"; boxes
python3 grab.py "$OUT/open.png"
kill -- -$BR -$GZ 2>/dev/null; wait 2>/dev/null
echo "== gz errors"; grep -i -E "error|warn" "$OUT/gz.log" | grep -v -i egl | head -10
