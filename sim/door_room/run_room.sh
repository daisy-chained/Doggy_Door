#!/usr/bin/env bash
# Door room v0 check (headless): boxes, one frame + metadata closed, open both doors, boxes + frame + metadata again.
cd "$(dirname "$0")"; source /opt/ros/jazzy/setup.bash
OUT=${1:-out}; mkdir -p "$OUT"
setsid gz sim -s -r --headless-rendering door_room_v0.sdf > "$OUT/gz.log" 2>&1 & GZ=$!
setsid ros2 run ros_gz_bridge parameter_bridge /rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image > "$OUT/bridge.log" 2>&1 & BR=$!
sleep 10
boxes() { timeout 10 gz topic -e -t /boxes -n 1 > "$1"; tr -d ' ' < "$1" | grep -E "^(x|y|label):" | paste -sd' ' | sed 's/x:/\n x:/g' | paste - - | sed 's/^/ /'; }
boxes3d() { timeout 10 gz topic -e -t /boxes3d -n 1 > "$1"; }
echo "== closed: boxes (min x y, max x y, object id)"; boxes "$OUT/closed_boxes.txt"; boxes3d "$OUT/closed_boxes3d.txt"
python3 grab.py "$OUT/closed.png"; python3 meta.py "$OUT/closed.json" "$OUT/closed_boxes.txt" "$OUT/closed_boxes3d.txt" door_a=0 door_b=0
gz topic -t /door_a/cmd_pos -m gz.msgs.Double -p 'data: -1.2'
gz topic -t /door_b/cmd_pos -m gz.msgs.Double -p 'data: -1.2'
sleep 6
echo "== open: boxes"; boxes "$OUT/open_boxes.txt"; boxes3d "$OUT/open_boxes3d.txt"
python3 grab.py "$OUT/open.png"; python3 meta.py "$OUT/open.json" "$OUT/open_boxes.txt" "$OUT/open_boxes3d.txt" door_a=-1.2 door_b=-1.2
kill -- -$BR -$GZ 2>/dev/null; wait 2>/dev/null
echo "== gz errors"; grep -i -E "error|warn" "$OUT/gz.log" | grep -v -i egl | head -10
