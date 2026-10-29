#!/usr/bin/env bash
# Door room check (headless): boxes, one frame + metadata closed, open every door, boxes + frame + metadata again.
# usage: run_room.sh [world (default door_room_v0)] [out dir (default out/<world>)]
cd "$(dirname "$0")"; source /opt/ros/jazzy/setup.bash
WORLD=${1:-door_room_v0}; ATTRS="$WORLD.attributes.json"; OUT=${2:-out/$WORLD}; mkdir -p "$OUT"
setsid gz sim -s -r --headless-rendering "$WORLD.sdf" > "$OUT/gz.log" 2>&1 & GZ=$!
setsid ros2 run ros_gz_bridge parameter_bridge /rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image > "$OUT/bridge.log" 2>&1 & BR=$!
sleep 10
boxes() { timeout 10 gz topic -e -t /boxes -n 1 > "$1"; tr -d ' ' < "$1" | grep -E "^(x|y|label):" | paste -sd' ' | sed 's/x:/\n x:/g' | paste - - | sed 's/^/ /'; }
boxes3d() { timeout 10 gz topic -e -t /boxes3d -n 1 > "$1"; }
# "door_a=-1.2 door_b=0.85": every object with a joint and its open command, from the attributes file
OPEN=$(python3 -c "import json,sys; o=json.load(open(sys.argv[1]))['objects']; print(' '.join(f'{n}={v[\"joint\"][\"open\"]}' for n,v in o.items() if 'joint' in v))" "$ATTRS")
CLOSED=$(echo "$OPEN" | sed -E 's/=[^ ]+/=0/g')
echo "== closed: boxes (min x y, max x y, object id)"; boxes "$OUT/closed_boxes.txt"; boxes3d "$OUT/closed_boxes3d.txt"
python3 grab.py "$OUT/closed.png"; python3 meta.py "$ATTRS" "$OUT/closed.json" "$OUT/closed_boxes.txt" "$OUT/closed_boxes3d.txt" $CLOSED
for d in $OPEN; do gz topic -t "/${d%%=*}/cmd_pos" -m gz.msgs.Double -p "data: ${d#*=}"; done
sleep 6
echo "== open: boxes"; boxes "$OUT/open_boxes.txt"; boxes3d "$OUT/open_boxes3d.txt"
python3 grab.py "$OUT/open.png"; python3 meta.py "$ATTRS" "$OUT/open.json" "$OUT/open_boxes.txt" "$OUT/open_boxes3d.txt" $OPEN
kill -- -$BR -$GZ 2>/dev/null; wait 2>/dev/null
echo "== gz errors"; grep -i -E "error|warn" "$OUT/gz.log" | grep -v -i egl | head -10
