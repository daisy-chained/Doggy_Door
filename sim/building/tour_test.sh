#!/usr/bin/env bash
# Live test of the robot run (phase 2e): one building, robot + Nav2 headless, door tour over every reachable door side.
# Everything is stopped at the end (also on errors / timeout). Meant to run alone (no training / render next to it: earlyoom).
#
#   bash sim/building/tour_test.sh [building=sim/building/out/bld_train_000] [platform=sim/configs/robots/wheeled_low.yaml] [x=1.0] [y=4.04]
#
# Output in runs/tours/<building>_<platform>_<time>/: tour.json (per stop: reached, seconds, position / heading error),
# robot.log, nav.log, tour.log. No rosbag (camera at 10 Hz for a whole tour = GBs); --bag on door_tour.py when needed.
set -u
cd "$(dirname "$0")/../.."
BLD=${1:-sim/building/out/bld_train_000}
PLAT=${2:-sim/configs/robots/wheeled_low.yaml}
X=${3:-1.0}
Y=${4:-4.04}
OUT=runs/tours/$(basename "$BLD")_$(basename "$PLAT" .yaml)_$(date +%Y%m%d_%H%M)
mkdir -p "$OUT"

source /opt/ros/jazzy/setup.bash
export GZ_PARTITION=tour_$$ ROS_DOMAIN_ID=$(( ($$ % 100) + 100 ))   # own Gazebo + ROS network, nothing else gets in
PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -INT -- -"$p" 2>/dev/null; done
  sleep 5
  for p in "${PIDS[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done
}
trap cleanup EXIT

setsid ros2 launch sim/launch/robot.launch.py world:="$BLD.sdf" platform:="$PLAT" x:="$X" y:="$Y" > "$OUT/robot.log" 2>&1 &
PIDS+=($!)
echo "waiting for the robot (odom)"
timeout 120 bash -c 'until ros2 topic echo --once /odom >/dev/null 2>&1; do sleep 2; done' || { echo "robot did not come up"; exit 1; }

setsid ros2 launch sim/launch/nav.launch.py building:="$BLD" platform:="$PLAT" > "$OUT/nav.log" 2>&1 &
PIDS+=($!)
echo "robot up, Nav2 starting; tour log in $OUT/tour.log"

timeout 5400 python3 sim/building/door_tour.py "$BLD" --start "$X" "$Y" --out "$OUT/tour.json" 2>&1 | tee "$OUT/tour.log"
status=${PIPESTATUS[0]}
echo "door_tour exit $status (124 = 90 min timeout)"
exit "$status"
