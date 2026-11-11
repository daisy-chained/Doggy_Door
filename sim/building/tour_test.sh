#!/usr/bin/env bash
# Live test of the robot run (phase 2e): one building, robot + Nav2 headless, door tour over every reachable door side.
# Everything is stopped at the end (also on errors / timeout). Meant to run alone (no training / render next to it: earlyoom).
#
#   bash sim/building/tour_test.sh [building=sim/building/out/bld_train_000] [platform=sim/configs/robots/wheeled_low.yaml] [x=1.0] [y=4.04] [door_tour options...]
#   e.g. ... 1.0 4.04 --hinge-look runs/dfine_n960_v7_hw8   (hinge look, option A; score: python3 sim/building/score_hinge_look.py OUT)
#
# Output in runs/tours/<building>_<platform>_<time>/: tour.json (per stop: reached, seconds, position / heading error),
# stops/NNN_door<id>_<side>_<dist>/ (what the robot sees at each stop: rgb / depth / seg + true boxes), robot.log, nav.log, tour.log. No rosbag (camera at 10 Hz for a whole tour = GBs); --bag on door_tour.py when needed.
set -u
cd "$(dirname "$0")/../.."
BLD=${1:-sim/building/out/bld_train_000}
PLAT=${2:-sim/configs/robots/wheeled_low.yaml}
X=${3:-1.0}
Y=${4:-4.04}
SLAM=${SLAM:-0}                                    # SLAM=1: slam_toolbox maps live, no given floor map (run 4)
SCRIPT=${TOUR_SCRIPT:-sim/building/door_tour.py}   # what drives the robot (same arguments as door_tour.py)
SL=false; [ "$SLAM" = 1 ] && SL=true
OUT=runs/tours/$(basename "$BLD")_$(basename "$PLAT" .yaml)$([ "$SLAM" = 1 ] && echo _slam)_$(date +%Y%m%d_%H%M)
mkdir -p "$OUT"

set +u; source /opt/ros/jazzy/setup.bash; set -u   # ROS setup scripts use unset variables
export GZ_PARTITION=tour_$$ ROS_DOMAIN_ID=$(( ($$ % 100) + 100 ))   # own Gazebo + ROS network, nothing else gets in
export PYTHONUNBUFFERED=1   # door_tour lines reach tour.log / the runq log as they happen (piped output was held back)
PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -INT -- -"$p" 2>/dev/null; done
  sleep 5
  for p in "${PIDS[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done
  ros2 daemon stop >/dev/null 2>&1   # the ROS daemon detaches from the run and was left behind after every tour
}
trap cleanup EXIT

setsid ros2 launch sim/launch/robot.launch.py world:="$BLD.sdf" platform:="$PLAT" x:="$X" y:="$Y" slam:="$SL" > "$OUT/robot.log" 2>&1 &
PIDS+=($!)
echo "waiting for the robot (odom)"
timeout 120 bash -c 'until ros2 topic echo --once /odom >/dev/null 2>&1; do sleep 2; done' || { echo "robot did not come up"; exit 1; }

# Nav2 sometimes hangs at startup (a lifecycle change_state call times out and the node never goes active; 007 on 11-10 and the
# 11-11 check run): wait up to 90 s for bt_navigator to be active, else restart Nav2, at most 2 times (Claude 2026-11-11).
nav_ok=0
for try in 1 2 3; do
  setsid ros2 launch sim/launch/nav.launch.py building:="$BLD" platform:="$PLAT" slam:="$SL" >> "$OUT/nav.log" 2>&1 &
  NAV=$!
  echo "robot up, Nav2 starting (try $try); tour log in $OUT/tour.log"
  if timeout 90 bash -c 'until ros2 lifecycle get /bt_navigator 2>/dev/null | grep -q "^active"; do sleep 3; done'; then
    PIDS+=($NAV); nav_ok=1; break
  fi
  echo "Nav2 not active after 90 s, restarting it"
  kill -INT -- -"$NAV" 2>/dev/null; sleep 5; kill -KILL -- -"$NAV" 2>/dev/null; sleep 2
done
[ "$nav_ok" = 1 ] || { echo "Nav2 did not come up"; exit 1; }

timeout 5400 python3 "$SCRIPT" "$BLD" --start "$X" "$Y" --out "$OUT/tour.json" --frames "$OUT/stops" "${@:5}" 2>&1 | tee "$OUT/tour.log"
status=${PIPESTATUS[0]}
echo "door_tour exit $status (124 = 90 min timeout)"
exit "$status"
