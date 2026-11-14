#!/usr/bin/env bash
# Live test of the robot run (phase 2e): one building, robot + Nav2 headless, door tour over every reachable door side.
# Everything is stopped at the end (also on errors / timeout). Meant to run alone (no training / render next to it: earlyoom).
#
#   bash sim/building/tour_test.sh [building=sim/building/out/bld_train_000] [platform=sim/configs/robots/wheeled_low.yaml] [x=1.0] [y=4.04] [door_tour options...]
#   e.g. ... 1.0 4.04 --hinge-look runs/dfine_n960_v7_hw8   (hinge look, option A; score: python3 sim/building/score_hinge_look.py OUT)
#
# Dark / smoke conditions (sim/building/conditions.py, doorsim/conditions.py; Madelief 2026-11-13), all optional:
#   LIGHT=0.05|mixed   factor on every room light (mixed = a draw per room like the training mix)
#   SMOKE=4|mixed      smoke visibility in m on the saved frames (empty / unset = none)
#   HEADLIGHT=0.8      robot's own spot light per camera (default 0 = off)       COND_SEED=0   seed of the draws / smoke shape
#   With LIGHT or SMOKE set the run folder gets the suffix _l<LIGHT>_s<SMOKE|none>_h<HEADLIGHT>, the world is OUT/world.sdf and
#   door_tour gets --conditions OUT/conditions.json. Without them everything is as before.
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
ODOM_NOISE=${ODOM_NOISE:-0}                        # ODOM_NOISE=1: drifting wheel odometry instead of the true pose (run 5)
SCRIPT=${TOUR_SCRIPT:-sim/building/door_tour.py}   # what drives the robot (same arguments as door_tour.py)
LIGHT=${LIGHT:-}; SMOKE=${SMOKE:-}; HEADLIGHT=${HEADLIGHT:-0}; COND_SEED=${COND_SEED:-0}
COND=0; [ -n "$LIGHT$SMOKE" ] && COND=1
SL=false; [ "$SLAM" = 1 ] && SL=true
ON=false; [ "$ODOM_NOISE" = 1 ] && ON=true
OUT=runs/tours/$(basename "$BLD")_$(basename "$PLAT" .yaml)$([ "$SLAM" = 1 ] && echo _slam)$([ "$ODOM_NOISE" = 1 ] && echo _drift)$([ "$COND" = 1 ] && echo "_l${LIGHT:-1}_s${SMOKE:-none}_h${HEADLIGHT}")_$(date +%Y%m%d_%H%M)
mkdir -p "$OUT"
WORLD="$BLD.sdf"; COND_ARGS=()
if [ "$COND" = 1 ]; then
  python3 sim/building/conditions.py "$BLD" --light "${LIGHT:-1}" --smoke "${SMOKE:-none}" --headlight "$HEADLIGHT" --seed "$COND_SEED" --out "$OUT" || exit 1
  WORLD="$OUT/world.sdf"; COND_ARGS=(--conditions "$OUT/conditions.json")
fi

set +u; source /opt/ros/jazzy/setup.bash; set -u   # ROS setup scripts use unset variables
export GZ_PARTITION=tour_$$ ROS_DOMAIN_ID=${TOUR_DOMAIN_ID:-$(( ($$ % 100) + 100 ))}   # TOUR_DOMAIN_ID: side-by-side runs pick distinct ones   # own Gazebo + ROS network, nothing else gets in
export PYTHONUNBUFFERED=1   # door_tour lines reach tour.log / the runq log as they happen (piped output was held back)
PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -INT -- -"$p" 2>/dev/null; done
  sleep 5
  for p in "${PIDS[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done
  ros2 daemon stop >/dev/null 2>&1   # the ROS daemon detaches from the run and was left behind after every tour
}
trap cleanup EXIT

setsid ros2 launch sim/launch/robot.launch.py world:="$WORLD" platform:="$PLAT" x:="$X" y:="$Y" slam:="$SL" odom_noise:="$ON" headlight:="$HEADLIGHT" > "$OUT/robot.log" 2>&1 &
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

timeout "${TOUR_TIMEOUT:-5400}" python3 "$SCRIPT" "$BLD" --start "$X" "$Y" --out "$OUT/tour.json" --frames "$OUT/stops" "${COND_ARGS[@]}" "${@:5}" 2>&1 | tee "$OUT/tour.log"
status=${PIPESTATUS[0]}
echo "door_tour exit $status (124 = timeout after ${TOUR_TIMEOUT:-5400} s)"
exit "$status"
