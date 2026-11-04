#!/usr/bin/env bash
# 2026-11-04 (Doggy Door, Madelief's radar proxy design): wide depth + label render for every frame of door_room_v4_2k,
# then the geometric doorway finder on single-chip + imaging radar + the clean upper bound, test split -> runs/radar/.
set -uo pipefail
R=$HOME/git/Doggy_Door; L=$R/runs/logs/radar_v4_2026-11-04
mkdir -p "$L" $R/runs/radar
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/steps.txt"; }
step() { local name=$1; shift; log "START $name"; if "$@" > "$L/$name.log" 2>&1; then log "OK    $name"; else log "FAIL  $name (see $name.log)"; return 1; fi; }
set +u; source /opt/ros/jazzy/setup.bash; set -u   # gz Python bindings
cd "$R"
for try in 1 2 3; do  # resumable; a Gazebo start can time out under load
  step render_$try python3 sim/doorsim/radar_render.py datasets/door_room_v4_2k --workers 3 && break
  [ $try = 3 ] && exit 1
done
step doorways_test python3 sim/doorsim/radar_doorways.py datasets/door_room_v4_2k --split test --models single imaging --clean --out runs/radar/v4_test.jsonl
log "DONE"
