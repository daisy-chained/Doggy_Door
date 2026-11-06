#!/usr/bin/env bash
# 2026-11-06 (Doggy Door, 2c): depth-aware door property classifier (Madelief, option A: dha crop next to the RGB crop, more margin).
# Three runs on the same export as props_door_v4 (rgb, margin 0.1) so each change shows on its own:
# rgb margin 0.3 (margin only), dha margin 0.3 (depth only), rgb_dha margin 0.3 (both). Then true-box + detector-box scores.
# Waits for the adaptive fusion queue (GPU) to finish first.
set -uo pipefail
R=$HOME/git/Doggy_Door; L=$R/runs/logs/props_dha_2026-11-06; PY=$R/train/.venv/bin/python
mkdir -p "$L"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/steps.txt"; }
step() { local name=$1; shift; log "START $name"; if timeout 3h "$@" > "$L/$name.log" 2>&1; then log "OK    $name"; else log "FAIL  $name (see $name.log)"; return 1; fi; }
cd "$R"
while pgrep -f fusion_adaptive_2026-11-06.sh > /dev/null; do sleep 60; done
EX=exports/door_room_v4_2k_fx_ign
step props_door_v4_rgb_m30 $PY train/train_props.py train/configs/props_door.yaml export=$EX margin=0.3 name=props_door_v4_rgb_m30
step props_door_v4_dha_m30 $PY train/train_props.py train/configs/props_door_dha.yaml input=dha name=props_door_v4_dha_m30
step props_door_v4_rgbdha_m30 $PY train/train_props.py train/configs/props_door_dha.yaml name=props_door_v4_rgbdha_m30
for r in props_door_v4_rgb_m30 props_door_v4_dha_m30 props_door_v4_rgbdha_m30; do
  step score_$r $PY train/score_props.py runs/$r runs/dfine_n704_v4_ign $EX
done
log "DONE"
