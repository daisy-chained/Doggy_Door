#!/usr/bin/env bash
# 2026-11-04: rerun of step 4 (late fusion) of depth_v4_2026-11-04.sh after the fusion score fix (best score per model).
set -uo pipefail
R=$HOME/git/Doggy_Door; L=$R/runs/logs/depth_v4_2026-11-04; PY=$R/train/.venv/bin/python; S=runs/levels_v4dw
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/steps.txt"; }
step() { local name=$1; shift; log "START $name"; if "$@" > "$L/$name.log" 2>&1; then log "OK    $name"; else log "FAIL  $name (see $name.log)"; return 1; fi; }
cd "$R"
for in in dha depth; do
  mkdir -p $S/fusion_rgb_$in
  step fusion2_rgb_${in}_test $PY train/score_fusion.py runs/dfine_n704_v4dw_rgb runs/dfine_n704_v4dw_$in exports/door_room_v4_2k_fx_ign_dw --out $S/fusion_rgb_$in/test.json
  for e in exports/v4_test_light*_fx_dw exports/v4_test_vis*_dw; do
    step fusion2_rgb_${in}_$(basename $e) $PY train/score_fusion.py runs/dfine_n704_v4dw_rgb runs/dfine_n704_v4dw_$in $e --out $S/fusion_rgb_$in/$(basename $e).json
  done
done
log "DONE fusion2"
