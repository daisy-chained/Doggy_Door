#!/usr/bin/env bash
# 2026-11-06 (Doggy Door, 2d, Madelief's design): detector confidence calibration, temperature per class vs box-aware
# (Kuppers 2020), fitted on the mixed val split, D-ECE on test + every light / smoke level set. rgb, dha, thermal and fused
# (adaptive). Waits for the props queue (GPU) to finish first.
set -uo pipefail
R=$HOME/git/Doggy_Door; L=$R/runs/logs/calibrate_2026-11-06; PY=$R/train/.venv/bin/python
mkdir -p "$L"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/steps.txt"; }
step() { local name=$1; shift; log "START $name"; if timeout 4h "$@" > "$L/$name.log" 2>&1; then log "OK    $name"; else log "FAIL  $name (see $name.log)"; return 1; fi; }
cd "$R"
sleep 30; while pgrep -f "props_dha_2026-11-06.sh|fusion_adaptive_2026-11-06.sh" > /dev/null; do sleep 60; done
RGB=runs/dfine_n704_v4dw_rgb; DHA=runs/dfine_n704_v4dw_dha; TH=runs/dfine_n704_v4th_thermal
DWFIT=exports/door_room_v4_2k_fx_ign_dw; THFIT=exports/door_room_v4_2k_fx_th_mix_ign_dw
DW=($DWFIT); for f in runs/levels_v4dw/fusion_rgb_dha/v4_*.json; do DW+=(exports/$(basename $f .json)); done
THS=(); for f in runs/levels_v4th/fusion_rgb_dha_th/*.json; do THS+=(exports/$(basename $f .json)); done
S=runs/calibration
step rgb $PY train/calibrate.py $RGB --fit $DWFIT --eval "${DW[@]}" --out $S/rgb
step dha $PY train/calibrate.py $DHA --fit $DWFIT --eval "${DW[@]}" --out $S/dha
step thermal $PY train/calibrate.py $TH --fit $THFIT --eval "${THS[@]}" --out $S/thermal
step fused_rgb_dha $PY train/calibrate.py $RGB $DHA --adaptive --fit $DWFIT --eval "${DW[@]}" --out $S/fused_rgb_dha
step fused_rgb_dha_th $PY train/calibrate.py $RGB $DHA $TH --adaptive --fit $THFIT --eval "${THS[@]}" --out $S/fused_rgb_dha_th
log "DONE"
