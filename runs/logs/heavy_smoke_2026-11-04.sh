#!/usr/bin/env bash
# 2026-11-04 (Doggy Door, beat-smoke option B): heavier-smoke training set (effects_heavy: 80 % smoky, V 1-8 m), rgb + dha runs
# with the same settings, scored on the same visibility + light level sets, fused. Compare with runs/levels_v4dw.
set -uo pipefail
R=$HOME/git/Doggy_Door; L=$R/runs/logs/heavy_smoke_2026-11-04; PY=$R/train/.venv/bin/python; S=runs/levels_v4heavy
mkdir -p "$L" $S
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/steps.txt"; }
step() { local name=$1; shift; log "START $name"; if "$@" > "$L/$name.log" 2>&1; then log "OK    $name"; else log "FAIL  $name (see $name.log)"; return 1; fi; }
cd "$R"
while pgrep -f depth_v4_2026-11-04_fusion2.sh > /dev/null; do sleep 30; done   # GPU free first
step effects python3 sim/doorsim/effects.py datasets/door_room_v4_2k --config sim/configs/effects_heavy.yaml --workers 8 --out datasets/door_room_v4_2k_fxheavy || exit 1
TRAIN=exports/door_room_v4_2k_fxheavy_ign_dw
step export python3 sim/doorsim/export.py datasets/door_room_v4_2k_fxheavy --out $TRAIN --ignore-below 0.05 --doorway || exit 1
DET="model.checkpoint=ustc-community/dfine-nano-coco imgsz=704 lr=0.00011241813335700244 weight_decay=1.2042902113374963e-06 warmup_steps=291 batch=8 epochs=30"
for in in rgb dha; do step det_$in $PY train/train.py train/configs/dfine_s_640.yaml export=$TRAIN name=dfine_n704_v4heavy_$in input=$in $DET; done
VIS=(exports/v4_test_vis*_dw); LH0=(exports/v4_test_light*_h0_fx_dw)
for in in rgb dha; do
  step levels_vis_$in $PY train/score.py runs/dfine_n704_v4heavy_$in "${VIS[@]}" --out $S/visibility.jsonl
  step levels_light_h0_$in $PY train/score.py runs/dfine_n704_v4heavy_$in "${LH0[@]}" --out $S/light_headlight_off.jsonl
done
mkdir -p $S/fusion_rgb_dha
step fusion_test $PY train/score_fusion.py runs/dfine_n704_v4heavy_rgb runs/dfine_n704_v4heavy_dha exports/door_room_v4_2k_fx_ign_dw --out $S/fusion_rgb_dha/test.json
for e in "${VIS[@]}"; do step fusion_$(basename $e) $PY train/score_fusion.py runs/dfine_n704_v4heavy_rgb runs/dfine_n704_v4heavy_dha $e --out $S/fusion_rgb_dha/$(basename $e).json; done
for in in rgb dha; do step mixed_$in $PY train/score.py runs/dfine_n704_v4heavy_$in exports/door_room_v4_2k_fx_ign_dw --out $S/mixed_test.jsonl; done
log "DONE"
