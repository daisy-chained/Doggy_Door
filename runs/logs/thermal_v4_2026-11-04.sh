#!/usr/bin/env bash
# 2026-11-04 (Doggy Door, beat-smoke option A, Madelief's thermal design): thermal camera 320 x 240 next to the RGB-D camera,
# room-temperature + fire-heated scenarios (training on a per-scene mix, tested separately), LWIR smoke k = 0.1 x visible
# (+ k = 0.3 check at V 4 / 2 / 1 m). Thermal-only detector, then late fusion rgb + dha + thermal. Compare with runs/levels_v4dw.
set -uo pipefail
R=$HOME/git/Doggy_Door; L=$R/runs/logs/thermal_v4_2026-11-04; PY=$R/train/.venv/bin/python; S=runs/levels_v4th
mkdir -p "$L" $S
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/steps.txt"; }
step() { local name=$1; shift; log "START $name"; if "$@" > "$L/$name.log" 2>&1; then log "OK    $name"; else log "FAIL  $name (see $name.log)"; return 1; fi; }
cd "$R"
HEAVY_PID=${HEAVY_PID:-}

# 1. temperature images (both scenarios) for every frame of the base dataset (label render + lookup)
step render python3 sim/doorsim/thermal_render.py datasets/door_room_v4_2k --workers 4 || exit 1

# 2. thermal datasets: training mix (from the training smoke set) + test sets that reuse the RGB sets' smoke frame for frame
step fx_train python3 sim/doorsim/thermal_fx.py datasets/door_room_v4_2k_fx --out datasets/door_room_v4_2k_fx_th_mix --scenario mix --k 0.1 --workers 8 || exit 1
SETS=()
for sc in room fire; do
  n=v4_test_clear_th_$sc
  step fx_$n python3 sim/doorsim/thermal_fx.py datasets/door_room_v4_2k --out datasets/$n --scenario $sc --k 0.1 --split test --workers 8 && SETS+=($n)
  for v in 16 8 6 4 3 2 1.5 1; do
    n=v4_test_vis${v}_th_$sc
    step fx_$n python3 sim/doorsim/thermal_fx.py datasets/v4_test_vis$v --out datasets/$n --scenario $sc --k 0.1 --workers 8 && SETS+=($n)
  done
  for v in 4 2 1; do
    n=v4_test_vis${v}_th_${sc}_k0.3
    step fx_$n python3 sim/doorsim/thermal_fx.py datasets/v4_test_vis$v --out datasets/$n --scenario $sc --k 0.3 --workers 8 && SETS+=($n)
  done
done

# 3. exports with the doorway class (training: ignore < 5 % LWIR transmission on train / val, as before)
TRAIN=exports/door_room_v4_2k_fx_th_mix_ign_dw
step export_train python3 sim/doorsim/export.py datasets/door_room_v4_2k_fx_th_mix --out $TRAIN --ignore-below 0.05 --doorway || exit 1
EX=()
for n in "${SETS[@]}"; do step export_$n python3 sim/doorsim/export.py datasets/$n --out exports/${n}_dw --doorway && EX+=(exports/${n}_dw); done

# 4. thermal detector (same settings as the rgb / dha runs), once the heavy-smoke queue has the GPU free
if [ -n "$HEAVY_PID" ]; then log "waiting for heavy-smoke queue (pid $HEAVY_PID)"; while kill -0 "$HEAVY_PID" 2> /dev/null; do sleep 30; done; fi
DET="model.checkpoint=ustc-community/dfine-nano-coco imgsz=704 lr=0.00011241813335700244 weight_decay=1.2042902113374963e-06 warmup_steps=291 batch=8 epochs=30"
step det_thermal $PY train/train.py train/configs/dfine_s_640.yaml export=$TRAIN name=dfine_n704_v4th_thermal input=thermal $DET || exit 1

# 5. scores: every run on the thermal sets (rgb / dha see the same frames' RGB-D), fusion rgb + dha + thermal and rgb + dha
TH=runs/dfine_n704_v4th_thermal; RGB=runs/dfine_n704_v4dw_rgb; DHA=runs/dfine_n704_v4dw_dha
for r in $TH $RGB $DHA; do
  step mixed_$(basename $r) $PY train/score.py $r $TRAIN --out $S/mixed_test.jsonl
  step levels_$(basename $r) $PY train/score.py $r "${EX[@]}" --out $S/levels.jsonl
done
step distance_thermal $PY train/score_distance.py $TH $TRAIN
mkdir -p $S/fusion_rgb_dha_th $S/fusion_rgb_dha $S/fusion_dha_th
for e in $TRAIN "${EX[@]}"; do
  b=$(basename $e)
  step fusion3_$b $PY train/score_fusion.py $RGB $DHA $TH $e --out $S/fusion_rgb_dha_th/$b.json
  step fusion2_$b $PY train/score_fusion.py $RGB $DHA $e --out $S/fusion_rgb_dha/$b.json
  step fusion_dha_th_$b $PY train/score_fusion.py $DHA $TH $e --out $S/fusion_dha_th/$b.json
done
log "DONE"
