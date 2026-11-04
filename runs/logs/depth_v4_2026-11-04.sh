#!/usr/bin/env bash
# 2026-11-04 (Doggy Door): depth-based detection (Madelief: depth-only first, then late fusion; depth + height + normal
# angle, plus plain depth; smoke-baked depth; doorway class). Exports with --doorway, three detector runs with the same
# tiny_10g winner settings (rgb / depth / dha), light + smoke level sets for all three, late fusion rgb + depth runs.
# Steps log to runs/logs/depth_v4_2026-11-04/.
set -uo pipefail
R=$HOME/git/Doggy_Door
L=$R/runs/logs/depth_v4_2026-11-04
PY=$R/train/.venv/bin/python
mkdir -p "$L"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/steps.txt"; }
step() { local name=$1; shift; log "START $name"; if "$@" > "$L/$name.log" 2>&1; then log "OK    $name"; else log "FAIL  $name (see $name.log)"; return 1; fi; }
cd "$R"

# 1. exports with the doorway class: training (ignore < 5 % on train / val, as before) + every fixed-level test set
TRAIN=exports/door_room_v4_2k_fx_ign_dw
step export_train python3 sim/doorsim/export.py datasets/door_room_v4_2k_fx --out $TRAIN --ignore-below 0.05 --doorway || exit 1
LH0=(); LH1=(); VIS=()
for lv in 1.0 0.5 0.2 0.1 0.05 0.02 0.01 0.005; do
  for h in 0 1.0; do
    n=v4_test_light${lv}_h${h}_fx
    step export_$n python3 sim/doorsim/export.py datasets/$n --out exports/${n}_dw --doorway \
      && { [ "$h" = 0 ] && LH0+=(exports/${n}_dw) || LH1+=(exports/${n}_dw); }
  done
done
for v in 16 8 6 4 3 2 1.5 1; do
  n=v4_test_vis${v}
  step export_$n python3 sim/doorsim/export.py datasets/$n --out exports/${n}_dw --doorway && VIS+=(exports/${n}_dw)
done

# 2. detector runs: same settings, only the input differs
DET="model.checkpoint=ustc-community/dfine-nano-coco imgsz=704 lr=0.00011241813335700244 weight_decay=1.2042902113374963e-06 warmup_steps=291 batch=8 epochs=30"
RUNS=()
for in in rgb dha depth; do
  step det_$in $PY train/train.py train/configs/dfine_s_640.yaml export=$TRAIN name=dfine_n704_v4dw_$in input=$in $DET && RUNS+=($in)
done

# 3. scoring per level + per distance for every run that trained
S=runs/levels_v4dw
mkdir -p $S
for in in "${RUNS[@]}"; do
  r=runs/dfine_n704_v4dw_$in
  step levels_light_h0_$in $PY train/score.py $r "${LH0[@]}" --out $S/light_headlight_off.jsonl
  step levels_light_h1_$in $PY train/score.py $r "${LH1[@]}" --out $S/light_headlight_on.jsonl
  step levels_vis_$in $PY train/score.py $r "${VIS[@]}" --out $S/visibility.jsonl
  step distance_$in $PY train/score_distance.py $r exports/door_room_v4_2k_fx_ign_dw
done

# 4. late fusion: rgb + each depth run, on the mixed test split and on every level set
for in in dha depth; do
  [[ " ${RUNS[*]} " == *" rgb "* && " ${RUNS[*]} " == *" $in "* ]] || continue
  mkdir -p $S/fusion_rgb_$in
  step fusion_rgb_${in}_test $PY train/score_fusion.py runs/dfine_n704_v4dw_rgb runs/dfine_n704_v4dw_$in $TRAIN --out $S/fusion_rgb_$in/test.json
  for e in "${LH0[@]}" "${LH1[@]}" "${VIS[@]}"; do
    b=$(basename $e)
    step fusion_rgb_${in}_$b $PY train/score_fusion.py runs/dfine_n704_v4dw_rgb runs/dfine_n704_v4dw_$in $e --out $S/fusion_rgb_$in/$b.json
  done
done

# 5. plots (one line per run)
step plot_light_h0 $PY train/plot_levels.py $S/light_headlight_off.jsonl --axis light_level
step plot_light_h1 $PY train/plot_levels.py $S/light_headlight_on.jsonl --axis light_level
step plot_vis $PY train/plot_levels.py $S/visibility.jsonl --axis visibility_m
log "DONE"
