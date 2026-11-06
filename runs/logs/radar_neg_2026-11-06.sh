#!/usr/bin/env bash
# 2026-11-06 (Doggy Door, learned radar sanity check, Claude): closed doors were ignore boxes, so the detector also "found"
# 32 % (imaging) / 67 % (single-chip) of closed doors (strict precision 0.75 / 0.57). Retrain on relabelled exports where closed
# doors in view are negatives; score everything (old + new) with the threshold chosen on strict F1. Waits for the GPU queues.
set -uo pipefail
R=$HOME/git/Doggy_Door; L=$R/runs/logs/radar_neg_2026-11-06; PY=$R/train/.venv/bin/python
mkdir -p "$L"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/steps.txt"; }
step() { local name=$1; shift; log "START $name"; if timeout 3h "$@" > "$L/$name.log" 2>&1; then log "OK    $name"; else log "FAIL  $name (see $name.log)"; return 1; fi; }
cd "$R"
sleep 30; while pgrep -f "props_dha_2026-11-06.sh|fusion_adaptive_2026-11-06.sh|calibrate_2026-11-06.sh" > /dev/null; do sleep 60; done
DET="model.checkpoint=ustc-community/dfine-nano-coco imgsz=640 lr=0.00011241813335700244 weight_decay=1.2042902113374963e-06 warmup_steps=291 batch=8 epochs=30"
for m in single imaging; do
  for v in 1 3; do
    n=radar_${m}_v$v
    step rescore_$n $PY train/score_radar.py runs/dfine_n640_$n exports/$n --select f1_strict --out runs/radar/learned_${n}_strict.json
    step det_${n}_neg $PY train/train.py train/configs/dfine_s_640.yaml export=exports/${n}_neg name=dfine_n640_${n}_neg input=rgb $DET \
      && step score_${n}_neg $PY train/score_radar.py runs/dfine_n640_${n}_neg exports/${n}_neg --select f1_strict --out runs/radar/learned_${n}_neg_strict.json
  done
done
log "DONE"
