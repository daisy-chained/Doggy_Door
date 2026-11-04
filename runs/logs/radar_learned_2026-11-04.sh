#!/usr/bin/env bash
# 2026-11-04 (Doggy Door, Madelief "just do both"): learned doorway detector on the radar's top-down power image, one run
# per radar type x views (1 / 3), same D-FINE settings as the camera runs, imgsz 640 (160 px BEV upsampled 4x);
# scored with the same found / centre-error metric as the geometric finders (threshold picked on val).
set -uo pipefail
R=$HOME/git/Doggy_Door; L=$R/runs/logs/radar_learned_2026-11-04; PY=$R/train/.venv/bin/python
mkdir -p "$L" $R/runs/radar
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/steps.txt"; }
step() { local name=$1; shift; log "START $name"; if "$@" > "$L/$name.log" 2>&1; then log "OK    $name"; else log "FAIL  $name (see $name.log)"; return 1; fi; }
cd "$R"
WAIT_PID=${WAIT_PID:-}
if [ -n "$WAIT_PID" ]; then log "waiting for pid $WAIT_PID (GPU)"; while kill -0 "$WAIT_PID" 2> /dev/null; do sleep 30; done; fi
DET="model.checkpoint=ustc-community/dfine-nano-coco imgsz=640 lr=0.00011241813335700244 weight_decay=1.2042902113374963e-06 warmup_steps=291 batch=8 epochs=30"
for m in single imaging; do
  for v in 1 3; do
    n=radar_${m}_v$v
    step det_$n $PY train/train.py train/configs/dfine_s_640.yaml export=exports/$n name=dfine_n640_$n input=rgb $DET \
      && step score_$n $PY train/score_radar.py runs/dfine_n640_$n exports/$n --out runs/radar/learned_$n.json
  done
done
log "DONE"
