#!/usr/bin/env bash
# 2026-11-06 (Doggy Door): adaptive late fusion (Madelief, option A "quality rule"): each model's weight per image follows a
# "can this sensor see" score from its own input (RGB: worse of brightness and band-pass detail, depth/dha: share of clean
# depth pixels, thermal: band-pass detail), as a ratio to the 90th percentile on its training export's val split, floor 0.05.
# Power 1 (main) and power 2 (sharper, check). Same runs + sets as the equal-weight fusion of 11-04, so the numbers compare.
set -uo pipefail
R=$HOME/git/Doggy_Door; L=$R/runs/logs/fusion_adaptive_2026-11-06; PY=$R/train/.venv/bin/python
mkdir -p "$L"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L/steps.txt"; }
step() { local name=$1; shift; log "START $name"; if "$@" > "$L/$name.log" 2>&1; then log "OK    $name"; else log "FAIL  $name (see $name.log)"; return 1; fi; }
cd "$R"
RGB=runs/dfine_n704_v4dw_rgb; DHA=runs/dfine_n704_v4dw_dha; TH=runs/dfine_n704_v4th_thermal

# 1. rgb + dha on the light / smoke level sets (compare runs/levels_v4dw/fusion_rgb_dha)
DW=(exports/door_room_v4_2k_fx_ign_dw); for f in runs/levels_v4dw/fusion_rgb_dha/v4_*.json; do DW+=(exports/$(basename $f .json)); done
# 2. rgb + dha + thermal on the thermal sets (compare runs/levels_v4th/fusion_rgb_dha_th)
THS=(); for f in runs/levels_v4th/fusion_rgb_dha_th/*.json; do THS+=(exports/$(basename $f .json)); done

for pw in 1 2; do
  S=runs/levels_v4dw/fusion_rgb_dha_adaptive_p$pw; mkdir -p $S
  for e in "${DW[@]}"; do b=$(basename $e); [[ $b == door_room_* ]] && o=test || o=$b
    step dw_p${pw}_$b $PY train/score_fusion.py $RGB $DHA $e --adaptive --power $pw --out $S/$o.json; done
  S=runs/levels_v4th/fusion_rgb_dha_th_adaptive_p$pw; mkdir -p $S
  for e in "${THS[@]}"; do b=$(basename $e)
    step th_p${pw}_$b $PY train/score_fusion.py $RGB $DHA $TH $e --adaptive --power $pw --out $S/$b.json; done
done
log "DONE"
