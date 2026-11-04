#!/usr/bin/env bash
# After the radar queue (pid $1): finder on stacked views 1 / 3 / 10 (same side of the wall), test split.
cd $HOME/git/Doggy_Door
while kill -0 $1 2> /dev/null; do sleep 30; done
python3 sim/doorsim/radar_doorways.py datasets/door_room_v4_2k --split test --models single imaging --clean --views 1 3 10 \
  --out runs/radar/v4_test_views.jsonl > runs/logs/radar_v4_2026-11-04/views.log 2>&1 && echo "$(date +%H:%M:%S) OK    views" >> runs/logs/radar_v4_2026-11-04/steps.txt \
  || echo "$(date +%H:%M:%S) FAIL  views" >> runs/logs/radar_v4_2026-11-04/steps.txt
