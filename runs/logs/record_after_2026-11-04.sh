#!/usr/bin/env bash
# After tonight's queues (heavy smoke -> thermal -> learned radar; pid $1 = the last one): record every score in MLflow.
cd $HOME/git/Doggy_Door
while kill -0 $1 2> /dev/null; do sleep 60; done
train/.venv/bin/python train/record.py backfill > runs/logs/record_after_2026-11-04.log 2>&1
