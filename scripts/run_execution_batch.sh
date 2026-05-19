#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

logdir="results/execution-logs"
mkdir -p "$logdir"

run(){
  name="$1"; shift
  echo "=== START $name $(date -u +%FT%TZ) ===" | tee -a "$logdir/$name.log"
  "$@" 2>&1 | tee -a "$logdir/$name.log"
  echo "=== END $name $(date -u +%FT%TZ) ===" | tee -a "$logdir/$name.log"
}

run sgjm25_calib_a python3 -m sgjm.training --config runs/calibration-configs/sgjm25_calib_a.json --backend cpu
run sgjm25_calib_b python3 -m sgjm.training --config runs/calibration-configs/sgjm25_calib_b.json --backend cpu
run sgjm25_calib_c python3 -m sgjm.training --config runs/calibration-configs/sgjm25_calib_c.json --backend cpu

run sgjm250_calib_a python3 -m sgjm.training --config runs/calibration-configs/sgjm250_calib_a.json --backend cpu
run sgjm250_calib_b python3 -m sgjm.training --config runs/calibration-configs/sgjm250_calib_b.json --backend cpu

run sgjm1b_smoke python3 -m sgjm.training --config runs/calibration-configs/sgjm1b_smoke.json --backend cpu

echo "ALL_DONE $(date -u +%FT%TZ)" | tee "$logdir/ALL_DONE.txt"
