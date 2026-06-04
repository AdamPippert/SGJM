#!/usr/bin/env bash
set -euo pipefail

# Run on HYDE. Sequential long-run queue with resumable checkpoints.
# Usage: bash scripts/vacation_preload_hyde.sh

cd ~/Development/SGJM
mkdir -p results/vacation-preload logs/vacation
export PYTHONPATH=src

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
run_job() {
  local name="$1"; shift
  echo "[$(stamp)] START $name" | tee -a "logs/vacation/$name.log"
  "$@" 2>&1 | tee -a "logs/vacation/$name.log"
  echo "[$(stamp)] END $name" | tee -a "logs/vacation/$name.log"
}

# 1) Continue/launch 1B ROCm long run
if [ -f runs/sgjm-1b-rocm/final.pt ]; then
  echo "[$(stamp)] 1B appears complete (final.pt exists), skipping long run" | tee -a logs/vacation/sgjm-1b-rocm.log
else
  run_job sgjm-1b-rocm python3 -m sgjm.training --size 1b --backend rocm --checkpoint-dir runs/sgjm-1b-rocm --steps 30000 --data-source python_extended --seq-len 2048 --batch-size 1 --lr 6e-5
fi

# 2) 250M post-run stability pass
run_job sgjm-250m-rocm-refresh python3 -m sgjm.training --size 250m --backend rocm --checkpoint-dir runs/sgjm-250m-rocm-refresh --steps 12000 --data-source python_extended --seq-len 1024 --batch-size 2 --lr 8e-5

# 3) 100M sweep pair
run_job sgjm-100m-rocm-jepa005 python3 -m sgjm.training --size 100m --backend rocm --checkpoint-dir runs/sgjm-100m-rocm-jepa005 --steps 16000 --data-source python_extended --seq-len 1024 --batch-size 2 --lr 1.2e-4
run_job sgjm-100m-rocm-jepa025 python3 -m sgjm.training --size 100m --backend rocm --checkpoint-dir runs/sgjm-100m-rocm-jepa025 --steps 16000 --data-source python_extended --seq-len 1024 --batch-size 2 --lr 1.2e-4

echo "[$(stamp)] VACATION_PRELOAD_DONE" | tee -a logs/vacation/ALL_DONE.log
