#!/usr/bin/env bash
# Start SGJM 1B v2 training (verifier fix) on Mac Studio + Hyde simultaneously.
# Scheduled for Friday 2026-05-22 18:00 PDT.
set -euo pipefail

SSH="usr/bin/ssh -i /Users/apippert/.ssh/adams-mac-studio -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new"
LOG="/tmp/sgjm-1b-v2-launch-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== SGJM 1B v2 launch $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# ── Mac Studio ────────────────────────────────────────────────────────────────
echo "[mac-studio] pulling and starting..."
/usr/bin/ssh -i /Users/apippert/.ssh/adams-mac-studio \
  -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
  adam@Adams-Mac-Studio.local bash << 'REMOTE'
set -euo pipefail
cd ~/Development/ml-experiments/SGJM
git pull
pkill -f 'sgjm.training' 2>/dev/null || true
sleep 3
mkdir -p runs/sgjm-1b-mlx-v2
nohup bash -c '
  cd ~/Development/ml-experiments/SGJM
  source .venv/bin/activate
  python -m sgjm.training \
    --backend mlx --size 1b \
    --data-source python_extended \
    --checkpoint-dir runs/sgjm-1b-mlx-v2 2>&1
' > runs/sgjm-1b-mlx-v2/stdout.log 2>&1 &
echo $! > runs/sgjm-1b-mlx-v2/train.pid
echo "Mac Studio PID: $(cat runs/sgjm-1b-mlx-v2/train.pid)"
REMOTE

# ── Hyde ─────────────────────────────────────────────────────────────────────
echo "[hyde] pulling and starting..."
/usr/bin/ssh -i /Users/apippert/.ssh/adams-mac-studio \
  -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
  adam@hyde.tail4df14e.ts.net bash << 'REMOTE'
set -euo pipefail
cd ~/Development/SGJM
git pull
pkill -f 'sgjm.training' 2>/dev/null || true
sleep 3
mkdir -p runs/sgjm-1b-rocm-v2
nohup bash -c '
  cd ~/Development/SGJM
  source .venv/bin/activate
  export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
  python -m sgjm.training \
    --backend rocm --size 1b \
    --data-source python_extended \
    --checkpoint-dir runs/sgjm-1b-rocm-v2 2>&1
' > runs/sgjm-1b-rocm-v2/stdout.log 2>&1 &
echo $! > runs/sgjm-1b-rocm-v2/train.pid
echo "Hyde PID: $(cat runs/sgjm-1b-rocm-v2/train.pid)"
REMOTE

echo "=== both machines started — checking logs in 5 min ==="
sleep 300

echo "[mac-studio] first entries:"
/usr/bin/ssh -i /Users/apippert/.ssh/adams-mac-studio \
  -o BatchMode=yes adam@Adams-Mac-Studio.local \
  "head -3 ~/Development/ml-experiments/SGJM/runs/sgjm-1b-mlx-v2/train.jsonl 2>/dev/null || echo 'no log yet'"

echo "[hyde] first entries:"
/usr/bin/ssh -i /Users/apippert/.ssh/adams-mac-studio \
  -o BatchMode=yes adam@hyde.tail4df14e.ts.net \
  "head -3 ~/Development/SGJM/runs/sgjm-1b-rocm-v2/train.jsonl 2>/dev/null || echo 'no log yet'"

echo "=== done. full log at $LOG ==="
