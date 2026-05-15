#!/usr/bin/env bash
# Bootstrap SGJM on a blank Apple Silicon Mac.
# Usage: bash setup_remote.sh [repo_url] [branch]
set -euo pipefail

REPO_URL="${1:-https://github.com/AdamPippert/SGJM.git}"
BRANCH="${2:-claude/custom-gpt-setup-kW8tb}"
INSTALL_DIR="$HOME/Development/ml-experiments/SGJM"
CONDA_DIR="$HOME/miniforge3"
PYTHON_VERSION="3.12"
ENV_NAME="sgjm"

# ── 1. Homebrew ──────────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
  echo "[setup] Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add brew to PATH for the rest of this script
  eval "$(/opt/homebrew/bin/brew shellenv)"
else
  echo "[setup] Homebrew already installed: $(brew --version | head -1)"
fi

# ── 2. Git ────────────────────────────────────────────────────────────────────
if ! command -v git &>/dev/null; then
  brew install git
fi

# ── 3. Miniforge (conda with arm64 packages) ─────────────────────────────────
if [[ ! -d "$CONDA_DIR" ]]; then
  echo "[setup] Installing Miniforge3 for arm64..."
  MINIFORGE_PKG="Miniforge3-MacOSX-arm64.sh"
  curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE_PKG" -o "/tmp/$MINIFORGE_PKG"
  bash "/tmp/$MINIFORGE_PKG" -b -p "$CONDA_DIR"
  rm "/tmp/$MINIFORGE_PKG"
else
  echo "[setup] Miniforge already at $CONDA_DIR"
fi

# Activate conda for this script
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate base

# ── 4. Python environment ─────────────────────────────────────────────────────
if ! conda env list | grep -q "^$ENV_NAME "; then
  echo "[setup] Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION..."
  conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

# ── 5. Clone / update repo ────────────────────────────────────────────────────
mkdir -p "$(dirname "$INSTALL_DIR")"
if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  echo "[setup] Cloning $REPO_URL → $INSTALL_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
else
  echo "[setup] Repo exists, pulling latest..."
  git -C "$INSTALL_DIR" fetch origin
  git -C "$INSTALL_DIR" checkout "$BRANCH"
  git -C "$INSTALL_DIR" pull --ff-only
fi

cd "$INSTALL_DIR"

# ── 6. Install Python dependencies ────────────────────────────────────────────
echo "[setup] Installing SGJM with MLX backend..."
pip install --upgrade pip
pip install -e '.[mlx,dev]'

# ── 7. Smoke test ─────────────────────────────────────────────────────────────
echo "[setup] Running smoke test..."
python -m sgjm.training --size smoke --backend mlx --steps 4 \
  --checkpoint-dir /tmp/sgjm-smoke-test
echo "[setup] Smoke test PASSED"

# ── 8. Print versions ─────────────────────────────────────────────────────────
echo ""
echo "=== Environment ready ==="
python --version
python -c "import mlx.core as mx; print(f'MLX {mx.__version__}')"
python -c "import sgjm; print(f'SGJM installed at {sgjm.__file__}')"
echo "Install dir: $INSTALL_DIR"
echo "Run 1B training with:"
echo "  conda activate $ENV_NAME"
echo "  cd $INSTALL_DIR"
echo "  python -m sgjm.training --size 1b --backend mlx --data-source python_extended \\"
echo "    --steps 50000 --checkpoint-dir runs/sgjm-1b 2>&1 | tee runs/sgjm-1b/train.log &"
