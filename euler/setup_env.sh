#!/usr/bin/env bash
# One-time environment setup for ShellSim_3D on Euler.
#
# Run this once on the login node (NOT inside a SLURM job):
#   cd $HOME/ShellSim_3D
#   bash euler/setup_env.sh
#
# It will:
#   1. Load the Euler module stack with Python 3.11 + CUDA 12.
#   2. Create a venv at $HOME/venvs/shellsim.
#   3. pip-install everything in euler/requirements_euler.txt.
#
# After this, every SLURM job activates the same venv via `source euler/activate.sh`.

set -euo pipefail

# Resolve repo root from the script location so we can be invoked from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/venvs/shellsim}"

echo "[setup] Repo root: $REPO_ROOT"
echo "[setup] Target venv: $VENV_DIR"

# 1 - Modules ---------------------------------------------------------------
# stack/2024-06 + python_cuda/3.11.6 -> Python 3.11, CUDA 12.1.1.
# eth_proxy is required for outbound HTTPS from compute/login nodes (PyPI, wandb).
module load stack/2024-06
module load python_cuda/3.11.6
module load eth_proxy

# 2 - Create the venv -------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[setup] Creating venv at $VENV_DIR"
    python -m venv "$VENV_DIR"
else
    echo "[setup] Venv already exists at $VENV_DIR -- reusing"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 3 - Install Python dependencies -------------------------------------------
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/euler/requirements_euler.txt"

# 4 - Sanity checks ---------------------------------------------------------
python - <<'PY'
import importlib, sys
mods = ["numpy", "scipy", "h5py", "pyDOE", "cupy", "torch", "lightning", "wandb", "matplotlib", "sklearn"]
print(f"Python: {sys.version.split()[0]}")
for m in mods:
    try:
        v = importlib.import_module(m).__version__
        print(f"  {m:12s} {v}")
    except Exception as e:
        print(f"  {m:12s} FAILED ({e})")
PY

echo
echo "[setup] Done."
echo "[setup] Next: configure wandb (one-time):  wandb login"
echo "[setup] Then submit a run with:            bash euler/submit.sh"
