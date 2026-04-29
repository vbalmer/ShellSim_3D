#!/usr/bin/env bash
#SBATCH --job-name=shellsim_smoke_train
#SBATCH --output=euler/logs/smoke_train-%j.out
#SBATCH --error=euler/logs/smoke_train-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#
# Smoke-test trainer. Validates: torch + lightning + GPU, data load from
# DATA_DIR, MODEL_DIR / LOGS_DIR plumbing, full save path (best/last .pt,
# stats.pkl, inp.pkl, test_data.pkl), wandb behaviour.
# Tiny model, 3 epochs. ~5-10 min.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/euler/logs"

# Smoke uses its own dirs.
export DATA_DIR="${DATA_DIR:-$SCRATCH/ShellSim3D/smoke/data}"
export MODEL_DIR="${MODEL_DIR:-$HOME/ShellSim3D/smoke/models}"
export LOGS_DIR="${LOGS_DIR:-$HOME/ShellSim3D/smoke/logs}"
mkdir -p "$MODEL_DIR" "$LOGS_DIR"

# Tiny architecture + few epochs so this finishes in minutes regardless of GPU.
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export BATCH_SIZE="${BATCH_SIZE:-1024}"
export HIDDEN_LAYERS="${HIDDEN_LAYERS:-[256, 256, 256]}"

# Default to offline wandb so smoke runs don't pollute the dashboard.
# Comment out the next line if you want to verify wandb online auth instead.
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "============================================================"
echo "[smoke-train] Job ID:         $SLURM_JOB_ID"
echo "[smoke-train] Node:           $SLURMD_NODENAME"
echo "[smoke-train] DATA_DIR:       $DATA_DIR"
echo "[smoke-train] MODEL_DIR:      $MODEL_DIR"
echo "[smoke-train] LOGS_DIR:       $LOGS_DIR"
echo "[smoke-train] NUM_EPOCHS:     $NUM_EPOCHS"
echo "[smoke-train] BATCH_SIZE:     $BATCH_SIZE"
echo "[smoke-train] HIDDEN_LAYERS:  $HIDDEN_LAYERS"
echo "[smoke-train] WANDB_MODE:     $WANDB_MODE"
echo "[smoke-train] Started:        $(date -Iseconds)"
echo "============================================================"

# shellcheck disable=SC1091
source "$REPO_ROOT/euler/activate.sh"
export MPLBACKEND=Agg
nvidia-smi || true

cd "$REPO_ROOT/training"
python -u train.py

echo "[smoke-train] Finished: $(date -Iseconds)"
echo "[smoke-train] Model artefacts:"
ls -lh "$MODEL_DIR" || true
echo "[smoke-train] Run archive:"
ls -lh "$LOGS_DIR" || true
