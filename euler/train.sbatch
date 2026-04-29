#!/usr/bin/env bash
#SBATCH --job-name=shellsim_train
#SBATCH --output=euler/logs/train-%j.out
#SBATCH --error=euler/logs/train-%j.err
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#
# Step 2 of the pipeline: train the FFNN on the sampled data.
# Reads:    $DATA_DIR/output_{eps_g,sig_g,D}.h5
# Writes:   $MODEL_DIR/{inp,stats,test_data}.pkl
#           $MODEL_DIR/{best_trained_model_*.pt, last_trained_model.pt}
#           $LOGS_DIR/v_<n>/...                (when SAVE_FOLDER=True in train.py)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

mkdir -p "$REPO_ROOT/euler/logs"

# --- Default I/O locations (override by exporting before sbatch) -----------
export DATA_DIR="${DATA_DIR:-$SCRATCH/ShellSim3D/data}"
export MODEL_DIR="${MODEL_DIR:-$HOME/ShellSim3D/models/current}"
export LOGS_DIR="${LOGS_DIR:-$HOME/ShellSim3D/models/logs}"
mkdir -p "$MODEL_DIR" "$LOGS_DIR"

echo "============================================================"
echo "[train] Job ID:       $SLURM_JOB_ID"
echo "[train] Node:         $SLURMD_NODENAME"
echo "[train] Repo root:    $REPO_ROOT"
echo "[train] DATA_DIR:     $DATA_DIR"
echo "[train] MODEL_DIR:    $MODEL_DIR"
echo "[train] LOGS_DIR:     $LOGS_DIR"
echo "[train] Started:      $(date -Iseconds)"
echo "============================================================"

# --- Activate the venv + modules -------------------------------------------
# shellcheck disable=SC1091
source "$REPO_ROOT/euler/activate.sh"

# Headless plotting (no DISPLAY on compute nodes).
export MPLBACKEND=Agg

nvidia-smi || true

# --- wandb on Euler ---------------------------------------------------------
# train_utils.py always calls wandb.init(). If you have not run `wandb login`
# already (one-time, on the login node), uncomment the next line to disable
# the cloud sync and store runs locally:
# export WANDB_MODE=offline

# --- Run training -----------------------------------------------------------
# train.py uses bare imports of its sibling modules, so we cd into training/.
cd "$REPO_ROOT/training"
python -u train.py

echo "[train] Finished: $(date -Iseconds)"
echo "[train] Model artefacts:"
ls -lh "$MODEL_DIR" || true
