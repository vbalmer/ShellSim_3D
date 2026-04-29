#!/usr/bin/env bash
#SBATCH --job-name=shellsim_sample
#SBATCH --output=euler/logs/sample-%j.out
#SBATCH --error=euler/logs/sample-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#
# Step 1 of the pipeline: sample 3D RC strain/stress/stiffness data.
# Writes:
#   $DATA_DIR/output_eps_g.h5
#   $DATA_DIR/output_sig_g.h5
#   $DATA_DIR/output_D.h5
#
# Submit with:
#   bash euler/submit.sh        (recommended, chains training after this)
# or stand-alone:
#   sbatch euler/sample.sbatch

set -euo pipefail

# --- Locate repo root (the directory that contains this script's parent) ---
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

mkdir -p "$REPO_ROOT/euler/logs"

# --- Default I/O locations (override by exporting before sbatch) -----------
# Sampled data goes to $SCRATCH (multi-TB, fast, auto-purged after ~15 days).
export DATA_DIR="${DATA_DIR:-$SCRATCH/ShellSim3D/data}"
mkdir -p "$DATA_DIR"

echo "============================================================"
echo "[sample] Job ID:       $SLURM_JOB_ID"
echo "[sample] Node:         $SLURMD_NODENAME"
echo "[sample] Repo root:    $REPO_ROOT"
echo "[sample] DATA_DIR:     $DATA_DIR"
echo "[sample] Started:      $(date -Iseconds)"
echo "============================================================"

# --- Activate the venv + modules -------------------------------------------
# shellcheck disable=SC1091
source "$REPO_ROOT/euler/activate.sh"

# Headless plotting (no DISPLAY on compute nodes).
export MPLBACKEND=Agg

nvidia-smi || true

# --- Run the sampler --------------------------------------------------------
# The script uses bare imports of its sibling modules, so we cd into sampling/.
cd "$REPO_ROOT/sampling"
python -u sampler_analytical_RC3D.py

echo "[sample] Finished: $(date -Iseconds)"
ls -lh "$DATA_DIR"
