#!/usr/bin/env bash
#SBATCH --job-name=shellsim_smoke_sample
#SBATCH --output=euler/logs/smoke_sample-%j.out
#SBATCH --error=euler/logs/smoke_sample-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#
# Smoke-test sampler. Validates: modules, venv, cupy + GPU, h5 writes,
# DATA_DIR plumbing. Tiny dataset, no specific GPU requested.
# Output: $DATA_DIR/output_{eps_g,sig_g,D}.h5  (~30 MB total).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/euler/logs"

# Smoke uses its own dirs so it can never overwrite a real run.
export DATA_DIR="${DATA_DIR:-$SCRATCH/ShellSim3D/smoke/data}"
export N_SAMPLES_3D="${N_SAMPLES_3D:-1e6}"
mkdir -p "$DATA_DIR"

echo "============================================================"
echo "[smoke-sample] Job ID:        $SLURM_JOB_ID"
echo "[smoke-sample] Node:          $SLURMD_NODENAME"
echo "[smoke-sample] DATA_DIR:      $DATA_DIR"
echo "[smoke-sample] N_SAMPLES_3D:  $N_SAMPLES_3D"
echo "[smoke-sample] Started:       $(date -Iseconds)"
echo "============================================================"

# shellcheck disable=SC1091
source "$REPO_ROOT/euler/activate.sh"
export MPLBACKEND=Agg
nvidia-smi || true

cd "$REPO_ROOT/sampling"
python -u sampler_analytical_RC3D.py

echo "[smoke-sample] Finished: $(date -Iseconds)"
ls -lh "$DATA_DIR"
