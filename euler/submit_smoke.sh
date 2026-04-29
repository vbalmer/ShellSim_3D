#!/usr/bin/env bash
# Submit the smoke pipeline (tiny sample -> tiny train), chained with afterok.
# Use this BEFORE bash euler/submit.sh -- it validates that the Euler setup is
# working (modules, venv, GPU, paths, wandb) without burning real compute.
#
# Usage:
#   bash euler/submit_smoke.sh
#
# Tweak knobs on the command line:
#   N_SAMPLES_3D=5e5 NUM_EPOCHS=2 bash euler/submit_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/euler/logs"

# Smoke-specific dirs so we can never overwrite a real production run.
export DATA_DIR="${DATA_DIR:-$SCRATCH/ShellSim3D/smoke/data}"
export MODEL_DIR="${MODEL_DIR:-$HOME/ShellSim3D/smoke/models}"
export LOGS_DIR="${LOGS_DIR:-$HOME/ShellSim3D/smoke/logs}"
export N_SAMPLES_3D="${N_SAMPLES_3D:-1e6}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export BATCH_SIZE="${BATCH_SIZE:-1024}"
export HIDDEN_LAYERS="${HIDDEN_LAYERS:-[256, 256, 256]}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export REPO_ROOT

echo "[smoke] DATA_DIR       = $DATA_DIR"
echo "[smoke] MODEL_DIR      = $MODEL_DIR"
echo "[smoke] LOGS_DIR       = $LOGS_DIR"
echo "[smoke] N_SAMPLES_3D   = $N_SAMPLES_3D"
echo "[smoke] NUM_EPOCHS     = $NUM_EPOCHS"
echo "[smoke] BATCH_SIZE     = $BATCH_SIZE"
echo "[smoke] HIDDEN_LAYERS  = $HIDDEN_LAYERS"
echo "[smoke] WANDB_MODE     = $WANDB_MODE"

SBATCH_FLAGS=(--export=ALL)

JOB_SAMPLE=$(sbatch "${SBATCH_FLAGS[@]}" --parsable euler/smoke_sample.sbatch)
echo "[smoke] sample job:    $JOB_SAMPLE"

JOB_TRAIN=$(sbatch "${SBATCH_FLAGS[@]}" --parsable \
                   --dependency=afterok:"$JOB_SAMPLE" \
                   euler/smoke_train.sbatch)
echo "[smoke] train job:     $JOB_TRAIN  (runs after $JOB_SAMPLE succeeds)"

echo
echo "[smoke] Watch:"
echo "  squeue --me"
echo "  tail -f euler/logs/smoke_sample-${JOB_SAMPLE}.out"
echo "  tail -f euler/logs/smoke_train-${JOB_TRAIN}.out"
echo
echo "[smoke] Expected: ~5-10 min queue + ~5-10 min runtime per stage."
echo "[smoke] If both finish without error and \$MODEL_DIR contains *.pt files,"
echo "[smoke] the production pipeline (bash euler/submit.sh) is good to go."
