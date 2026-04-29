#!/usr/bin/env bash
# Submit the chained sampling -> training pipeline.
#
# Usage (from the repo root):
#   bash euler/submit.sh
#
# Tweak the I/O locations below or override them on the command line, e.g.:
#   DATA_DIR=$SCRATCH/foo MODEL_DIR=$HOME/bar bash euler/submit.sh
#
# To skip sampling and only re-train on the existing data:
#   bash euler/submit.sh --train-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$REPO_ROOT/euler/logs"

export DATA_DIR="${DATA_DIR:-$SCRATCH/ShellSim3D/data}"
export MODEL_DIR="${MODEL_DIR:-$HOME/ShellSim3D/models/current}"
export LOGS_DIR="${LOGS_DIR:-$HOME/ShellSim3D/models/logs}"
export REPO_ROOT

echo "[submit] DATA_DIR  = $DATA_DIR"
echo "[submit] MODEL_DIR = $MODEL_DIR"
echo "[submit] LOGS_DIR  = $LOGS_DIR"

TRAIN_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --train-only) TRAIN_ONLY=1 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# Make sure sbatch sees our env vars (DATA_DIR, MODEL_DIR, LOGS_DIR, REPO_ROOT).
SBATCH_FLAGS=(--export=ALL)

if [[ "$TRAIN_ONLY" -eq 1 ]]; then
    echo "[submit] --train-only: submitting training only"
    JOB_TRAIN=$(sbatch "${SBATCH_FLAGS[@]}" --parsable euler/train.sbatch)
    echo "[submit] training job:    $JOB_TRAIN"
else
    JOB_SAMPLE=$(sbatch "${SBATCH_FLAGS[@]}" --parsable euler/sample.sbatch)
    echo "[submit] sampling job:    $JOB_SAMPLE"

    JOB_TRAIN=$(sbatch "${SBATCH_FLAGS[@]}" --parsable \
                       --dependency=afterok:"$JOB_SAMPLE" \
                       euler/train.sbatch)
    echo "[submit] training job:    $JOB_TRAIN  (runs after $JOB_SAMPLE succeeds)"
fi

echo
echo "[submit] Watch progress with:  squeue --me"
echo "[submit] Tail the log:         tail -f euler/logs/sample-${JOB_SAMPLE:-<id>}.out"
