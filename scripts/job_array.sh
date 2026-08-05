#!/bin/bash -l
#
# Array training script — one task per seed (or per HP-sweep agent).
#
# Usage:
#   sbatch --array=0-9 scripts/job_array.sh --model ra-tft --task B --epochs 50
#
# Each array task gets a unique seed = $SLURM_ARRAY_TASK_ID. For wandb sweeps,
# set --sweep <sweep_id> and the task ID is used as the agent index instead.

#SBATCH --job-name=ra-tft-array
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail

echo "=== Array job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "Args: $@"

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CKPT_DIR=$SCRATCH/checkpoints/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID/task$TASK_ID
mkdir -p "$CKPT_DIR" "$SCRATCH/wandb" "$SCRATCH/results"

export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

trap 'echo "[$(date -Iseconds)] received USR1 (preemption), exiting cleanly"; exit 0' USR1

cd ~/ra-tft

# If user passes --sweep <sweep_id>, run a wandb sweep agent instead of a seed.
if [[ " $* " == *" --sweep "* ]]; then
    SWEEP_ID=$(echo "$@" | sed -n 's/.*--sweep \([^ ]*\).*/\1/p')
    OTHER_ARGS=$(echo "$@" | sed 's/--sweep [^ ]* *//')
    echo "Running wandb sweep agent: $SWEEP_ID"
    wandb agent --count 1 "$SWEEP_ID"
else
    python -u scripts/run_experiment.py \
        --seed "$TASK_ID" \
        --checkpoint-dir "$CKPT_DIR" \
        --results-dir "$SCRATCH/results" \
        --resume-latest \
        "$@"
fi

echo "=== Task $TASK_ID finished at $(date -Iseconds) ==="
