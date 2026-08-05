#!/bin/bash -l
#
# Stage-3 multi-seed: same architecture at the same HPs, 10 seeds.
#
# Usage:
#   sbatch --array=0-9 scripts/job_stage3_multiseed.sh <model> [extra-args ...]
#
# Each task uses --seed = $SLURM_ARRAY_TASK_ID. The remaining args are passed
# verbatim to run_experiment.py — that's where the best HPs from Stage 2 go.

#SBATCH --job-name=stage3-multiseed
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail
MODEL=${1:?usage: $0 <model> [extra args]}
shift
SEED=${SLURM_ARRAY_TASK_ID:-0}

echo "=== Stage-3 multiseed model=$MODEL seed=$SEED ==="
nvidia-smi | head -10 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
CKPT=$SCRATCH/checkpoints/stage3-$SLURM_ARRAY_JOB_ID/$MODEL/seed$SEED
mkdir -p "$CKPT" "$SCRATCH/results"
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

trap 'echo "preempted, exiting"; exit 0' USR1
cd ~/ra-tft

python -u scripts/run_experiment.py \
    --model "$MODEL" --task B --seed "$SEED" \
    --epochs 50 --batch-size 64 \
    --encoder-steps 252 --decoder-steps 63 \
    --data-dir "$SCRATCH/data/processed" \
    --device cuda \
    --tag "stage3-multiseed" \
    --checkpoint-dir "$CKPT" \
    --results-dir "$SCRATCH/results" \
    "$@"
