#!/bin/bash -l
#
# Stage-4 ablation: TDT with one component disabled, 3 seeds.
#
# Submit one array per ablation variant. The "extra-args" tail is where the
# ablation toggle goes — e.g. --lambda-diff 0 for "no diffusion", or
# --lambda-clf 0 for "no discriminative head".
#
# Usage:
#   sbatch --array=0-2 scripts/job_stage4_ablation.sh <variant_name> [extra args ...]

#SBATCH --job-name=stage4-ablation
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
VARIANT=${1:?usage: $0 <variant_name> [extra args]}
shift
SEED=${SLURM_ARRAY_TASK_ID:-0}

echo "=== Stage-4 ablation variant=$VARIANT seed=$SEED ==="
nvidia-smi | head -10 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
CKPT=$SCRATCH/checkpoints/stage4-$SLURM_ARRAY_JOB_ID/$VARIANT/seed$SEED
mkdir -p "$CKPT" "$SCRATCH/results"
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

trap 'echo "preempted, exiting"; exit 0' USR1
cd ~/ra-tft

python -u scripts/run_experiment.py \
    --model tdt --task B --seed "$SEED" \
    --epochs 50 --batch-size 64 \
    --encoder-steps 252 --decoder-steps 63 \
    --data-dir "$SCRATCH/data/processed" \
    --device cuda \
    --tag "stage4-$VARIANT" \
    --checkpoint-dir "$CKPT" \
    --results-dir "$SCRATCH/results" \
    "$@"
