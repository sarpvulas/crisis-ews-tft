#!/bin/bash -l
#
# Stage-5 vanilla TDT — pure generative diffusion baseline, no class head.
# 10 seeds at the same lr=1e-3 used by the Stage-4 diagnostic (lowest variance).
#
# Strips every component the Stage-4 ablation showed contributes positively, so
# the comparison vs full TDT directly measures the value of our additions.
#
# Usage:  sbatch --array=0-9 scripts/job_stage5_vanilla_tdt.sh

#SBATCH --job-name=stage5-vanilla-tdt
#SBATCH --partition=interruptible_gpu
#SBATCH --constraint=a100|a40
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail
SEED=${SLURM_ARRAY_TASK_ID:-0}

echo "=== Stage-5 vanilla TDT seed=$SEED ==="
nvidia-smi | head -10 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
CKPT=$SCRATCH/checkpoints/stage5-vanilla-tdt-$SLURM_ARRAY_JOB_ID/seed$SEED
mkdir -p "$CKPT" "$SCRATCH/results"
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

trap 'echo "preempted, exiting"; exit 0' USR1
cd ~/ra-tft

# Vanilla TDT = no class head, no joint loss, no balanced sampler, generative scoring.
python -u scripts/run_experiment.py \
    --model tdt --task B --seed "$SEED" \
    --epochs 50 --batch-size 64 --lr 1e-3 \
    --encoder-steps 252 --decoder-steps 63 \
    --data-dir "$SCRATCH/data/processed" \
    --device cuda \
    --tag "stage5-vanilla-tdt" \
    --checkpoint-dir "$CKPT" \
    --results-dir "$SCRATCH/results" \
    --lambda-clf 0 \
    --use-class-head 0 \
    --predict-via likelihood_ratio \
    --balanced-sampler 0 \
    --cond-drop-prob 0.1
