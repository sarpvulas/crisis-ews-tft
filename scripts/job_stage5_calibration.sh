#!/bin/bash -l
#
# Stage-5 calibration sweep — refit each Stage-3 multiseed config with
# --posthoc-calibrate platt. Raw test metrics stay identical to Stage 3;
# `calibrated_test_metrics` is added to each result JSON.
#
# 7 models × 10 seeds = 70 jobs, ~1-3 min each on a40/a100.
#
# Usage:
#   sbatch --array=0-69 scripts/job_stage5_calibration.sh
#
# Array layout (i = SLURM_ARRAY_TASK_ID):
#   model = MODELS[i // 10]
#   seed  = i % 10
#
# Per-model extra args mirror Stage-3 best HPs.

#SBATCH --job-name=stage5-calibration
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
IDX=${SLURM_ARRAY_TASK_ID:-0}
MODELS=(xgboost lstm plessis-rf itransformer ra-tft tdt-xgb tdt)
MODEL=${MODELS[$((IDX / 10))]}
SEED=$((IDX % 10))

echo "=== Stage-5 calibration model=$MODEL seed=$SEED ==="
nvidia-smi | head -10 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
CKPT=$SCRATCH/checkpoints/stage5-calibration-$SLURM_ARRAY_JOB_ID/$MODEL/seed$SEED
mkdir -p "$CKPT" "$SCRATCH/results"
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

trap 'echo "preempted, exiting"; exit 0' USR1
cd ~/ra-tft

# Per-model HP tail matching Stage 3 (defaults are fine for most models).
EXTRA_ARGS=()
case "$MODEL" in
  tdt|tdt-xgb)
    # Stage 2 HP winners (same as Stage 3 used).
    EXTRA_ARGS=(--lr 1e-3 --d-model 64 --e-layers 4 --balanced-sampler 1)
    ;;
esac

python -u scripts/run_experiment.py \
    --model "$MODEL" --task B --seed "$SEED" \
    --epochs 50 --batch-size 64 \
    --encoder-steps 252 --decoder-steps 63 \
    --data-dir "$SCRATCH/data/processed" \
    --device cuda \
    --tag "stage5-platt" \
    --checkpoint-dir "$CKPT" \
    --results-dir "$SCRATCH/results" \
    --posthoc-calibrate platt \
    "${EXTRA_ARGS[@]}"
