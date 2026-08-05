#!/bin/bash -l
#
# Multi-seed CLEAN comparison on the shared trough-labeled Task B.
# Job ARRAY: one seed per array task (own GPU) → all seeds run in parallel.
# Leak fix + best-checkpoint restore are in the synced code.
#
#SBATCH --job-name=trough-ms
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint='a100|a40'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
SEED=$SLURM_ARRAY_TASK_ID
echo "=== trough-ms seed=$SEED on $(hostname) at $(date -Iseconds) ==="
nvidia-smi | head -10 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough
CKPT=$SCRATCH/checkpoints/trough-ms-${SLURM_ARRAY_JOB_ID}-${SEED}   # per-seed, no collisions
RES=$SCRATCH/results
mkdir -p "$CKPT" "$RES"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

run () {  # $1=model  $2=extra flags
  echo; echo "--- seed=$SEED $1 ---"
  python -u scripts/run_experiment.py \
    --model "$1" --task B --seed "$SEED" $2 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --data-dir "$DATA" --tag shared-trough-ms \
    --checkpoint-dir "$CKPT" --results-dir "$RES" || echo "!! seed=$SEED $1 failed"
}

run xgboost ""
run lstm    "--epochs 40"
run ra-tft  "--epochs 40"
run tdt     "--epochs 40"

echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
