#!/bin/bash -l
#
# Quick single-seed comparison on the SHARED Bloomberg Task-B dataset
# (broad-equity crisis EWS, primary bear2022 split). 4 key models, seed 0.
# Goal: first real comparable numbers + confirm the dataset trains on GPU.
#
#SBATCH --job-name=shared-trough-clean
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint='a100|a40'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%j.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%j.err

set -euo pipefail
echo "=== shared-taskb quick on $(hostname) at $(date -Iseconds) ==="
nvidia-smi | head -10 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough
CKPT=$SCRATCH/checkpoints/shared-trough-clean-$SLURM_JOB_ID
RES=$SCRATCH/results
mkdir -p "$CKPT" "$RES"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

run () {  # $1=model  $2=extra epochs flag
  echo; echo "--- $1 ---"
  python -u scripts/run_experiment.py \
    --model "$1" --task B --seed 0 $2 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --data-dir "$DATA" --tag shared-trough-clean \
    --checkpoint-dir "$CKPT" --results-dir "$RES" || echo "!! $1 failed"
}

run xgboost ""
run lstm    "--epochs 40"
run ra-tft  "--epochs 40"
run tdt     "--epochs 40"

echo; echo "=== done at $(date -Iseconds) ==="
ls -t "$RES"/*shared-trough-clean* 2>/dev/null | head
