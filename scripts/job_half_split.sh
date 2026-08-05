#!/bin/bash -l
#
# SUPERVISOR EXPERIMENT 1: split the sample in half and train on the RIGHT
# (recent) half while testing on the LEFT (old) half -- deliberately backwards
# in time.
#
# Rationale: under walk-forward, the GFC fold trains on 2000-2004 only, which is
# why it is the weakest fold (PR 0.22). Reversing the arrow of time trains on the
# data-rich modern era (China-2015, 2018, COVID, 2022 = 5 onsets) and tests on
# the largest episodes in the sample (dot-com, GFC, 2010, eurozone = 4 onsets,
# 3067 test windows -- far more test signal than any single fold).
#
# The FORWARD direction over the identical halves is run as the CONTROL, so the
# reversed number is interpretable: same boundary (2013-03-15), same sizes, same
# 63-row embargo, only the arrow of time differs.
#
#   reverse:  test 2000-2012 | val 2013-2015 | train 2016-2026
#   forward:  train 2000-2010 | val 2010-2012 | test 2013-2026
#
# Both val blocks contain positives, so PR-AUC checkpoint selection and Platt
# calibration both work here (unlike 4 of the 6 walk-forward folds).
#
# Tags: half-<direction>-<cell>
#
#SBATCH --job-name=halfsplit
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint='a100'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=$SLURM_ARRAY_TASK_ID
echo "=== halfsplit seed=$SEED on $(hostname) at $(date -Iseconds) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

run () {  # <model> <data-dir> <tag> <extra-flags>
  echo; echo "--- seed=$SEED model=$1 tag=$3 ---"
  python -u scripts/run_experiment.py \
    --model "$1" --task B --seed "$SEED" --epochs 40 $4 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --data-dir "$2" --tag "$3" \
    --checkpoint-dir "$SCRATCH/checkpoints/half-${SLURM_ARRAY_JOB_ID}-${SEED}-$3" \
    --results-dir "$RES" || echo "!! seed=$SEED $1 $3 failed"
}

REGIME_OFF="--use-regime-module 0 --use-regime-attention 0 --gamma 0"
AF_FLAGS="$REGIME_OFF --use-aux 1 --lambda-aux 0.5 --loss focal --focal-gamma 2 --pos-weight 3.4"

for DIR in reverse forward; do
  DATA=~/ra-tft/data/shared/processed/taskb_half_${DIR}

  run ra-tft "$DATA" "half-${DIR}-M0" "$REGIME_OFF"
  run ra-tft "$DATA" "half-${DIR}-AF" "$AF_FLAGS"
  run lstm   "$DATA" "half-${DIR}-lstm" ""

  # XGBoost is deterministic and slow (~9 min); one seed is enough.
  if [ "$SEED" -eq 0 ]; then
    run xgboost "$DATA" "half-${DIR}-xgb" ""
  fi
done
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
