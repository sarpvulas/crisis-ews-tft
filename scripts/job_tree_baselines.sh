#!/bin/bash -l
#
# STRONGER TREE BASELINES — add modern gradient-boosted / ensemble trees beyond
# the single XGBoost, on the 6-fold embargoed single-market dataset, so the
# deep-vs-tree comparison is against a competitive tree field (the journal-level
# ask). Models: xgboost (deterministic), plessis-rf (RandomForest), histgb
# (sklearn HistGradientBoosting, LightGBM-style). RF/HistGB have RNG -> 5 seeds;
# XGBoost on seed0 only. Platt calibration. Tags tree-xgb / tree-rf / tree-histgb.
#
#SBATCH --job-name=treebase
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint='a100'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
SEED=$SLURM_ARRAY_TASK_ID
echo "=== treebase seed=$SEED on $(hostname) at $(date -Iseconds) ==="

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10_emb63_6f
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

run () {  # <model> <tag-prefix>
  for FOLD in gfc eurozone china2015 selloff2018 covid bear2022; do
    FDIR="$SCRATCH/fv_tree/${SLURM_ARRAY_JOB_ID}-${SEED}/$FOLD"
    mkdir -p "$FDIR"
    for SP in train val test; do ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"; done
    echo; echo "--- seed=$SEED fold=$FOLD model=$1 ---"
    python -u scripts/run_experiment.py \
      --model "$1" --task B --seed "$SEED" \
      --encoder-steps 252 --decoder-steps 63 --device cuda \
      --posthoc-calibrate platt \
      --data-dir "$FDIR" --tag "$2-$FOLD" \
      --checkpoint-dir "$SCRATCH/checkpoints/$2-${SLURM_ARRAY_JOB_ID}-${SEED}-$FOLD" \
      --results-dir "$RES" || echo "!! seed=$SEED $1 $FOLD failed"
  done
}

run plessis-rf tree-rf
run histgb     tree-histgb
if [ "$SEED" = "0" ]; then run xgboost tree-xgb; fi
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
