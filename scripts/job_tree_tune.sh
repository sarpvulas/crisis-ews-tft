#!/bin/bash -l
#
# TREE HYPERPARAMETER TUNING — make the deep-vs-tree comparison fair. Grid of 3
# configs each for XGBoost / RandomForest / HistGradientBoosting on the 6-fold
# embargoed single-market data. Best config per model is selected on VALIDATION
# PR-AUC (see tree_tune_agg.py), then its TEST score reported. One array task =
# one (model,config) across all 6 folds. Tags tune-<m>-c<idx>-<fold>.
#
#SBATCH --job-name=treetune
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint='a100'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
echo "=== treetune cfg=$SLURM_ARRAY_TASK_ID on $(hostname) at $(date -Iseconds) ==="
module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10_emb63_6f
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

# idx -> "model tagprefix flags"
CFG=(
  "xgboost xgb-c0 --max-depth 3 --xgb-lr 0.05 --n-estimators 400"
  "xgboost xgb-c1 --max-depth 5 --xgb-lr 0.05 --n-estimators 400"
  "xgboost xgb-c2 --max-depth 7 --xgb-lr 0.10 --n-estimators 600"
  "plessis-rf rf-c0 --max-depth 12 --min-samples-leaf 2 --n-estimators 400"
  "plessis-rf rf-c1 --max-depth 24 --min-samples-leaf 2 --n-estimators 400"
  "plessis-rf rf-c2 --max-depth 40 --min-samples-leaf 1 --n-estimators 600"
  "histgb hgb-c0 --xgb-lr 0.03 --n-estimators 500"
  "histgb hgb-c1 --xgb-lr 0.05 --n-estimators 500"
  "histgb hgb-c2 --xgb-lr 0.10 --n-estimators 800"
)
read -r MODEL TAGP FLAGS <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
echo "model=$MODEL tag=$TAGP flags=$FLAGS"

for FOLD in gfc eurozone china2015 selloff2018 covid bear2022; do
  FDIR="$SCRATCH/fv_ttune/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}/$FOLD"
  mkdir -p "$FDIR"
  for SP in train val test; do ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"; done
  echo; echo "--- $TAGP fold=$FOLD ---"
  python -u scripts/run_experiment.py \
    --model "$MODEL" --task B --seed 0 $FLAGS \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --data-dir "$FDIR" --tag "tune-$TAGP-$FOLD" \
    --checkpoint-dir "$SCRATCH/checkpoints/ttune-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-$FOLD" \
    --results-dir "$RES" || echo "!! $TAGP $FOLD failed"
done
echo; echo "=== cfg=$SLURM_ARRAY_TASK_ID done at $(date -Iseconds) ==="
