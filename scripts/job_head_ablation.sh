#!/bin/bash -l
#
# SUPERVISOR EXPERIMENT 2: does the second head earn its place?
#
# The contribution (AF) is a TFT with BOTH a classification head and an
# auxiliary forward-drawdown regression head. The supervisor's question: is the
# pair actually better than either head alone -- or is the classifier the weak
# part, so that simply SWAPPING it for a regressor is enough?
#
# Clean 3-way, all cells sharing the same backbone (regime branch OFF) and the
# same focal/pw3.4 objective wherever a classification term exists, so the ONLY
# thing that varies is which head(s) exist:
#
#   conf-CLFONLY  classification head only   (focal/pw3.4, no aux)   <- NEW
#   conf-AFnoreg  both heads                 (already run, 30 seeds, job 35861786)
#   conf-REGONLY  regression head only       (--beta 0, scored via aux) <- NEW
#
# conf-M0 (vanilla, plain BCE/pw10) already exists as the untuned reference.
# CLFONLY is the control that holds the LOSS fixed: conf-M0 differs from AFnoreg
# in two ways at once (no aux AND a different loss), so it cannot isolate the head.
#
# REGONLY notes:
#   --beta 0 removes the classification term from the objective entirely, so only
#   the drawdown MSE trains. --predict-from aux scores through the regression
#   head, mapped to a probability by a 1-D logistic FITTED ON THE TRAIN SPLIT --
#   the direction is learned, not assumed, because the drawdown/label relation
#   reverses between crisis-excluded (train/val) and all-window (test)
#   populations. Checkpoint selection falls back to the aux MSE, since a
#   classifier PR-AUC would be meaningless for an untrained classification head.
#
# Tags: conf-CLFONLY-<fold> / conf-REGONLY-<fold> -- same conf- family and the
# same 6 folds, so these merge straight into the existing aggregation.
#
#SBATCH --job-name=headabl
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
echo "=== headabl seed=$SEED on $(hostname) at $(date -Iseconds) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10_emb63_6f
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

run () {  # <fold-dir> <fold> <tag-suffix> <extra-flags>
  echo; echo "--- seed=$SEED fold=$2 tag=$3 ---"
  python -u scripts/run_experiment.py \
    --model ra-tft --task B --seed "$SEED" --epochs 40 $4 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --data-dir "$1" --tag "conf-$3-$2" \
    --checkpoint-dir "$SCRATCH/checkpoints/headabl-${SLURM_ARRAY_JOB_ID}-${SEED}-$3-$2" \
    --results-dir "$RES" || echo "!! seed=$SEED $3 $2 failed"
}

REGIME_OFF="--use-regime-module 0 --use-regime-attention 0 --gamma 0"

for FOLD in gfc eurozone china2015 selloff2018 covid bear2022; do
  FDIR="$SCRATCH/fv_head/${SLURM_ARRAY_JOB_ID}-${SEED}/$FOLD"
  mkdir -p "$FDIR"
  for SP in train val test; do ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"; done

  run "$FDIR" "$FOLD" CLFONLY "$REGIME_OFF --use-aux 0 --loss focal --focal-gamma 2 --pos-weight 3.4"
  # --dump-preds on REGONLY so the aggregation can score the SAME model in both
  # score directions. The train-fitted map is monotone, so negating the dumped
  # probabilities recovers the opposite ranking exactly -- this makes the result
  # immune to the objection "you just picked the wrong sign".
  run "$FDIR" "$FOLD" REGONLY "$REGIME_OFF --use-aux 1 --lambda-aux 0.5 --beta 0 --predict-from aux --dump-preds"
done
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
