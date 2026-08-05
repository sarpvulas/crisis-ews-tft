#!/bin/bash -l
#
# Stage-0 cluster smoke: three quick tests (lstm, topo-ratft, xgboost) in one
# job to amortize queue wait. Runs on interruptible_gpu so it schedules fast
# (and exercises the same partition production runs will use).
#
# All tests run on synthetic data — the goal is "harness works end-to-end on
# GPU", not "we beat XGBoost." A successful run produces three result JSONs
# under /scratch/users/<user>/ra-tft/results/.

#SBATCH --job-name=stage0-smoke
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%j.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%j.err

set -euo pipefail

echo "=== Stage-0 smoke on $(hostname) at $(date -Iseconds) ==="
nvidia-smi | head -10 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
CKPT=$SCRATCH/checkpoints/stage0-smoke-$SLURM_JOB_ID
mkdir -p "$CKPT" "$SCRATCH/results"

export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

cd ~/ra-tft

echo
echo "--- 1/3 LSTM smoke ---"
python -u scripts/run_experiment.py \
    --model lstm --task B --seed 0 --epochs 2 --batch-size 32 \
    --encoder-steps 60 --decoder-steps 12 --device cuda \
    --tag stage0-smoke --checkpoint-dir "$CKPT" --results-dir "$SCRATCH/results"

echo
echo "--- 2/3 topo-ratft smoke ---"
python -u scripts/run_experiment.py \
    --model topo-ratft --task B --seed 0 --epochs 2 --batch-size 16 \
    --encoder-steps 60 --decoder-steps 12 --device cuda \
    --topo-cache-dir "$SCRATCH/features/topology" \
    --tag stage0-smoke --checkpoint-dir "$CKPT" --results-dir "$SCRATCH/results"

echo
echo "--- 3/3 XGBoost smoke ---"
python -u scripts/run_experiment.py \
    --model xgboost --task B --seed 0 \
    --encoder-steps 60 --decoder-steps 12 --device cuda \
    --tag stage0-smoke --checkpoint-dir "$CKPT" --results-dir "$SCRATCH/results"

echo
echo "=== Stage-0 smoke complete at $(date -Iseconds) ==="
ls -la "$SCRATCH/results/" | tail -20
