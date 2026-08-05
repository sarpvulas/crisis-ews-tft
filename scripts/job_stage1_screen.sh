#!/bin/bash -l
#
# Stage-1 single-seed screen on REAL Task B data.
# Each architecture trains 30 epochs at default HPs. Result JSONs land in
# /scratch/.../results/ and feed Stage-2 (HP sweep) for survivors.

#SBATCH --job-name=stage1-screen
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%j.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail
echo "=== Stage-1 screen on $(hostname) at $(date -Iseconds) ==="
nvidia-smi | head -10 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=$SCRATCH/data/processed
CKPT=$SCRATCH/checkpoints/stage1-screen-$SLURM_JOB_ID
RESULTS=$SCRATCH/results
mkdir -p "$CKPT" "$RESULTS" "$SCRATCH/features/topology"
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

cd ~/ra-tft

trap 'echo "preempted, exiting"; exit 0' USR1

# Common arguments. Task B uses 252-step encoder / 63-step decoder per the
# project default; we keep that for fair comparison across models.
COMMON="--task B --seed 0 --epochs 30 --batch-size 64 --encoder-steps 252 --decoder-steps 63
        --data-dir $DATA --device cuda --tag stage1-screen
        --checkpoint-dir $CKPT --results-dir $RESULTS"

run () {
    local name=$1; shift
    echo
    echo "=========================="
    echo "MODEL: $name"
    echo "=========================="
    python -u scripts/run_experiment.py --model "$name" $COMMON "$@"
}

# Strong baselines first (fast).
run "logistic" || true
run "xgboost"  || true
run "lstm"     || true
run "tft"      || true
run "ra-tft"   || true
run "itransformer" || true

# Novel architectures.
run "topo-ratft" --topo-cache-dir "$SCRATCH/features/topology" || true
run "tdt" || true

echo
echo "=== Stage-1 screen complete at $(date -Iseconds) ==="
ls -la "$RESULTS"/*stage1-screen* | tail -20
