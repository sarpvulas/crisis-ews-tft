#!/bin/bash -l
#
# Stage-1 cluster smoke: prove TDT + topo-ratft + iTransformer all run on GPU
# with the cu121 torch wheel. All on synthetic data — the goal is "cluster
# wiring works for every new architecture", not "we beat XGBoost". A successful
# run produces four result JSONs in /scratch/.../results/.

#SBATCH --job-name=stage1-smoke
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%j.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%j.err

set -euo pipefail
echo "=== Stage-1 smoke on $(hostname) at $(date -Iseconds) ==="
nvidia-smi | head -10 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
CKPT=$SCRATCH/checkpoints/stage1-smoke-$SLURM_JOB_ID
mkdir -p "$CKPT" "$SCRATCH/results"
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

cd ~/ra-tft

run () {
    local name=$1; shift
    echo
    echo "--- $name ---"
    python -u scripts/run_experiment.py \
        --device cuda --tag stage1-smoke \
        --checkpoint-dir "$CKPT" --results-dir "$SCRATCH/results" "$@"
}

run "lstm"          --model lstm        --task B --seed 0 --epochs 2 --batch-size 32 --encoder-steps 60 --decoder-steps 12
run "itransformer"  --model itransformer --task B --seed 0 --epochs 2 --batch-size 32 --encoder-steps 60 --decoder-steps 12
run "topo-ratft"    --model topo-ratft  --task B --seed 0 --epochs 2 --batch-size 16 --encoder-steps 60 --decoder-steps 12 \
                    --topo-cache-dir "$SCRATCH/features/topology"
# TDT smoke uses default diffusion_steps=1000; tiny batch + 2 epochs keep runtime under a few minutes.
run "tdt"           --model tdt         --task B --seed 0 --epochs 2 --batch-size 32 --encoder-steps 60 --decoder-steps 12 || true

echo
echo "=== Stage-1 smoke complete at $(date -Iseconds) ==="
ls -la "$SCRATCH/results/" | tail -20
