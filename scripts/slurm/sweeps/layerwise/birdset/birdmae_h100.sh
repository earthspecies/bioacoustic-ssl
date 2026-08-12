#!/usr/bin/env bash
#SBATCH --job-name=JM-5min
#SBATCH --partition=h100-80
#SBATCH --output=../logs/job_manager/%j.log
#SBATCH --time=05:00
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# Layerwise prototypical probing of the published Bird-MAE-Base encoder across
# the 8 BirdSet downstream tasks. Converts the HF weights once here, then fans
# the per-dataset GPU jobs out via submitit. See docs/external_baselines.md.

set -eo pipefail

# Load secrets + shared env from the repo .env (see .env.example).
set -a
source "$HOME/soundscape_mae/.env"
set +a

cd ~/soundscape_mae

mkdir -p logs checkpoints

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv sync

# Convert before fanning out, so the child jobs never race on the checkpoint.
# strict-loads into ViTEncoder, so a geometry mismatch fails here.
uv run python scripts/convert_external_ckpt.py birdmae_base

srun uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/layerwise/birdset/birdmae \
  hydra/launcher=gpu_h100
