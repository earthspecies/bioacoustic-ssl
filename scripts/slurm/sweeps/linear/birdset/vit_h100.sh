#!/usr/bin/env bash
#SBATCH --job-name=JM-5min
#SBATCH --partition=h100-80
#SBATCH --output=../logs/job_manager/%j.log
#SBATCH --time=05:00
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# Load secrets + shared env from the repo .env (see .env.example).
set -a
source "$HOME/soundscape_mae/.env"
set +a

cd ~/soundscape_mae

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv sync

srun uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/linear/birdset/vit \
  data/loaders=default \
  hydra/launcher=gpu_h100
