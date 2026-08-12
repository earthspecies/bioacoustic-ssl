#!/usr/bin/env bash
#SBATCH --job-name=JM-5min
#SBATCH --partition=a100-40
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


uv sync

srun uv run python scripts/beans_eval.py --multirun \
  experiment=sweeps/layerwise/beans/vit \
  hydra/launcher=gpu