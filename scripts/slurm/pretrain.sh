#!/usr/bin/env bash
#SBATCH --job-name=pretrain
#SBATCH --partition=h100-80
#SBATCH --output=logs/pretrain/mae_terrestial_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52 # 26
#SBATCH --gres=gpu:1
#SBATCH --mem=300G

# Load secrets + shared env from the repo .env (see .env.example).
set -a
source "$HOME/soundscape_mae/.env"
set +a

cd ~/soundscape_mae

uv sync

srun --export=ALL uv run python scripts/pretrain.py \
    experiment=pretrain/pretrain_xc_pam_new