#!/usr/bin/env bash
# MAE pretraining, single GPU. ~400k steps is days of wall clock, so set
# --time to whatever your partition allows and resume with
# `trainer.resume_from_checkpoint=<ckpt> trainer.warm_restart=<bool>`.
#
#   sbatch scripts/slurm/examples/pretrain.sh
#   sbatch -p gpu-long scripts/slurm/examples/pretrain.sh trainer.max_steps=1000
#
# No --partition/--account here: Slurm's defaults apply, and `sbatch -p ...`
# overrides the header. Output goes to slurm-<jobid>.out in the submit dir.
#SBATCH --job-name=mae-pretrain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=48:00:00

set -eo pipefail

# sbatch runs the job in the directory it was submitted from; these examples
# assume that is the repo root.
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Secrets + shared env (W&B, HF, Earthdata, caches). See .env.example.
set -a
if [ -f .env ]; then source .env; fi
set +a

uv sync

# Everything after the script name on the sbatch command line is forwarded as a
# Hydra override, e.g. `sbatch ... pretrain.sh data.weights=[384,43,85]`.
srun --export=ALL uv run python scripts/pretrain.py \
  experiment=pretrain/pretrain_xc \
  "$@"
