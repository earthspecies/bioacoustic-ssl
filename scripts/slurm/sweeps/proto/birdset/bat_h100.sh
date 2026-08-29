#!/usr/bin/env bash
#SBATCH --job-name=JM-5min
#SBATCH --partition=h100-80
#SBATCH --output=../logs/job_manager/%j.log
#SBATCH --time=05:00
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# Frozen prototypical probing of the published BAT ViT-B/16 (AudioSet-2M) encoder
# across the 8 BirdSet downstream tasks. Fans the per-dataset GPU jobs out via
# submitit.
#
#   sbatch scripts/slurm/sweeps/proto/birdset/bat_h100.sh
#   sbatch scripts/slurm/sweeps/proto/birdset/bat_h100.sh seed=2
#
# Unlike the birdmae/audiomae sweeps there is no checkpoint-conversion step: BAT
# is not loadable into ViTEncoder, so `proto/bat` pulls the pinned Hub weights at
# construction with strict=True. The prefetch below warms the shared HF cache and
# strict-loads once, so the fan-out neither races on the download nor discovers a
# broken checkpoint eight times in parallel.
#
# Sizing: 513 tokens against the other arms' 257, and BAT's forward stacks all 12
# layers' tokens (12, B, 512, 768) — the dominant allocation. The encoder is
# frozen, so no backward graph is retained through it; at train batch 256 that
# still means roughly 10-15 GB, which fits h100-80 with plenty of room and should
# fit an a100-40 (`hydra/launcher=gpu`). If it does not, drop
# `data.loaders.train.batch_size` — the linear LR-scaling rule in birdset_eval.py
# compensates automatically, though it does change the probe's gradient noise.

set -eo pipefail

# Load secrets + shared env from the repo .env (see .env.example).
set -a
source "$HOME/soundscape_mae/.env"
set +a

cd ~/soundscape_mae

mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv sync

# Warm the HF cache and validate the strict load before fanning out.
uv run python -c "
from soundscape_ssl.models.architectures.bat import BAT_REPO_ID, BAT_REVISION, BatModel
model = BatModel.from_pretrained(BAT_REPO_ID, revision=BAT_REVISION)
print(f'BAT {BAT_REVISION[:8]} loaded: {sum(p.numel() for p in model.parameters()):,} params')
"

srun uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/proto/birdset/bat \
  hydra/launcher=gpu_h100 \
  "$@"
