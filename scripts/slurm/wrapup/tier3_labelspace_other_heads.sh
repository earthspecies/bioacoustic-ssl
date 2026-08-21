#!/usr/bin/env bash
#SBATCH --job-name=JM-wrapup3
#SBATCH --partition=h100-80
#SBATCH --output=../logs/job_manager/%j.log
#SBATCH --time=20:00
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# TIER 3 — put the OTHER three heads on the repaired label space too.
#
# 21 GPU jobs, ~44 GPU-h. Tier 1 fixes the layerwise table; this fixes the
# protocol section, whose whole point is comparing heads on identical weights.
# Right now layerwise-finetune ran after the 2026-08-20 repair while final-layer,
# linear and finetune ran before it, so the head ordering is read partly across a
# label-space change on NES/PER/SNE.
#
#   proto/vit_relabel       {+NASA ev 400k}      x 3 tasks = 3   (the only live
#                                                                arm with an FL probe)
#   proto/audiomae_relabel  AudioMAE             x 3 tasks = 3
#   proto/birdmae_relabel   Bird-MAE-Base        x 3 tasks = 3
#   linear/vit_relabel      {XC 1M, +NASA ev 1M} x 3 tasks = 6
#   finetune/vit_relabel    {XC 1M, +NASA ev 1M} x 3 tasks = 6
#
# H100 for all of it: the finetune batch needs it (ViT-B backward at batch_size
# 256 does not fit an A100-40), and the rest is faster there.
#
# See tier1 for why these run in the background with per-batch sweep dirs.

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
uv run python scripts/convert_external_ckpt.py audiomae
uv run python scripts/convert_external_ckpt.py birdmae_base

uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/proto/birdset/vit_relabel \
  data/loaders=default \
  hydra/launcher=gpu_h100 \
  hydra.sweep.dir=multirun/wrapup/t3_proto_vit_relabel &

uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/proto/birdset/audiomae_relabel \
  hydra/launcher=gpu_h100 \
  hydra.sweep.dir=multirun/wrapup/t3_proto_audiomae_relabel &

uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/proto/birdset/birdmae_relabel \
  hydra/launcher=gpu_h100 \
  hydra.sweep.dir=multirun/wrapup/t3_proto_birdmae_relabel &

uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/linear/birdset/vit_relabel \
  data/loaders=default \
  hydra/launcher=gpu_h100 \
  hydra.sweep.dir=multirun/wrapup/t3_linear_vit_relabel &

uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/finetune/birdset/vit_relabel \
  data/loaders=default \
  hydra/launcher=gpu_h100 \
  hydra.sweep.dir=multirun/wrapup/t3_finetune_vit_relabel &

wait
