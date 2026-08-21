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
# FALLBACK BATCH — do not run this unless the GPU-hours are going spare. 21 GPU
# jobs, ~44 GPU-h, and the gap it closes can be closed for free instead: once
# tier 1 has landed, every layerwise cell logs test_AP_per_class, and cmAP is a
# mean over exactly those, so dropping the restored class down-converts a
# post-repair run onto the pre-repair space EXACTLY. That puts all four heads on
# one label space at no compute cost.
#
# Run this only if the head comparison is needed on AUROC or T1 as well — those
# are not per-class means in this logging, so they cannot be down-converted — or
# if the head section must be quoted on BirdSet's official 89/132/56 space rather
# than the one-class-short one. See
# .scratch/publish-readiness/issues/01-missing-probes.md.
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
