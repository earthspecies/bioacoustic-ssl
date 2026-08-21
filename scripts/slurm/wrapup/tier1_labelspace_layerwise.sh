#!/usr/bin/env bash
#SBATCH --job-name=JM-wrapup1
#SBATCH --partition=h100-80
#SBATCH --output=../logs/job_manager/%j.log
#SBATCH --time=20:00
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# TIER 1 — put the whole layerwise BirdSet table on one label space.
#
# 21 GPU jobs, ~32 GPU-h. This is the batch the headline table cannot be
# published without: the 2026-08-20 NES/PER/SNE repair means 3 of the 7 test
# tasks currently mix 88/131/55-class and 89/132/56-class runs, so the reported
# test mean averages over two different label spaces (see
# .scratch/publish-readiness/issues/01-missing-probes.md).
#
#   vit_relabel         seed 1  x {XC 1M, +NASA ev 400k, +NASA ev 1M} x 3 tasks =  9
#   vit_relabel_seed2   seed 2  x {XC 1M, +NASA ev 1M}                x 3 tasks =  6
#   audiomae_relabel    seed 1  x AudioMAE                            x 3 tasks =  3
#   birdmae_relabel     seed 1  x Bird-MAE-Base                       x 3 tasks =  3
#
# Each batch is submitted as its own submitit array at array_parallelism=3, so
# expect up to 12 concurrent GPU jobs. Append
# `hydra.launcher.array_parallelism=N` to a line to change that.
#
# The four submissions run in the background rather than under `srun` so they do
# not serialise on this job's single task slot, and each gets its own
# `hydra.sweep.dir` so their multirun dirs cannot collide. This job only needs to
# live long enough to sbatch them: hydra then blocks waiting for results and is
# killed by the 20 min limit, which the already-queued array jobs survive.

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
  experiment=sweeps/layerwise/birdset/vit_relabel \
  data/loaders=default \
  hydra/launcher=gpu_h100 \
  hydra.sweep.dir=multirun/wrapup/t1_vit_relabel &

uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/layerwise/birdset/vit_relabel_seed2 \
  data/loaders=default \
  hydra/launcher=gpu_h100 \
  hydra.sweep.dir=multirun/wrapup/t1_vit_relabel_seed2 &

uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/layerwise/birdset/audiomae_relabel \
  hydra/launcher=gpu_h100 \
  hydra.sweep.dir=multirun/wrapup/t1_audiomae_relabel &

uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/layerwise/birdset/birdmae_relabel \
  hydra/launcher=gpu_h100 \
  hydra.sweep.dir=multirun/wrapup/t1_birdmae_relabel &

wait
