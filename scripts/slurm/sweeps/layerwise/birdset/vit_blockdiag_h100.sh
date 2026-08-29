#!/usr/bin/env bash
#SBATCH --job-name=JM-blockdiag
#SBATCH --partition=h100-80
#SBATCH --output=../logs/job_manager/%j.log
#SBATCH --time=05:00
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# Benchmark the block-diagonal prototype mixer against the dense one, on all 8
# BirdSet tasks. 16 GPU jobs (2 backbones x 8 tasks, seed 1), ~21 GPU-h.
#
# Why: the full-XC release head had to constrain the readout, because the dense
# mixer is quadratic in num_classes — 2.755 B parameters at 11 737 classes. At
# BirdSet scale both mixers fit, so this is the only place the constraint can be
# priced. Everything except `mixer` is identical to
# sweeps/layerwise/birdset/vit, so the comparison is against the dense runs
# already in W&B rather than against anything this script produces.
#
# What to compare, and when:
#   XC (400k)  comparable the moment this finishes — its dense layerwise row is
#              complete and already on the repaired label space.
#   XC (1M)    5 of 8 tasks comparable now; NES/PER/SNE need the dense side
#              re-probed first by scripts/slurm/wrapup/tier1_labelspace_layerwise.sh,
#              since those three cells are pre-2026-08-20 on that backbone.
#
# Reading the result: the noise floor on the 7-task test mean is <=0.001 cmAP, so
# a |delta| at or below that says the constraint is free and the release head
# needs no caveat. A consistent loss says the release model gives something up
# and the model card should say so — and that the cross-class weights the dense
# mixer learns are doing real work, which is worth reporting either way.
#
# These runs are named `BD-<TASK>-test_5s-...` on purpose. They share
# `module.model._target_` and `freeze_backbone` with the dense layerwise probes,
# so every existing resolver would file them as ordinary `layerwise` cells and,
# being newer, let them supersede the published numbers in its
# most-recent-finished dedup. The prefix breaks `parts[1] == "test_5s"`, so both
# notebook parsers skip them. Pull them by name prefix, or by
# `config.module.model.mixer == "block_diagonal"`.

set -eo pipefail

# Load secrets + shared env from the repo .env (see .env.example).
set -a
source "$HOME/soundscape_mae/.env"
set +a

cd ~/soundscape_mae

mkdir -p logs

uv sync

srun uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/layerwise/birdset/vit_blockdiag \
  data/loaders=default \
  hydra/launcher=gpu_h100 \
  hydra.sweep.dir=multirun/blockdiag/birdset_layerwise
