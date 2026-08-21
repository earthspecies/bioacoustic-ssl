#!/usr/bin/env bash
#SBATCH --job-name=JM-wrapup2
#SBATCH --partition=a100-40
#SBATCH --output=../logs/job_manager/%j.log
#SBATCH --time=20:00
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# TIER 2 — finish the BEANS grid for the six live arms.
#
# 18 GPU jobs, ~45 GPU-h. BEANS is the only benchmark where PAM pretraining has
# ever looked positive, and the grid it is claimed from is 16 of 36 cells. In
# particular `XC (1M)` — the best backbone we have, and the XC-only control the
# cross-domain claim needs at the 1M schedule — has no BEANS number at all.
#
#   vit_missing_arms  {XC 1M, +NASA rg50 400k}                 x 6 tasks = 12
#   vit_bats          {XC 400k, +NASA ev 400k, +NASA rg 400k}  x bats    =  3
#   vit_cbi           {+NASA ev 400k, +NASA ev 1M}             x cbi     =  2
#   proto/vit_cbi     {+NASA ev 400k}                          x cbi     =  1
#
# The two `bats` cells for XC (400k) and +NASA rg (400k) are re-submissions: both
# crashed on 2026-08-19 with a cuDNN load failure from a broken per-job venv on
# /scratch (slurm-8x-a100-40gb-2), not a data or config fault. If it recurs,
# `hydra.launcher.exclude=<node>` is the escape hatch.
#
# See tier1 for why these run in the background with per-batch sweep dirs.

set -eo pipefail

# Load secrets + shared env from the repo .env (see .env.example).
set -a
source "$HOME/soundscape_mae/.env"
set +a

cd ~/soundscape_mae

mkdir -p logs

uv sync

uv run python scripts/beans_eval.py --multirun \
  experiment=sweeps/layerwise/beans/vit_missing_arms \
  hydra/launcher=gpu \
  hydra.sweep.dir=multirun/wrapup/t2_beans_missing_arms &

uv run python scripts/beans_eval.py --multirun \
  experiment=sweeps/layerwise/beans/vit_bats \
  hydra/launcher=gpu \
  hydra.sweep.dir=multirun/wrapup/t2_beans_bats &

uv run python scripts/beans_eval.py --multirun \
  experiment=sweeps/layerwise/beans/vit_cbi \
  hydra/launcher=gpu \
  hydra.sweep.dir=multirun/wrapup/t2_beans_cbi &

uv run python scripts/beans_eval.py --multirun \
  experiment=sweeps/proto/beans/vit_cbi \
  hydra/launcher=gpu \
  hydra.sweep.dir=multirun/wrapup/t2_beans_proto_cbi &

wait
