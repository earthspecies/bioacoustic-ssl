#!/usr/bin/env bash
#SBATCH --job-name=JM-xchead
#SBATCH --partition=h100-80
#SBATCH --output=../logs/job_manager/%j.log
#SBATCH --time=20:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# Train the full Xeno-Canto classification head for the HF model release:
# layerwise prototypical probing of a frozen 1M-step MAE encoder over all 11 737
# XC species, 100 000 steps at batch 256 (~35 epochs of 734 910 recordings).
#
# Two GPU jobs, one per backbone (XC 1M and XC + NASA events 1M). Budget 24-48 h
# each; they run concurrently under the launcher's array_parallelism.
#
# Resources are overridden away from the launcher defaults because this streams
# the WHOLE Xeno-Canto metadata frame, not one BirdSet task's few thousand rows:
# 734 910 rows x 87 columns, copied into every dataloader worker by `spawn`.
# gpu_h100.yaml's 26 CPUs / 120 GB is sized for the benchmark probes; this
# mirrors scripts/slurm/pretrain.sh, which streams the same corpus.
#
# Checkpointing and the log interval are set as single-valued sweeper params in
# the experiment config, not here: train.yaml merges the `trainer` group after
# `experiment`, so trainer/train.yaml beats anything an experiment assigns and
# only a sweeper/CLI override outranks it (the workaround
# sweeps/proto/birdset/bat.yaml documents). Passing them here as well would give
# hydra two values for the same key.
#
# TO RESUME a job that hit the 48 h timeout, relaunch that ONE arm with the
# newest state checkpoint. `resume_from_state` restores head + optimizer +
# scheduler + step; `resume_from_checkpoint` alone would restart from step 0 with
# a freshly initialised head and quietly waste the run:
#
#   uv run python scripts/birdset_eval.py \
#     experiment=xc_classifier/vit_layerwise \
#     trainer.resume_from_checkpoint=$HOME/soundscape_mae/checkpoints/XC_1M.ckpt \
#     trainer.resume_from_state=<multirun>/<job>/checkpoints/step_0075000.ckpt \
#     hydra/launcher=gpu_h100 hydra.launcher.cpus_per_task=52 \
#     hydra.launcher.mem_gb=300
#
# Evaluation is a separate, later step: BirdSet with the 11 737 logits masked
# down to each task's species. metadata/xc_v0.2.0_classes.parquet maps output
# index -> gbifID, and the index is the rank of the gbifID in ascending order.

set -eo pipefail

# Load secrets + shared env from the repo .env (see .env.example).
set -a
source "$HOME/soundscape_mae/.env"
set +a

cd ~/soundscape_mae

mkdir -p logs checkpoints

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv sync

# The label space must exist before the dataset config can read it. Idempotent:
# rebuilds the same 11 737-row table from the same split CSV.
uv run python scripts/build_xc_label_space.py

srun uv run python scripts/birdset_eval.py --multirun \
  experiment=xc_classifier/vit_layerwise \
  hydra/launcher=gpu_h100 \
  hydra.launcher.cpus_per_task=52 \
  hydra.launcher.mem_gb=300 \
  hydra.sweep.dir=multirun/xc_classifier/vit_layerwise
