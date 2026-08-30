#!/usr/bin/env bash
# One downstream probe on BirdSet HSN — the repo's default arm: block-diagonal
# layerwise prototypical head on a frozen encoder.
#
#   sbatch scripts/slurm/examples/birdset_hsn.sh
#   # a real backbone rather than a random one:
#   sbatch scripts/slurm/examples/birdset_hsn.sh \
#     'trainer.resume_from_checkpoint=${paths.ckpt_dir}/XC_1M.ckpt'
#   # another task, another head:
#   sbatch scripts/slurm/examples/birdset_hsn.sh \
#     data/datasets=train/birdset/per module/model=proto/vit
#
# HSN is 2500 steps (`configs/data/datasets/train/birdset/hsn.yaml`), well
# under an hour on one modern GPU once the audio is cached.
#SBATCH --job-name=birdset-hsn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -eo pipefail

cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

set -a
if [ -f .env ]; then source .env; fi
set +a

uv sync

# Dataloader workers are a per-machine number: `data/loaders=default` is sized
# for a laptop, {a100,h100} for a fat node. Match it to --cpus-per-task.
srun uv run python scripts/birdset_eval.py \
  data.loaders.train.num_workers=8 \
  data.loaders.val_test.num_workers=8 \
  "$@"
