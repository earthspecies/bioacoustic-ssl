#!/usr/bin/env bash
# The published Bird-MAE-Base baseline on BirdSet, end to end: convert the
# released weights into our checkpoint format, then probe them.
#
#   sbatch scripts/slurm/examples/birdmae_base_birdset.sh
#   sbatch scripts/slurm/examples/birdmae_base_birdset.sh data/datasets=train/birdset/per
#
# Three things must move together and are all set below — the checkpoint, the
# encoder geometry (`module/model=layerwise/birdmae`, which pulls
# `backbone/vit_external`: (frames, n_mels) input, no QK-norm) and the
# front-end (`data/transforms=birdmae`, that model's own preprocessing).
# Change one without the others and you probe a backbone that loaded partially,
# which `strict=False` reports rather than refuses.
#
# The conversion itself is the guard: it strict-loads into our ViTEncoder, so a
# geometry mismatch fails here instead of quietly producing a near-chance probe
# result later. It is cheap to repeat — the weights come out of the HF cache
# after the first run.
#
# For all 8 BirdSet tasks instead of one, the sweep already exists — run it the
# way sweep_birdset_layerwise.sh does, after this script has produced the
# checkpoint:
#
#   uv run python scripts/birdset_eval.py --multirun \
#     experiment=sweeps/layerwise/birdset/birdmae \
#     hydra/launcher="${HYDRA_LAUNCHER:-slurm}"
#
# `scripts/convert_external_ckpt.py audiomae` converts AudioMAE the same way
# (pair it with `module/model=layerwise/audiomae data/transforms=audiomae`).
#SBATCH --job-name=birdmae-base-birdset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -eo pipefail

cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Secrets + shared env. The Bird-MAE repo is public, so HF_TOKEN is optional,
# but HF_HOME/HF_HUB_CACHE decide where the download lands. See .env.example.
set -a
if [ -f .env ]; then source .env; fi
set +a

# Same directory `${paths.ckpt_dir}` resolves to, so the tracked sweep configs
# find this checkpoint by the name they already expect.
CKPT_DIR="${CKPT_DIR:-$PWD/checkpoints}"
CKPT="$CKPT_DIR/birdmae_base.ckpt"
mkdir -p "$CKPT_DIR"

uv sync

uv run python scripts/convert_external_ckpt.py birdmae_base --out "$CKPT"

srun uv run python scripts/birdset_eval.py \
  module/model=layerwise/birdmae \
  data/transforms=birdmae \
  data/datasets=train/birdset/hsn \
  "trainer.resume_from_checkpoint=$CKPT" \
  data.loaders.train.num_workers=20 \
  data.loaders.val_test.num_workers=16 \
  "$@"
