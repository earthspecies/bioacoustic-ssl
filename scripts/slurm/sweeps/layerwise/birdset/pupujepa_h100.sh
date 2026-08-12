#!/usr/bin/env bash
#SBATCH --job-name=JM-5min
#SBATCH --partition=h100-80
#SBATCH --output=../logs/job_manager/%j.log
#SBATCH --time=05:00
#SBATCH --mem=12G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# Layerwise prototypical probing of PupuJEPA-Base across the 8 BirdSet downstream
# tasks — the GATE-0 readout of *where* the transferable features sit. Fans the
# per-dataset GPU jobs out via submitit. See docs/external_baselines.md.
#
#   sbatch scripts/slurm/sweeps/layerwise/birdset/pupujepa_h100.sh
#   sbatch scripts/slurm/sweeps/layerwise/birdset/pupujepa_h100.sh seed=2
#
# As in the final-layer sweep there is nothing to convert: the weights are a local
# safetensors file loaded at construction with strict=True, and the strict-load
# below surfaces a missing file or a renamed key once here rather than eight times
# in parallel on GPUs.
#
# Sizing — this is the memory-hungry arm, and the reason it gets its own script
# rather than a flag. 1000 tokens (patch 4 frames wide, no padding) x 12 block
# outputs, each passed through a *trainable* per-block LayerNorm and accumulated
# into a weighted sum, so unlike the final-layer arm autograd does retain a graph:
# the block outputs, the 12 LayerNorm outputs and the 12 accumulation
# intermediates all stay live for backward. At train batch 256 that is roughly
# 0.4 GB per (256, 1000, 768) bf16 tensor x ~36 live tensors, i.e. ~15-20 GB of
# activations. It should fit h100-80; it will very likely NOT fit an a100-40, so
# prefer `hydra/launcher=gpu_h100` here (the default below).
#
# If it OOMs, halve the batch size AND double max_steps together, so samples-seen
# stays comparable against the other arms (`max_steps` is per-dataset, so it has
# to be given per dataset):
#
#   sbatch scripts/slurm/sweeps/layerwise/birdset/pupujepa_h100.sh \
#       data.loaders.train.batch_size=128 trainer.max_steps=2500   # HSN: 2x 1250
#
# `prototype_lr` and `layer_weights_lr` are absolute group LRs, so the probe's
# learning rates are unaffected by the batch-size change.
#
# Verify the arm first: sbatch scripts/slurm/verify_pupujepa_cpu.sh

set -eo pipefail

# Load secrets + shared env from the repo .env (see .env.example).
set -a
source "$HOME/soundscape_mae/.env"
set +a

cd ~/soundscape_mae

mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv sync

# Validate the strict load before fanning out. num_classes is irrelevant here —
# this only exercises the checkpoint path and the backbone key match.
uv run python -c "
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path

with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None):
    cfg = compose(config_name='train', overrides=['module/model=layerwise/pupujepa'])
cfg.data.num_classes = 21
model = instantiate(cfg.module.model)
n = sum(p.numel() for p in model.parameters())
print(f'PupuJEPA loaded from {cfg.module.model.ckpt_path}: {n:,} params')
"

srun uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/layerwise/birdset/pupujepa \
  hydra/launcher=gpu_h100 \
  "$@"
