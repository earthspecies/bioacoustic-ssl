#!/usr/bin/env bash
#SBATCH --job-name=JM-5min
#SBATCH --partition=h100-80
#SBATCH --output=../logs/job_manager/%j.log
#SBATCH --time=05:00
#SBATCH --mem=12G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# Frozen LINEAR probing of PupuJEPA-Base (I-JEPA on pooled XenoCanto +
# iNaturalist + AudioSet) across the 8 BirdSet downstream tasks: mean-pooled
# patch tokens -> single nn.Linear. Fans the per-dataset GPU jobs out via
# submitit. The proto counterpart is proto/birdset/pupujepa_h100.sh.
#
#   sbatch scripts/slurm/sweeps/linear/birdset/pupujepa_h100.sh
#   sbatch scripts/slurm/sweeps/linear/birdset/pupujepa_h100.sh seed=2
#
# PupuJEPA has no CLS token, so the pooling is the mean over patch tokens — which
# is the clip embedding upstream's own `embed.py` publishes, i.e. the read-out the
# model actually deploys.
#
# As with the proto sweep there is no checkpoint-conversion step: the weights are
# a local safetensors file that `linear/pupujepa` loads at construction, per
# submodule with strict=True. The strict-load below does what the birdmae/audiomae
# sweeps' conversion step does — surfaces a missing file or a renamed key once,
# here, instead of eight times in parallel on GPUs. `--mem=12G` covers that
# CPU-side load (1 GB checkpoint + a 113 M-param fp32 model); the fan-out itself
# needs nothing.
#
# Sizing: 1000 tokens against the other arms' 257, because PupuJEPA's patch is 4
# frames wide rather than 16 (none of it padding — RoPE is generated per input
# grid, so 5 s maps to a (125, 8) grid exactly). The encoder is frozen and its
# inputs carry no grad, so memory is transient forward activations only, and the
# linear head is strictly cheaper than the proto head. If the proto sweep fits,
# this does.
#
# If it does OOM anyway, halving `data.loaders.train.batch_size` also halves
# samples-seen — `max_steps` is fixed per dataset (`${data.datasets.max_steps}`) —
# which breaks comparability against the other arms. Double `trainer.max_steps`
# alongside it to hold samples-seen constant, per dataset:
#
#   sbatch scripts/slurm/sweeps/linear/birdset/pupujepa_h100.sh \
#       data.loaders.train.batch_size=128 trainer.max_steps=2500   # HSN: 2x 1250
#
# Unlike the proto arm the head LR here DOES scale with the batch size:
# `prototype_lr` is nulled, so the head sits in the last layer-decay group at
# scale 1.0 and its LR is `base_lr * batch_size / 256`. Halving the batch halves
# the head LR — pass `module.optimizer.base_lr=6e-4` to hold it fixed.
#
# Verify the arm first — it reproduces upstream bit-for-bit or it does not:
#   sbatch scripts/slurm/verify_pupujepa_cpu.sh

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
    cfg = compose(config_name='train', overrides=['module/model=linear/pupujepa'])
cfg.data.num_classes = 21
model = instantiate(cfg.module.model)
n = sum(p.numel() for p in model.parameters())
print(f'PupuJEPA loaded from {cfg.module.model.ckpt_path}: {n:,} params')
"

srun uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/linear/birdset/pupujepa \
  hydra/launcher=gpu_h100 \
  "$@"
