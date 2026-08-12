#!/usr/bin/env bash
#SBATCH --job-name=JM-5min
#SBATCH --partition=h100-80
#SBATCH --output=../logs/job_manager/%j.log
#SBATCH --time=05:00
#SBATCH --mem=12G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# Frozen prototypical probing of PupuJEPA-Base (I-JEPA on pooled XenoCanto +
# iNaturalist + AudioSet) across the 8 BirdSet downstream tasks. Fans the
# per-dataset GPU jobs out via submitit. See docs/external_baselines.md.
#
#   sbatch scripts/slurm/sweeps/proto/birdset/pupujepa_h100.sh
#   sbatch scripts/slurm/sweeps/proto/birdset/pupujepa_h100.sh seed=2
#
# Unlike the birdmae/audiomae sweeps there is no checkpoint-conversion step: the
# weights are a local safetensors file that `proto/pupujepa` loads at
# construction, per submodule with strict=True. The strict-load below therefore
# does what their conversion step does — surfaces a missing file or a renamed key
# once, here, instead of eight times in parallel on GPUs. `--mem=12G` covers that
# CPU-side load (1 GB checkpoint + a 113 M-param fp32 model); the fan-out itself
# needs nothing.
#
# Sizing: 1000 tokens against the other arms' 257, because PupuJEPA's patch is 4
# frames wide rather than 16. Unlike BAT's 513 none of it is padding — RoPE is
# generated per input grid, so 5 s maps to a (125, 8) grid exactly. The encoder is
# frozen and its inputs carry no grad, so autograd retains no graph through it and
# memory is transient forward activations only: roughly 0.4 GB per live
# (256, 1000, 768) bf16 tensor, a handful live at once. That fits h100-80
# comfortably and should fit an a100-40 (`hydra/launcher=gpu`).
#
# If it does OOM, halving `data.loaders.train.batch_size` also halves samples-seen
# — `max_steps` is fixed per dataset (`${data.datasets.max_steps}`) — which breaks
# comparability against the other arms. Double `trainer.max_steps` alongside it to
# hold samples-seen constant, per dataset:
#
#   sbatch scripts/slurm/sweeps/proto/birdset/pupujepa_h100.sh \
#       data.loaders.train.batch_size=128 trainer.max_steps=2500   # HSN: 2x 1250
#
# The probe's own learning rate is unaffected either way: `prototype_lr` is an
# absolute group LR, so the linear batch-size scaling in birdset_eval.py only
# moves the (frozen, unused) backbone groups.
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
    cfg = compose(config_name='train', overrides=['module/model=proto/pupujepa'])
cfg.data.num_classes = 21
model = instantiate(cfg.module.model)
n = sum(p.numel() for p in model.parameters())
print(f'PupuJEPA loaded from {cfg.module.model.ckpt_path}: {n:,} params')
"

srun uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/proto/birdset/pupujepa \
  hydra/launcher=gpu_h100 \
  "$@"
