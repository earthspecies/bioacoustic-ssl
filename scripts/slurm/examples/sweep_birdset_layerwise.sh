#!/usr/bin/env bash
# A BirdSet sweep: layerwise prototypical probing over all 8 tasks x the
# checkpoints named in the experiment config, one Slurm job per combination.
#
#   sbatch scripts/slurm/examples/sweep_birdset_layerwise.sh
#
# The grid lives in `hydra.sweeper.params` inside
# configs/experiment/sweeps/layerwise/birdset/vit.yaml — edit it there. Its
# checkpoints are read from $CKPT_DIR (default ./checkpoints); a clean clone has
# none, so either put them there or override on the command line:
#
#   sbatch scripts/slurm/examples/sweep_birdset_layerwise.sh \
#     'trainer.resume_from_checkpoint=${paths.ckpt_dir}/my_backbone.ckpt'
#
# THIS job is only the manager: it submits the GPU jobs through submitit and
# exits. Their resources come from $SLURM_LAUNCHER_* (see .env.example), not
# from the header below — hence 1 CPU, no GPU, minutes of wall clock. The GPU
# jobs outlive it.
#SBATCH --job-name=sweep-birdset-lw
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -eo pipefail

cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

set -a
if [ -f .env ]; then source .env; fi
set +a

# Resources for the fanned-out GPU jobs, read by
# configs/hydra/launcher/slurm.yaml. .env wins over these defaults, and
# HYDRA_LAUNCHER=<name> swaps in a site-local launcher entirely.
export SLURM_LAUNCHER_CPUS="${SLURM_LAUNCHER_CPUS:-8}"
export SLURM_LAUNCHER_MEM_GB="${SLURM_LAUNCHER_MEM_GB:-64}"

uv sync

srun uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/layerwise/birdset/vit \
  hydra/launcher="${HYDRA_LAUNCHER:-slurm}" \
  "$@"
