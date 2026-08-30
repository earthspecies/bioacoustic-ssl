#!/usr/bin/env bash
# The same sweep shape on BEANS: final-layer prototypical probing over the 6
# BEANS tasks. BEANS is multiclass, so the experiment config swaps in the
# multiclass transforms, cross-entropy and accuracy — a different entry point
# (`beans_eval.py`) reads it.
#
#   sbatch scripts/slurm/examples/sweep_beans_proto.sh
#
# Grid and checkpoints: `hydra.sweeper.params` in
# configs/experiment/sweeps/proto/beans/vit.yaml. See
# sweep_birdset_layerwise.sh for what this manager job does and does not size.
#SBATCH --job-name=sweep-beans-proto
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

export SLURM_LAUNCHER_CPUS="${SLURM_LAUNCHER_CPUS:-8}"
export SLURM_LAUNCHER_MEM_GB="${SLURM_LAUNCHER_MEM_GB:-64}"

uv sync

srun uv run python scripts/beans_eval.py --multirun \
  experiment=sweeps/proto/beans/vit \
  hydra/launcher="${HYDRA_LAUNCHER:-slurm}" \
  "$@"
