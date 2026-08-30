# Example sbatch wrappers

Five scripts, one per shape of run. They are deliberately generic: no partition,
no account, no site paths — submit them from the repo root and Slurm's own
defaults apply.

```bash
sbatch scripts/slurm/examples/pretrain.sh
sbatch -p <partition> scripts/slurm/examples/birdset_hsn.sh
sbatch scripts/slurm/examples/sweep_birdset_layerwise.sh
sbatch scripts/slurm/examples/sweep_beans_proto.sh
sbatch scripts/slurm/examples/birdmae_base_birdset.sh
```

| Script | Shape |
|---|---|
| `pretrain.sh` | one long single-GPU MAE pretraining run |
| `birdset_hsn.sh` | one downstream probe (the default arm: HSN, block-diagonal layerwise head) |
| `sweep_birdset_layerwise.sh` | a `--multirun` sweep that fans one GPU job out per (dataset x checkpoint x seed) |
| `sweep_beans_proto.sh` | the same on BEANS, with BEANS' multiclass transforms/loss |
| `birdmae_base_birdset.sh` | a published baseline: convert the released Bird-MAE-Base weights, then probe them |

Anything you need on top — a partition, an account, a per-job venv, module
loads — belongs in `.env`, in `sbatch` flags, or in your own copy of these
scripts. `scripts/slurm/` outside this directory is gitignored for exactly that
reason: keep your cluster's wrappers there and they stay out of the repo.

Resources for the *fanned-out* jobs of a sweep come from `$SLURM_LAUNCHER_*`
(see `.env.example` and `configs/hydra/launcher/slurm.yaml`), not from the
`#SBATCH` header of the wrapper — the header only sizes the small job-manager
job that does the submitting.
