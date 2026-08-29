# soundscape_mae — unlabeled PAM in bioacoustic MAE pretraining

Research code for a single question:

> **Can unlabeled passive-acoustic-monitoring (PAM) / soundscape audio improve a
> bioacoustic MAE backbone over Xeno-Canto (XC) alone — and if so, how must it be
> selected?**

The pipeline is: MAE-pretrain a ViT-B/16 on 5 s @ 32 kHz mel spectrograms from a
weighted mix of corpora, then probe the frozen encoder on **BirdSet** (8 tasks)
and **BEANS** (6 tasks) against published baselines (Bird-MAE-Base, AudioMAE, BAT).

**Metric of record:** BirdSet cmAP, test-set mean over 7 tasks (POW = validation,
always excluded).

## Where things stand

Short version as of 2026-08:

| Backbone | BirdSet layerwise cmAP (test mean, 7 tasks) |
|---|---|
| AudioMAE (AudioSet-2M) | 0.348 |
| Bird-MAE-Base (our probe) | 0.429 |
| XC 400k | 0.425 |
| XC + NASA 400k | 0.436 |
| **XC 1M** | **0.460** |
| XC + NASA 1M | 0.457 |

- Act 1 holds: our XC-only backbone reaches/exceeds Bird-MAE-Base parity.
- **Schedule length is the only effect clearing the probe noise floor** (+0.035
  for 400k→1M, 7/7 datasets; noise floor ≤0.001).
- **NASA soundscapes are null on BirdSet** — sign flips with schedule, magnitude
  at the noise floor — and mildly *diluting* along the step curve.
- The only positive PAM signal is a small single-seed cross-domain gain on BEANS
  (+0.006 mean, 3/5 up), which still lacks its matched layerwise XC-only control.

## Layout

```
src/soundscape_ssl/     library (installed package)
  data/
    datasets/           one module per corpus (see table below)
    transforms/         audio + batched-spectrogram transform pipeline
    iterable_dataset.py MixedStreamingDataset — weighted infinite mix of map-style datasets
  models/
    architectures/      mae, vit (encoder/decoder/classifier/proto heads), bat
    components/         attention, block, mlp, patch_embed, pos_embed
  training/
    mae_pretrainer.py   Fabric pretraining loop
    repr_eval.py        representation eval (kNN / ridge probe)
  loss/, metrics/, eval/

configs/                Hydra config tree (see "Configuration")
scripts/                entry points + one-off curation / probe / verification scripts
scripts/slurm/          sbatch wrappers, incl. sweeps/{proto,layerwise,linear,finetune}/
notebooks/              tutorial.ipynb (config -> data -> model walkthrough) + result
                        aggregation (birdset_results*.ipynb, beans_results.ipynb)
metadata/               NASA granule metadata, frozen label spaces, gbifID->species names
                        (tracked; source CSVs are not)
curated/                curated NASA event/region indices and materialized audio
tests/                  consistency (docstrings), unittests, integration
```

## Setup

Package manager is **uv**; everything runs through `uv run`.

```bash
uv sync                          # dev + gpu groups by default
cp .env.example .env             # then fill in real values
uv run python scripts/earthdata_login.py   # once per machine, for NASA Earthdata
```

Secrets and shared env (W&B, HF, Earthdata, CA bundle, HF cache) live in the
gitignored `.env`. Python entry points call `load_dotenv()`; slurm scripts
`source` it after the `#SBATCH` block. See `.env.example`.

For a CPU-only box: `uv sync --group cpu` (the `cpu`/`gpu` groups conflict).

## Pretraining

```bash
# single GPU
uv run python scripts/pretrain.py experiment=pretrain/pretrain_xc

# multi-GPU
uv run torchrun --nproc_per_node=4 scripts/pretrain.py \
  experiment=pretrain/pretrain_xc trainer.devices=4 trainer.strategy=ddp

# cluster
sbatch scripts/slurm/pretrain.sh
```

Available pretraining experiments (`configs/experiment/pretrain/`):

| Experiment | Mix |
|---|---|
| `pretrain_xc` | XC only |
| `pretrain_xc_pam_new` | XC + NASA BioScape/S2L fixed 5 s event slices, `w=[384,43,85]` |
| `pretrain_xc_nasa_regions` | same granules stored as 10–60 s regions, fresh random 5 s crop per access |
| `pretrain_xc_nasa_regions_balanced` | as above, rebalanced weights |
| `pretrain_xc_pam_audioset` | XC + AudioSet (never run) |

`data.weights` are **`MixedStreamingDataset` sampling weights, not dataset
sizes** — `[384,43,85]` is 75 % XC. Always verify the *logged* weights against the
config; a past run silently trained on a uniform mix.

Two `configs/trainer/pretrain.yaml` keys to check before every launch:
`resume_from_checkpoint` is **hardcoded to the last run's checkpoint** (set it to
`null` for a cold run), and `warm_restart` decides whether the cosine schedule is
restarted over the remaining steps or continued. Warm vs cold matters: ~60 % of
the apparent 400k→1M gain is already present at equal step count, i.e. partly a
warm-restart difference rather than extra steps. `eval_every_n_steps` runs the
in-loop representation eval (`training/repr_eval.py`, POW linear/GAP probe) to W&B.

## Evaluation

Two entry points, both Hydra multirun + submitit:

```bash
# BirdSet, layerwise probe, one job per dataset × checkpoint
uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/layerwise/birdset/vit hydra/launcher=gpu_h100

# BEANS
uv run python scripts/beans_eval.py --multirun \
  experiment=sweeps/proto/beans/vit hydra/launcher=gpu_h100
```

The sweep grid (datasets, checkpoints, seeds) is the `hydra.sweeper.params`
block *inside* the experiment config — edit it there, not on the command line.
`scripts/slurm/sweeps/<head>/<benchmark>/<backbone>_<partition>.sh` wraps each
combination as a small job-manager sbatch that fans the per-dataset GPU jobs out.

Four heads, all 1250 steps with the same transforms/LR recipe. Resolve which head
a run used from (`module.model._target_`, `module.freeze_backbone`) — the target
alone confuses finetuning with final-layer probing:

| Head | Config | Encoder | Readout |
|---|---|---|---|
| final-layer (FL) | `sweeps/proto/birdset/vit.yaml` | frozen | cosine-prototype on last block |
| layerwise (LW) | `sweeps/layerwise/birdset/vit.yaml` | frozen | same head on learned softmax sum of all 12 blocks |
| linear | `sweeps/linear/birdset/vit.yaml` | frozen | `nn.Linear` on CLS |
| finetune | `sweeps/finetune/birdset/vit.yaml` | trainable | FL head, encoder unfrozen |

Observed head ordering: **linear ≪ layerwise ≈ FL ≲ finetune**. Linear probing
cannot rank these backbones (0.13 vs 0.46 cmAP on identical frozen weights).

External baselines load **bit-exactly into our own `ViTEncoder`** — no wrapper, no
`timm`/`transformers` at runtime. Convert once, then probe with the normal configs:

```bash
uv run python scripts/convert_external_ckpt.py audiomae   # or birdmae
```

## Data sources

`src/soundscape_ssl/data/datasets/`, wired up via `configs/data/datasets/`:

| Module | Corpus | Notes |
|---|---|---|
| `xeno_canto.py` | Xeno-Canto | `XenoCantoRaw` (bytes, used for pretraining) / `XenoCantoLazy` |
| `nasa_earthaccess.py` | NASA BioSCape + Soundscapes-to-Landscapes | core PAM corpus, many splits (below) |
| `a2o_site.py` | Australian Acoustic Observatory | optional arm; license usage uncertain |
| `arbimon.py` | Arbimon | **dropped** — license forbids scraping at scale |
| `soundscape_pretrain.py` | HF `soundscape-pretrain` | a2o/arbimon have *different schemas*; load each via its own parquet glob |
| `beans.py`, `audioset.py`, `inaturalist.py` | downstream / aux | |
| `noaa.py`, `noaa_bucket.py`, `sanctsound.py`, `pifsc.py` | marine PAM | |

NASA splits (`NASAEarthAccess(split=...)`), all sharing one loader:

- `BIOSCAPE` / `S2L` — full granules, streamed from the ORNL DAAC over HTTP range reads.
- `*_EVENTS` — one row per ≥0.7 AudioProtoPNet detection, fetched over the network.
- `*_EVENTS_LOCAL`, `*_REGIONS` — materialized locally as flat `.bin` + parquet
  index and read via `np.memmap`.

Materialization / curation scripts:

| Script | Purpose |
|---|---|
| `curate_nasa.py` | model-confidence arm — AudioProtoPNet-20-BirdSet-XCL scores 5 s windows |
| `materialize_nasa_events.py` | download + decode the 5 s event slices to local shards |
| `materialize_nasa_regions.py` | store contiguous 10–60 s regions instead (fresh crop per access) |

`curate_nasa.py`'s curator needs `transformers` 4.x — run it with a
`uv run --with` override, not the project env.

## Configuration

Hydra, root configs `configs/pretrain.yaml` and `configs/train.yaml`:

```
data/
  pretrain.yaml | train.yaml       datasets + transforms + loaders
  datasets/pretrain/*              one file per pretraining corpus/split
  datasets/train/{birdset,beans}/  one file per downstream task
  transforms/                      pretrain, audiomae, birdmae, bat, ...
  loaders/                         default, pretrain, a100 (batch size / workers)
module/
  mae.yaml | vit.yaml              optimizer + scheduler + loss + metrics
  model/{mae,proto,layerwise,linear,backbone}/
experiment/
  pretrain/*                       the pretraining mixes
  sweeps/<head>/<benchmark>/<backbone>.yaml   eval sweeps incl. their sweeper grid
trainer/{pretrain,train}.yaml
hydra/launcher/{gpu,gpu_h100}.yaml submitit
```

Anything is overridable on the CLI, e.g.
`uv run python scripts/pretrain.py module.model.encoder_depth=6 trainer.max_steps=1000`.

## Operational notes / known pitfalls

Hard-won, all of these have cost a run at least once:

- **Dataloaders must use `spawn`** — `fork` is unsafe with the XC dataset. Spawn
  copies ~900k records per worker, so RAM scales with `num_workers` (OOM at
  20 workers / 34 GB).
- **NASA parquet OOM** — `pq.read_table(memory_map=True)` does *not* share pages;
  every worker copies the whole audio column. Fixed by the flat `.bin` + memmap
  layout the materialization scripts write.
- **Representation / kNN eval must run in fp32** — bf16 autocast collapses the
  anisotropic MAE kNN to chance.
- **Resolve backbones by full checkpoint path, not basename** —
  `step_0400000.ckpt` collides across three encoders.
- Resample with `soxr_hq`, not `kaiser_best` (330 ms/clip → the NOAA slowness).
- VBR-MP3 duration overestimates produce empty crops → `PeakNormalize` crash.
- GCS reads are forced anonymous via `data/datasets/_gcs_anon.py` (public buckets;
  fixes expiring-ADC auth on long runs).
- Probing label maps: a train/test label index shift from missing XC species
  manifests as random NES/PER/UHH scores.

## Development

```bash
uv run pytest tests/unittests
uv run pytest tests/consistency --base_folder soundscape_ssl
uv run pytest --doctest-modules soundscape_ssl
uv run ruff check . && uv run ruff format --check .
```

`pre-commit install` once; CI (`.github/workflows/ci.yml`) runs the same checks
plus `deptry`. `tests/` and `scripts/` are excluded from ruff. The
example template tests (`tests/integration/test_VanilaNN.py`,
`tests/unittests/test_linear.py`) are still the inherited placeholders — the
library itself has no unit tests yet.
