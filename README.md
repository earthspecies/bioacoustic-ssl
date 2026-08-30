# soundscape_mae

Research code for one question:

> Can unlabeled passive-acoustic-monitoring (PAM) audio improve a bioacoustic MAE
> backbone over Xeno-Canto (XC) alone, and if so, how must that audio be selected?

The pipeline is short. MAE-pretrain a ViT-B/16 on 5 s, 32 kHz mel spectrograms
drawn from a weighted mix of corpora, then probe the frozen encoder on **BirdSet**
(8 tasks) and **BEANS** (6 tasks) against published baselines (Bird-MAE-Base,
AudioMAE, BAT).

Metric of record is BirdSet cmAP, test-set mean over 7 tasks. POW is BirdSet's own
validation split and is always excluded.

## Where things stand

As of 2026-08:

| Backbone | BirdSet layerwise cmAP (test mean, 7 tasks) |
|---|---|
| AudioMAE (AudioSet-2M) | 0.348 |
| Bird-MAE-Base (our probe) | 0.429 |
| XC 400k | 0.425 |
| XC + NASA 400k | 0.436 |
| **XC 1M** | **0.460** |
| XC + NASA 1M | 0.457 |

Our XC-only backbone reaches Bird-MAE-Base parity, which was the first thing worth
checking. Beyond that, schedule length is the only effect clearing the probe noise
floor, at +0.035 for 400k to 1M on 7 of 7 datasets against a noise floor of ~0.001.
NASA soundscapes come out null on BirdSet. The sign flips with schedule length and
the magnitude sits at the noise floor. The one positive PAM signal is a small
single-seed cross-domain gain on BEANS (+0.006 mean, 3 of 5 up), which still lacks
its matched layerwise XC-only control.

## What is in the repo

```
src/soundscape_ssl/      the installed library
  data/datasets/         one module per corpus (table below)
  data/transforms/       audio and batched-spectrogram transforms
  data/iterable_dataset.py   MixedStreamingDataset, a weighted infinite mix of
                             map-style datasets
  models/architectures/  mae, vit (encoder, decoder, classifier, prototypical
                         heads), bat
  models/components/     attention, block, mlp, patch_embed, pos_embed
  training/              mae_pretrainer.py (Fabric loop), repr_eval.py,
                         lr_scheduler.py
  loss/, metrics/

hf_model/                the published HuggingFace model code. Release payload,
                         copied into the model repo, not part of the library
configs/                 Hydra config tree (see "Configuration")
scripts/                 three entry points plus curation, conversion and export
scripts/slurm/examples/  five generic sbatch wrappers
notebooks/               tutorial.ipynb (config to data to model walkthrough) and
                         xc_0.1_vs_0.2.ipynb (XC version comparison)
metadata/                NASA granule tables and the frozen XC label spaces
tests/                   docstring consistency plus unit tests
```

Everything generated stays out of git: `checkpoints/`, `curated/` (materialized
NASA shards), `artifacts/` (exported HuggingFace models), `outputs/` (Hydra run
directories), and all of `scripts/slurm/` outside `examples/`. Result-aggregation
notebooks and site-local sbatch wrappers are local too, so a fresh clone will not
have them.

Three entry points, all Hydra apps:

| Script | Does |
|---|---|
| `scripts/pretrain.py` | MAE pretraining |
| `scripts/birdset_eval.py` | BirdSet probing and finetuning, and the full-XC head |
| `scripts/beans_eval.py` | BEANS probing |

The rest of `scripts/` is supporting work: `earthdata_login.py`,
`build_xc_label_space.py`, `convert_external_ckpt.py`, `curate_nasa.py`,
`materialize_nasa_events.py`, `materialize_nasa_regions.py`, `export_hf_model.py`.

## Setup

The package manager is **uv**, and everything runs through `uv run`.

```bash
uv sync                                    # dev + gpu groups by default
cp .env.example .env                       # then fill in real values
uv run python scripts/earthdata_login.py   # once per machine, for NASA Earthdata
```

Secrets and shared environment (W&B, HF, Earthdata, CA bundle, HF cache) live in
the gitignored `.env`. Python entry points call `load_dotenv()`, and the slurm
scripts `source` it after the `#SBATCH` block. See `.env.example`.

On a CPU-only box use `uv sync --group cpu`. The `cpu` and `gpu` groups conflict.

## Pretraining

```bash
# single GPU
uv run python scripts/pretrain.py experiment=pretrain/pretrain_xc

# multi-GPU
uv run torchrun --nproc_per_node=4 scripts/pretrain.py \
  experiment=pretrain/pretrain_xc trainer.devices=4 trainer.strategy=ddp

# cluster
sbatch scripts/slurm/examples/pretrain.sh
```

The mixes in `configs/experiment/pretrain/`:

| Experiment | Mix | Weights |
|---|---|---|
| `pretrain_xc` | XC only | `null` |
| `pretrain_xc_nasa_regions` | XC + NASA BioScape and S2L regions | `[384, 43, 85]`, 75 % XC |
| `pretrain_xc_nasa_regions_balanced` | same three sources, rebalanced | `[256, 85, 171]`, 50 % XC |
| `pretrain_xc_pam_new` | currently identical to the balanced arm | `[256, 85, 171]` |

`pretrain_xc_pam_new` is worth a warning. Its name says fixed 5 s event slices, but
the file now composes the same three region datasets as the balanced arm, at the
same weights and under the same `run_name`. The event-slice mix it used to run is
commented out inside the file. Two configs that launch identical runs is a trap, so
either delete it or restore the mix it names.

`data.weights` are `MixedStreamingDataset` sampling weights, not dataset sizes.
Always check the *logged* weights against the config. One past run silently trained
on a uniform mix.

Two keys in `configs/trainer/pretrain.yaml` deserve a look before every launch.
`resume_from_checkpoint` defaults to `null`, a cold start, and points at a
checkpoint to continue a run. `warm_restart` decides whether the cosine schedule
restarts over the remaining steps or continues. The difference matters: about 60 %
of the apparent 400k to 1M gain is already there at equal step count, so part of it
is a warm-restart effect rather than extra steps. `eval_every_n_steps` runs the
in-loop representation eval (`training/repr_eval.py`, a POW linear and GAP probe)
and logs it to W&B.

## Downstream evaluation

Both eval entry points are Hydra multirun plus submitit.

```bash
# BirdSet, layerwise probe, one job per dataset x checkpoint
uv run python scripts/birdset_eval.py --multirun \
  experiment=sweeps/layerwise/birdset/vit hydra/launcher=slurm

# BEANS
uv run python scripts/beans_eval.py --multirun \
  experiment=sweeps/proto/beans/vit hydra/launcher=slurm
```

`hydra/launcher=slurm` is the tracked, generic submitit launcher. Partition, cpus,
memory, wall clock and array parallelism all come from `$SLURM_LAUNCHER_*` (see
`.env.example`), so it needs no edit on a new cluster. A site that needs more, such
as a per-job venv, module loads or a scratch mount, keeps its own launcher file in
`configs/hydra/launcher/` (gitignored) and selects it with `HYDRA_LAUNCHER=<name>`,
which the example wrappers honour.

The sweep grid (datasets, checkpoints, seeds) lives in the `hydra.sweeper.params`
block *inside* the experiment config. Edit it there, not on the command line.
`scripts/slurm/examples/sweep_birdset_layerwise.sh` and `sweep_beans_proto.sh` wrap
a sweep as a small job-manager sbatch that fans the per-dataset GPU jobs out and
exits. Copy one per arm, and keep cluster-specific wrappers in `scripts/slurm/`,
which is gitignored. See `scripts/slurm/examples/README.md`.

Four heads, all 1250 steps on the same transforms and LR recipe. Work out which
head a run used from the pair (`module.model._target_`, `module.freeze_backbone`).
The target alone confuses finetuning with final-layer probing.

| Head | Config | Encoder | Readout |
|---|---|---|---|
| final-layer | `sweeps/proto/birdset/vit.yaml` | frozen | cosine-prototype on the last block |
| layerwise | `sweeps/layerwise/birdset/vit.yaml` | frozen | same head on a learned softmax sum of all 12 blocks |
| linear | `sweeps/linear/birdset/vit.yaml` | frozen | `nn.Linear` on CLS |
| finetune | `sweeps/finetune/birdset/vit.yaml` | trainable | final-layer head, encoder unfrozen |

Ranked by score, linear is far behind, layerwise and final-layer land close
together, and finetune is a little ahead of both. Linear probing cannot rank these
backbones at all, giving 0.13 against 0.46 cmAP on identical frozen weights.

External baselines load bit-exactly into our own `ViTEncoder`, with no wrapper and
no `timm` or `transformers` at runtime. Convert once, then probe with the normal
configs:

```bash
uv run python scripts/convert_external_ckpt.py birdmae_base   # or audiomae
```

`scripts/slurm/examples/birdmae_base_birdset.sh` runs the conversion and the probe
as one job, and shows the triple that has to move together: the checkpoint,
`module/model=*/birdmae` for the external geometry, and `data/transforms=birdmae`.

## The HuggingFace release

The released model is **XenoMAE**, in two artifacts:

- `base`, the pretrained ViT-B/16 MAE encoder at 86 M parameters. The MAE decoder
  is dropped, so `AutoModel` returns the encoder.
- `xc-classifier`, that same encoder frozen under a layerwise prototypical head
  trained over all 10 799 Xeno-Canto taxa. Not a bird-only head: 9 733 Aves plus
  508 amphibians, 423 insects and 135 mammals. It carries its own encoder copy and
  ships `xc_classes.parquet`, the map from output index to gbifID that logit
  masking needs.

`hf_model/` holds three modules that get copied verbatim into the model repo and
import nothing from `soundscape_ssl`: `configuration_xenomae`, `modeling_xenomae`
and `feature_extraction_xenomae`. `conversion.py` stays here, because nobody
loading the weights needs it.

Training the head, then exporting both artifacts:

```bash
# train the full-XC head on a frozen pretraining checkpoint
uv run python scripts/birdset_eval.py --multirun \
  experiment=xc_classifier/vit_layerwise hydra/launcher=slurm

# export
uv run python scripts/export_hf_model.py base \
  --ckpt $CKPT_DIR/XC_1M.ckpt --out artifacts/xenomae/base
uv run python scripts/export_hf_model.py xc-classifier \
  --ckpt $CKPT_DIR/xc_head_XC_1M_step_0100000.ckpt --out artifacts/xenomae/xc-classifier
```

The export is gated. Before anything is written, the published model is checked
against the in-repo model that produced the checkpoint on a fixed input, and the
check is repeated against the artifact reloaded from disk. Bit-exact or nothing.
`tests/unittests/test_hf_xenomae.py` asserts the same property on a tiny geometry
with random weights, so the conversion stays a key rename rather than a
re-implementation that happens to agree.

The head's label space is frozen in `metadata/xc_v0.1.0_all_classes.parquet`, built
by `scripts/build_xc_label_space.py` from exactly the (version, split) pair the
train stream reads. Rebuild it whenever either changes. The two have silently
diverged once already, which put 8 gbifIDs outside the label space and left 946
output units with no training audio.

Note that several docstrings still tell you to run the release code under
`uv run --with "transformers==4.57.1"`. That was needed when the project pinned
torch 2.6. Since the bump to torch 2.11 the project environment imports `hf_model`
and passes its tests directly. `scripts/curate_nasa.py` still needs its own
override, because AudioProtoPNet's remote code wants transformers 4.x.

## Data sources

Modules in `src/soundscape_ssl/data/datasets/`, wired up through
`configs/data/datasets/`:

| Module | Corpus | Notes |
|---|---|---|
| `xeno_canto.py` | Xeno-Canto | `XenoCantoRaw` (bytes, what pretraining uses) and `XenoCantoLazy` |
| `nasa_earthaccess.py` | NASA BioSCape + Soundscapes-to-Landscapes | the core PAM corpus, several splits (below) |
| `a2o_site.py` | Australian Acoustic Observatory | optional arm, license usage uncertain |
| `beans.py`, `audioset.py`, `inaturalist.py` | downstream and auxiliary | |
| `noaa.py`, `noaa_bucket.py`, `sanctsound.py`, `pifsc.py` | marine PAM | |

NASA splits, selected with `NASAEarthAccess(split=...)` and all sharing one loader:

- `BIOSCAPE` and `S2L`, full granules streamed from the ORNL DAAC over HTTP range
  reads.
- `*_EVENTS`, one row per AudioProtoPNet detection at 0.7 or above, fetched over
  the network.
- `*_EVENTS_LOCAL` and `*_REGIONS`, materialized locally as a flat `.bin` plus a
  parquet index and read through `np.memmap`.

The scripts that produce those local stores:

| Script | Purpose |
|---|---|
| `curate_nasa.py` | scores 5 s windows with AudioProtoPNet-20-BirdSet-XCL and writes the detections |
| `materialize_nasa_events.py` | downloads and decodes the fixed 5 s event slices into local shards |
| `materialize_nasa_regions.py` | stores contiguous 10 to 60 s regions instead, so each access gets a fresh crop |

Regions exist because of a repetition asymmetry. A 400k-step run at 25 % PAM weight
re-draws each identical fixed 5 s event view about 330 times, against about 19 for
XC, whose crop offset is redrawn every epoch. The region store cuts that to about
76.

## Configuration

Hydra, with root configs `configs/pretrain.yaml` and `configs/train.yaml`. Both
compose and run on a clean clone with no checkpoints, no cluster and no W&B team.
`configs/train.yaml` defaults to BirdSet HSN with the block-diagonal layerwise
prototypical head on a randomly initialised encoder.

```bash
uv run python scripts/birdset_eval.py                       # the default arm
uv run python scripts/birdset_eval.py --cfg job --resolve   # print it, run nothing
uv run python scripts/beans_eval.py experiment=beans        # the BEANS equivalent
```

Every published arm is one `experiment=...` away from that. Machine-specific values
are environment variables rather than config edits: `$CKPT_DIR` (where
`trainer.resume_from_checkpoint` and the sweeps read pretraining checkpoints from,
defaulting to `./checkpoints`), `$PROJECT_ROOT`, and `$WANDB_ENTITY`,
`$WANDB_PROJECT`, `$WANDB_MODE`. See `.env.example`.

```
data/
  pretrain.yaml | train.yaml       datasets + transforms + loaders
  datasets/pretrain/*              one file per pretraining corpus or split
  datasets/train/{birdset,beans}/  one file per downstream task
  datasets/train/xc_all.yaml       the full-XC stream the released head trains on
  transforms/                      pretrain, audiomae, birdmae, bat, multiclass
  loaders/                         default, pretrain, a100, h100 (batch size, workers)
module/
  mae.yaml | vit.yaml              optimizer + scheduler + loss + metrics
  model/{mae,proto,layerwise,linear,backbone}/
experiment/
  pretrain/*                       the pretraining mixes
  sweeps/<head>/<benchmark>/<backbone>.yaml   eval sweeps, sweeper grid included
  xc_classifier/vit_layerwise.yaml the full-XC head for the release
trainer/{pretrain,train}.yaml
hydra/launcher/slurm.yaml          submitit, resources from $SLURM_LAUNCHER_*
```

Anything is overridable on the command line, for example
`uv run python scripts/pretrain.py module.model.encoder_depth=6 trainer.max_steps=1000`.
