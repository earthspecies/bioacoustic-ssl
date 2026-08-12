import itertools
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import torch
import wandb
from hydra.utils import instantiate
from lightning.fabric import Fabric
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from soundscape_ssl.data import Compose, MixedStreamingDataset
from soundscape_ssl.data.transforms.spectrogram import BatchSpectrogram
from soundscape_ssl.models.architectures.mae import MAE
from soundscape_ssl.models.utils.lr_decay import param_groups_lrd
from soundscape_ssl.training.lr_scheduler import CosineWarmupScheduler
from soundscape_ssl.training.repr_eval import (
    build_pow_eval,
    run_layerwise_eval,
)

log = logging.getLogger(__name__)

# Probe-free representation eval (per-layer kNN cmAP/AUROC on POW) logged during pretraining.
_REPR_ROOT = Path(__file__).resolve().parents[3]
REPR_EVAL_DATASET = _REPR_ROOT / "configs/data/datasets/train/birdset/pow.yaml"
REPR_EVAL_TRANSFORMS = _REPR_ROOT / "configs/data/transforms/default.yaml"


def pretrain(fabric: Fabric, cfg: DictConfig) -> None:
    fabric.seed_everything(cfg.seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")   # or "medium"

    datasets = [instantiate(config) for config in list(cfg.data.datasets.values())]

    dataset = MixedStreamingDataset(
        datasets=datasets,
        weights=cfg.data.weights,
    )

    loader = DataLoader(
        dataset,
        **cfg.data.loaders.train,
        collate_fn=Compose(build_transforms(cfg.data.transforms, "train")),  # type: ignore
        multiprocessing_context=mp.get_context("spawn"),
        drop_last=True,
    )

    model = MAE(**cfg.module.model)

    effective_lr = cfg.module.optimizer.base_lr * cfg.data.loaders.train.batch_size / 256
    no_wd = ["encoder.cls_token", "encoder.pos_embed", "decoder.mask_token", "decoder.pos_embed"]
    param_groups, _ = param_groups_lrd(
        model,
        weight_decay=cfg.module.optimizer.weight_decay,
        no_weight_decay_list=no_wd,
        layer_decay=cfg.module.optimizer.layer_decay,
    )
    optimizer = AdamW(param_groups, betas=tuple(cfg.module.optimizer.betas))

    # Warm restart: run a fresh cosine cycle over the *remaining* steps rather than
    # continuing the (already fully-decayed) cosine baked into the checkpoint, whose
    # state would otherwise drive the LR back up to peak past the original horizon.
    warm_restart = cfg.trainer.get("warm_restart", False)
    resume_step = _peek_resume_step(fabric, cfg)
    cycle_steps = cfg.trainer.max_steps - resume_step if warm_restart else cfg.trainer.max_steps

    scheduler = CosineWarmupScheduler(
        optimizer,
        num_warmup_steps=cfg.module.scheduler.warmup_steps,
        num_training_steps=cycle_steps,
        min_lr=cfg.module.scheduler.min_lr
    )

    model, optimizer, scheduler = fabric.setup(model, optimizer, scheduler=scheduler)  # type: ignore
    loader = fabric.setup_dataloaders(loader)

    # On-device transforms (mel spectrogram + padding): moved off the CPU collate to
    # keep the GPU fed. The mel filterbank/window buffers live inside the torchaudio
    # module, so move them explicitly (BatchSpectrogram is a plain Transform, not an
    # nn.Module). BatchPadding is device-agnostic F.pad, nothing to move.
    device_transforms = build_transforms(cfg.data.transforms, "train", device=True)
    for t in device_transforms:
        if isinstance(t, BatchSpectrogram):
            t._mel_spectrogram.to(fabric.device)
            if t._to_db is not None:
                t._to_db.to(fabric.device)
    to_spec = Compose(device_transforms)

    global_step = _load_checkpoint(fabric, model, optimizer, scheduler, cfg, load_scheduler=not warm_restart)

    if fabric.is_global_zero:
        run = wandb.init(
            entity="mwirth",
            project="soundscape_ssl",
            name=f"Pretrain-XC",
            config=OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        )

    log.info(
        f"Starting MAE pretraining: max_steps={cfg.trainer.max_steps}, "
        f"effective_lr={effective_lr:.2e}"
    )

    # Representation eval (per-layer kNN cmAP/AUROC on POW) — built once, run on global zero.
    eval_every = cfg.trainer.get("eval_every_n_steps")
    eval_data = None
    if eval_every and fabric.is_global_zero:
        loader_kwargs = OmegaConf.to_container(cfg.data.loaders.val_test, resolve=True)
        loader_kwargs["num_workers"] = min(loader_kwargs.get("num_workers", 10), 10)
        eval_specs, eval_targets, eval_stats = build_pow_eval(
            REPR_EVAL_DATASET, REPR_EVAL_TRANSFORMS, loader_kwargs
        )
        log.info(f"POW representation eval set: {eval_stats}")
        if eval_stats["n_present"] < 1:
            log.warning("POW eval has no present classes — disabling representation eval.")
        else:
            eval_data = (eval_specs, eval_targets, eval_stats["num_classes"])
            _run_repr_eval(fabric, model, *eval_data, run, global_step)  # step-0 baseline

    remaining = cfg.trainer.max_steps - global_step

    step_bar = tqdm(
        itertools.islice(loader, remaining),
        desc="pretraining",
        unit="step",
        total=cfg.trainer.max_steps,
        initial=global_step,
        disable=not fabric.is_global_zero,
        miniters=cfg.trainer.get("log_every_n_steps", 50),
    )

    model.train()
    for batch in step_bar:

        batch = to_spec(batch)  # audio (GPU, fp32) -> spectrogram, padded
        with fabric.autocast():
            outputs = model(batch["spectrogram"])
            loss = outputs["loss"]

        # if not torch.isfinite(loss):
        #     log.info(f"Batch {global_step} was not finite.")
        #     optimizer.zero_grad()
        #     continue
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=cfg.trainer.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        loss_val = loss.item()
        global_step += 1

        if fabric.is_global_zero and global_step % cfg.trainer.get("log_every_n_steps", 50) == 0:
            run.log({"train/loss": loss_val, **{f"lr/{i}": group["lr"] for i, group in enumerate(optimizer.param_groups)}}, 
                    step=global_step)

        if fabric.is_global_zero and global_step % cfg.trainer.save_every_n_steps == 0:
            _save_checkpoint(fabric, model, optimizer, scheduler, global_step, cfg)

        if eval_data is not None and global_step % eval_every == 0:
            _run_repr_eval(fabric, model, *eval_data, run, global_step)

    if fabric.is_global_zero:
        _save_checkpoint(fabric, model, optimizer, scheduler, global_step, cfg)
        if eval_data is not None and global_step % eval_every != 0:
            _run_repr_eval(fabric, model, *eval_data, run, global_step)  # final anchor
        log.info("Training complete.")
        run.finish()

# ---------------------------------------------------------------------------
# Representation eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_repr_eval(fabric, model, eval_specs, eval_targets, num_classes, run, global_step) -> None:
    """Per-layer ridge-probe + kNN cmAP/AUROC on the cached multilabel POW set.

    Runs in fp32 (no autocast): MAE embeddings are highly anisotropic, so a bf16
    similarity matmul rounds all cosines together and collapses the kNN to chance.
    """
    model.eval()
    results = run_layerwise_eval(
        model.encoder, eval_specs, eval_targets, fabric.device, num_classes
    )
    run.log(results, step=global_step)
    log.info(
        f"[step {global_step}] POW repr eval: "
        + " ".join(
            f"{k.rsplit('/', 2)[-2]}/{k.rsplit('/', 1)[-1]}={v:.4f}"
            for k, v in results.items()
            if "/best_" in k and not k.endswith("_layer")
        )
    )
    model.train()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _save_checkpoint(
    fabric: Fabric,
    model: MAE,
    optimizer: AdamW,
    scheduler: CosineWarmupScheduler,
    step: int,
    cfg: DictConfig,
) -> None:
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"step_{step:07d}.ckpt"
    fabric.save(ckpt_path, {"model": model, "optimizer": optimizer, "scheduler": scheduler, "step": step})
    log.info(f"Checkpoint saved → {ckpt_path}")


def _peek_resume_step(fabric: Fabric, cfg: DictConfig) -> int:
    """Read just the global step from the resume checkpoint.

    Needed to size the warm-restart LR cycle before the scheduler is built (the
    scheduler must be constructed from the raw optimizer prior to ``fabric.setup``).
    """
    resume_path = cfg.trainer.get("resume_from_checkpoint")
    if not resume_path:
        return 0
    state: dict[str, Any] = {"step": 0}
    fabric.load(resume_path, state, strict=False)
    return state["step"]


def _load_checkpoint(
    fabric: Fabric,
    model: MAE,
    optimizer: AdamW,
    scheduler: CosineWarmupScheduler,
    cfg: DictConfig,
    load_scheduler: bool = True,
) -> int:
    resume_path = cfg.trainer.get("resume_from_checkpoint")
    if not resume_path:
        return 0
    state: dict[str, Any] = {"model": model, "optimizer": optimizer, "step": 0}
    if load_scheduler:
        state["scheduler"] = scheduler
    fabric.load(resume_path, state, strict=False)
    log.info(
        f"Resumed from {resume_path} at step {state['step']}"
        + ("" if load_scheduler else " (warm restart: fresh LR cycle)")
    )
    return state["step"]


def build_transforms(cfg_list: ListConfig, stage: str, device: bool = False) -> list:   # stage: "train", "val", or "test"
    transforms = []
    for item in cfg_list:
        item = OmegaConf.to_container(item, resolve=True)
        allowed = item.pop("_stage_", ["train", "val", "test"])
        on_device = item.pop("_device_", False)
        if stage in allowed and on_device == device:
            transforms.append(instantiate(item))
    return transforms
