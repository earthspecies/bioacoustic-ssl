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
from soundscape_ssl.models.architectures.mae import MAE
from soundscape_ssl.models.utils.lr_decay import param_groups_lrd
from soundscape_ssl.training.lr_scheduler import CosineWarmupScheduler

log = logging.getLogger(__name__)


def pretrain(fabric: Fabric, cfg: DictConfig) -> None:
    fabric.seed_everything(cfg.seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")   # or "medium"

    datasets = [instantiate(config) for config in list(cfg.data.datasets.values())]

    dataset = MixedStreamingDataset(
        datasets=datasets,
        weights=None,  # TODO
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

    scheduler = CosineWarmupScheduler(
        optimizer,
        num_warmup_steps=cfg.module.scheduler.warmup_steps,
        num_training_steps=cfg.trainer.max_steps,
        min_lr=cfg.module.scheduler.min_lr
    )

    model, optimizer, scheduler = fabric.setup(model, optimizer, scheduler=scheduler)  # type: ignore
    loader = fabric.setup_dataloaders(loader)

    global_step = _load_checkpoint(fabric, model, optimizer, scheduler, cfg)

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

        with fabric.autocast():
            spec = batch["spectrogram"]
            outputs = model(spec)
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

    if fabric.is_global_zero:
        _save_checkpoint(fabric, model, optimizer, scheduler, global_step, cfg)
        log.info("Training complete.")
        run.finish()

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


def _load_checkpoint(
    fabric: Fabric,
    model: MAE,
    optimizer: AdamW,
    scheduler: CosineWarmupScheduler,
    cfg: DictConfig,
) -> int:
    resume_path = cfg.trainer.get("resume_from_checkpoint")
    if not resume_path:
        return 0
    state: dict[str, Any] = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "step": 0}
    fabric.load(resume_path, state, strict=False)
    log.info(f"Resumed from {resume_path} at step {state['step']}")
    return state["step"]


def build_transforms(cfg_list: ListConfig, stage: str) -> list:   # stage: "train", "val", or "test"
    transforms = []
    for item in cfg_list:
        item = OmegaConf.to_container(item, resolve=True)
        allowed = item.pop("_stage_", ["train", "val", "test"])
        if stage in allowed:
            transforms.append(instantiate(item))
    return transforms
