from dotenv import load_dotenv
load_dotenv()  # load repo .env (secrets, HF cache, CA bundle) before other imports
import logging
import multiprocessing as mp
import random
import time
from pathlib import Path
import traceback

import hydra
import torch
import torch.nn as nn
import wandb
from alp_data import dataset_from_config
import pandas as pd
import tenacity
import alp_data.io
import alp_data
import aioitertools
from lightning.fabric.fabric import Fabric
import huggingface_hub
import polars
from hydra.utils import instantiate
from lightning.fabric import Fabric
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from soundscape_ssl.data import CachedDataset, Compose, cleanup_all, compute_sample_weights, open_run_cache
from soundscape_ssl.models import ViTClassifier, ViTProtoFloat
from soundscape_ssl.models.utils.lr_decay import param_groups_lrd
from soundscape_ssl.training.lr_scheduler import CosineWarmupScheduler

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def hydra_main(cfg: DictConfig) -> None:
    try:
        fabric = Fabric(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            strategy=cfg.trainer.strategy,
            precision=cfg.trainer.precision,
        )
        fabric.launch(main, cfg)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        cleanup_all()


def main(fabric: Fabric, cfg: DictConfig) -> None:
    fabric.seed_everything(cfg.seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")   # or "medium"

    train_dataset, train_meta = dataset_from_config(instantiate(cfg.data.datasets["train"]))
    test_dataset, test_meta = dataset_from_config(instantiate(cfg.data.datasets["test"]))
    cfg.data.num_classes = test_meta["mulitlabel_from_feature"]["num_classes"] if "mulitlabel_from_feature" in test_meta else test_meta["label_from_feature"]["num_classes"]

    cache_cfg = cfg.data.get("cache", None)
    if cache_cfg is not None and cache_cfg.get("enabled", False):
        cache = open_run_cache(cache_cfg.get("dir"), cache_cfg.get("size_limit_gb", 50))
        train_dataset = CachedDataset(train_dataset, cache, "train")
        test_dataset = CachedDataset(test_dataset, cache, "test")

    # warmup loaders
    if fabric.is_global_zero:
        _ = train_dataset[0]
        _ = test_dataset[0]

    cw_cfg = cfg.data.get("class_weighting", None)
    if cw_cfg is not None and cw_cfg.get("enabled", False):
        weights = compute_sample_weights(
            train_dataset,
            label_column=cw_cfg.label_column,
            alpha=cw_cfg.alpha,
        )
        generator = torch.Generator().manual_seed(cw_cfg.seed)
        train_sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        **cfg.data.loaders.train,
        collate_fn=Compose(build_stages_list(cfg.data.transforms, "train")),  # type: ignore
        multiprocessing_context=mp.get_context("spawn"),
        drop_last=True,
        #worker_init_fn=worker_init_fn,
        sampler=train_sampler,
        shuffle=train_sampler is None,
    )

    test_loader = DataLoader(
        test_dataset,
        **cfg.data.loaders.val_test,
        collate_fn=Compose(build_stages_list(cfg.data.transforms, "test")),  # type: ignore
        multiprocessing_context=mp.get_context("spawn"),
        drop_last=False,
        # worker_init_fn=worker_init_fn,
        shuffle=False
    )

    _ = iter(train_loader)

    _ = iter(test_loader)

    model = instantiate(cfg.module.model)
    if cfg.module.get("freeze_backbone", True):
        model.freeze_backbone()

    effective_lr = cfg.module.optimizer.base_lr * cfg.data.loaders.train.batch_size / 256
    no_wd = ["cls_token", "pos_embed"]
    param_groups, _ = param_groups_lrd(
        model,
        weight_decay=cfg.module.optimizer.weight_decay,
        no_weight_decay_list=no_wd,
        layer_decay=cfg.module.optimizer.layer_decay,
        prototype_lr=cfg.module.optimizer.get("prototype_lr", None),
        layer_weights_lr=cfg.module.optimizer.get("layer_weights_lr", None),
    )
    optimizer = AdamW(param_groups, lr=effective_lr, betas=tuple(cfg.module.optimizer.betas))

    scheduler = CosineWarmupScheduler(
        optimizer,
        num_warmup_steps=cfg.module.scheduler.warmup_steps,
        num_training_steps=cfg.trainer.max_steps,
        min_lr=cfg.module.scheduler.min_lr
    )

    model, optimizer, scheduler = fabric.setup(model, optimizer, scheduler=scheduler)  # type: ignore
    train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=False)
    test_loader = fabric.setup_dataloaders(test_loader)

    global_step = _load_checkpoint(fabric, model, cfg)

    train_metrics = build_stages_list(list(cfg.module.metrics.values()), "train")
    test_metrics = build_stages_list(list(cfg.module.metrics.values()), "test")

    if fabric.is_global_zero:
        run = wandb.init(
            entity="mwirth",
            project="soundscape_ssl",
            name=_format_run_name(cfg.run_name),
            config=OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        )

    criterion = instantiate(cfg.module.loss)

    global_step = train(fabric,
                        cfg,
                        model,
                        optimizer,
                        scheduler,
                        criterion,
                        train_loader,
                        test_loader,
                        global_step,
                        train_metrics,
                        test_metrics,
                        run)

    if fabric.is_global_zero:
        # _save_checkpoint(fabric, model, optimizer, scheduler, global_step, cfg)
        log.info("Training complete.")
        run.finish()


def train(fabric: Fabric,
          cfg: DictConfig,
          model: ViTClassifier,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          criterion,
          train_loader: DataLoader,
          test_loader: DataLoader,
          global_step: int,
          train_metrics: list,
          test_metrics: list,
          run: wandb.Run) -> int:
    max_steps = cfg.trainer.max_steps
    log_every = cfg.trainer.get("log_every_n_steps", 50)
    validate_every = max_steps // cfg.trainer.validate_amount
    save_every = cfg.trainer.save_every_n_steps

    step_bar = tqdm(
        desc="Training",
        unit="step",
        total=max_steps,
        initial=global_step,
        disable=not fabric.is_global_zero,
        miniters=log_every,
    )

    model.train()
    # The train loader is finite; cycle through it until max_steps is reached.
    # Each fresh iteration re-draws from the WeightedRandomSampler.
    while global_step < max_steps:
        for batch in train_loader:
            with fabric.autocast():
                outputs = model(batch["spectrogram"])
                label = batch["label"]

            loss = criterion(outputs, label)
            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, max_norm=cfg.trainer.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            global_step += 1
            step_bar.update(1)

            if fabric.is_global_zero and global_step % log_every == 0:
                run.log({"train/loss_step": loss.item(),
                         **{f"train/{metric.__class__.__name__}": metric(outputs, label) for metric in train_metrics},
                         **{f"lr/{i}": group["lr"] for i, group in enumerate(optimizer.param_groups)},
                         },
                         step=global_step)

            if global_step % validate_every == 0:
                validate(fabric, model, test_loader, criterion, test_metrics, run, global_step)
                model.train()

            #if fabric.is_global_zero and global_step % save_every == 0:
                #_save_checkpoint(fabric, model, optimizer, scheduler, global_step, cfg)

            if global_step >= max_steps:
                break

    return global_step


@torch.no_grad()
def _log_layer_weights(model: nn.Module, run: wandb.Run, global_step: int) -> None:
    """Log the learned layerwise-probing weights for ViTProtoLayerwise.

    Logs the softmax-normalised per-layer weight (how much the probe relies on
    each transformer block) plus a single summary scalar — the softmax-weighted
    mean layer index — so the layer profile is captured per run. No-op for any
    other head, so it is safe to call unconditionally.
    """
    layer_weights = getattr(model, "layer_weights", None)
    if layer_weights is None:
        return

    w = torch.softmax(layer_weights.detach().float(), dim=0).cpu()
    layers = torch.arange(1, w.numel() + 1, dtype=w.dtype)
    centroid = float((w * layers).sum())  # 1-indexed; ~last layer if mass is late
    run.log({
        **{f"layer_weights/block_{i + 1:02d}": float(v) for i, v in enumerate(w)},
        "layer_weights/centroid": centroid,
    }, step=global_step)


@torch.no_grad()
def validate(fabric: Fabric,
             model: ViTClassifier,
             dataloder: DataLoader,
             criterion,
             metrics: list,
             run: wandb.Run,
             global_step: int = 0) -> None:
    model.eval()
    for batch in dataloder:
        with fabric.autocast():
            preds = model(batch["spectrogram"])
            targets: torch.Tensor = batch["label"]

        loss = criterion(preds, targets)
        for metric in metrics:
            metric.update(preds.cpu(), targets.int().cpu())

    if fabric.is_global_zero:
        run.log({"test/loss": loss.item(),
                 **{f"test/{metric.__class__.__name__}": metric.compute() for metric in metrics}},
                 step=global_step)
        _log_layer_weights(model, run, global_step)

    for metric in metrics:
        metric.reset()

# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _save_checkpoint(
    fabric: Fabric,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
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
    model: ViTClassifier,
    cfg: DictConfig,
) -> int:
    resume_path = cfg.trainer.get("resume_from_checkpoint")
    if not resume_path:
        return 0

    state_dict = fabric.load(resume_path)
    model_state_dict = {k[len("encoder."):]: v for k, v in state_dict["model"].items() if k.startswith("encoder.")}
    o = model.load_state_dict(model_state_dict, strict=False)
    print(o)
    log.info(o)
    return 0


def worker_init_fn(worker_id: int) -> None:
    # Each worker sleeps 0–1.5 s based on its worker ID + randomness
    time.sleep(random.uniform(0, 5))


def build_stages_list(cfg_list: ListConfig | list, stage: str) -> list:   # stage: "train", "val", or "test"
    classes = []
    for item in cfg_list:
        item = OmegaConf.to_container(item, resolve=True)
        allowed = item.pop("_stage_", ["train", "val", "test"])
        if stage in allowed:
            classes.append(instantiate(item))
    return classes

def _format_run_name(run_name: str) -> str:
    return "-".join([i.replace("/", ".").replace(".ckpt", "").split(".")[-1] for i in run_name.split("-")])

if __name__ == "__main__":
    hydra_main()
