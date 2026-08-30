from dotenv import load_dotenv
load_dotenv()  # load repo .env (secrets, HF cache, CA bundle) before other imports
import logging
import multiprocessing as mp
import random
import time
from pathlib import Path
import traceback
from typing import Any

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

from bioacoustic_ssl.data import Compose, apply_logit_mask, compute_sample_weights, logit_mask
from bioacoustic_ssl.models import ViTClassifier, ViTProtoFloat
from bioacoustic_ssl.models.utils.lr_decay import param_groups_lrd
from bioacoustic_ssl.training.lr_scheduler import CosineWarmupScheduler

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


def main(fabric: Fabric, cfg: DictConfig) -> None:
    fabric.seed_everything(cfg.seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")   # or "medium"

    train_dataset, train_meta = dataset_from_config(instantiate(cfg.data.datasets["train"]))
    test_dataset, test_meta = dataset_from_config(instantiate(cfg.data.datasets["test"]))
    cfg.data.num_classes = test_meta["mulitlabel_from_feature"]["num_classes"] if "mulitlabel_from_feature" in test_meta else test_meta["label_from_feature"]["num_classes"]

    # Class identity per label index, for per-class metric keys. The label map is
    # {class_id: index} (gbifID for BirdSet), so invert it and order by index.
    _label_meta = test_meta.get("mulitlabel_from_feature") or test_meta.get("label_from_feature") or {}
    _label_map = _label_meta.get("label_map") or {}
    class_ids = [cid for cid, _ in sorted(_label_map.items(), key=lambda kv: kv[1])]

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

    # Skip the test-loader warmup when validation is off: it spawns
    # `val_test.num_workers` processes that would never be used.
    if cfg.trainer.validate_amount:
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

    global_step = _load_checkpoint(fabric, model, cfg, optimizer, scheduler)

    train_metrics = build_stages_list(list(cfg.module.metrics.values()), "train")
    test_metrics = build_stages_list(list(cfg.module.metrics.values()), "test")

    if fabric.is_global_zero:
        run = wandb.init(
            entity=cfg.logger.entity,
            project=cfg.logger.project,
            mode=cfg.logger.mode,
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
                        run,
                        class_ids)

    if fabric.is_global_zero:
        if cfg.trainer.get("save_checkpoints", False):
            _save_checkpoint(fabric, model, optimizer, scheduler, global_step, cfg)
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
          run: wandb.Run,
          class_ids: list | None = None) -> int:
    max_steps = cfg.trainer.max_steps
    log_every = cfg.trainer.get("log_every_n_steps", 50)
    # `validate_amount: 0` disables validation entirely (the full-XC classifier
    # run, which is scored later by masked BirdSet evaluation, not on-line).
    validate_amount = cfg.trainer.validate_amount
    validate_every = max_steps // validate_amount if validate_amount else 0
    save_every = cfg.trainer.save_every_n_steps
    # Masked BirdSet validation (POW by default). None unless `val_masked` is
    # enabled, which only the full-label-space runs do.
    validate_masked = build_masked_validator(fabric, cfg)
    validate_masked_every = int(cfg.val_masked.every_n_steps) if validate_masked else 0

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

            if validate_every and global_step % validate_every == 0:
                validate(fabric, model, test_loader, criterion, test_metrics, run, global_step, class_ids)
                model.train()

            if validate_masked_every and global_step % validate_masked_every == 0:
                validate_masked(model, run, global_step)
                _log_layer_weights(model, run, global_step)

            # Off by default: a full state checkpoint is ~2.5 GB for the
            # 11 737-class XC head, and the 1250-step benchmark probes have no
            # use for one. Long runs set `trainer.save_checkpoints=true` so they
            # survive the 48 h submitit timeout and leave a final artifact.
            if cfg.trainer.get("save_checkpoints", False) and global_step % save_every == 0:
                _save_checkpoint(fabric, model, optimizer, scheduler, global_step, cfg)

            if global_step >= max_steps:
                break

    return global_step


def build_masked_validator(fabric: Fabric, cfg: DictConfig):
    """Periodic masked-BirdSet validation for a full-label-space head.

    Why this exists: the full-XC run sets ``validate_amount: 0`` — the only
    in-distribution split is XC ``validation``, which is now inside the train
    split — so it has NO stopping signal at all, and the 100 k-step run spent
    98 k steps after its loss floor (~1 800) without anything noticing. This
    scores the head the way it is actually judged: restrict the logits to one
    BirdSet task's species and compute that task's own metrics.

    The task is POW by default. POW is BirdSet's validation split and is already
    excluded from every published test-set mean, so selecting a checkpoint on it
    does not touch a test set.

    Returns ``None`` when ``val_masked`` is absent or disabled, so the ordinary
    per-task probes are unaffected.
    """
    vcfg = cfg.get("val_masked")
    if not vcfg or not vcfg.get("enabled", False):
        return None

    dataset, meta = dataset_from_config(instantiate(vcfg.dataset))
    label_meta = meta.get("mulitlabel_from_feature") or meta.get("label_from_feature") or {}
    class_ids = [cid for cid, _ in sorted(label_meta["label_map"].items(), key=lambda kv: kv[1])]

    mask, head_classes = logit_mask(vcfg.xc_classes, class_ids)
    if head_classes != cfg.data.num_classes:
        raise ValueError(
            f"val_masked.xc_classes has {head_classes} classes but the head has "
            f"{cfg.data.num_classes}. The mask would re-index against a label space "
            f"the head does not have and report plausible wrong numbers."
        )
    uncovered = mask < 0
    fill = float(vcfg.uncovered_fill)

    loader = DataLoader(
        dataset,
        **cfg.data.loaders.val_test,
        collate_fn=Compose(build_stages_list(cfg.data.transforms, "test")),  # type: ignore
        multiprocessing_context=mp.get_context("spawn"),
        drop_last=False,
        shuffle=False,
    )
    loader = fabric.setup_dataloaders(loader)

    # Metrics over the TASK's label space, not the head's. `cfg.module.metrics`
    # is bound to `data.num_classes`, so num_labels is overridden per metric.
    metrics = []
    for item in cfg.module.metrics.values():
        item = OmegaConf.to_container(item, resolve=True)
        item.pop("_stage_", None)
        if "num_labels" in item:
            item["num_labels"] = len(class_ids)
        metrics.append(instantiate(item))

    tag = str(vcfg.dataset.test.split).split("-")[0].lower()
    log.info(
        f"Masked validation on {vcfg.dataset.test.split}: {len(class_ids)} classes "
        f"out of {head_classes} head outputs ({int(uncovered.sum())} uncovered), "
        f"every {vcfg.every_n_steps} steps, logged under val_{tag}/."
    )

    @torch.no_grad()
    def validate_masked(model: nn.Module, run: wandb.Run, global_step: int) -> None:
        model.eval()
        for metric in metrics:
            metric.reset()

        for batch in loader:
            with fabric.autocast():
                logits = model(batch["spectrogram"])
            preds = apply_logit_mask(logits.float(), mask.to(logits.device), fill)
            for metric in metrics:
                metric.update(preds.cpu(), batch["label"].int().cpu())

        if fabric.is_global_zero:
            logs = {}
            for metric in metrics:
                value = metric.compute()
                if value.ndim == 0:
                    logs[f"val_{tag}/{metric.__class__.__name__}"] = float(value)
                else:
                    # Per-class metric: only the macro over the classes the head
                    # can actually predict, so the uncovered species' chance
                    # floor is not read as a model result.
                    logs[f"val_{tag}/{metric.__class__.__name__}_covered"] = float(value[~uncovered].mean())
            run.log(logs, step=global_step)
            log.info(f"step {global_step}: " + "  ".join(f"{k}={v:.4f}" for k, v in logs.items()))
        model.train()

    return validate_masked


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
    # Which encoder block each weight belongs to. Identity unless the head sets
    # `fusion_layers`, when the weights cover only the last k blocks and both
    # the keys and the centroid must be offset to stay comparable across runs.
    blocks = getattr(model, "fusion_blocks", None) or list(range(1, w.numel() + 1))
    layers = torch.tensor(blocks, dtype=w.dtype)
    centroid = float((w * layers).sum())  # 1-indexed; ~last layer if mass is late
    run.log({
        **{f"layer_weights/block_{b:02d}": float(v) for b, v in zip(blocks, w)},
        "layer_weights/centroid": centroid,
    }, step=global_step)


def _metric_logs(metric, class_ids: list | None = None) -> dict:
    """W&B keys for one metric, expanding per-class metrics into one series each.

    A metric built with ``average=None`` computes to a ``(num_labels,)`` tensor
    rather than a scalar. Logging that under a single key would collide with the
    macro metric of the same class and be unplottable, so each element gets its
    own key under a ``test_<Metric>_per_class/`` section — one W&B panel per
    class. Keys carry the class id from the label map (the gbifID for BirdSet)
    when it is available, falling back to the label index.
    """
    value = metric.compute()
    name = metric.__class__.__name__

    if value.ndim == 0:
        return {f"test/{name}": value}

    short = "AP" if "AveragePrecision" in name else name
    labels = class_ids if class_ids and len(class_ids) == len(value) else range(len(value))
    return {
        f"test_{short}_per_class/{label}": v.item()
        for label, v in zip(labels, value)
    }


@torch.no_grad()
def validate(fabric: Fabric,
             model: ViTClassifier,
             dataloder: DataLoader,
             criterion,
             metrics: list,
             run: wandb.Run,
             global_step: int = 0,
             class_ids: list | None = None) -> None:
    model.eval()
    for batch in dataloder:
        with fabric.autocast():
            preds = model(batch["spectrogram"])
            targets: torch.Tensor = batch["label"]

        loss = criterion(preds, targets)
        for metric in metrics:
            metric.update(preds.cpu(), targets.int().cpu())

    if fabric.is_global_zero:
        logs = {"test/loss": loss.item()}
        for metric in metrics:
            logs.update(_metric_logs(metric, class_ids))
        run.log(logs, step=global_step)
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
    # Namespaced by run name, not by the hydra job dir alone. `hydra.sweep.dir`
    # is pinned per sweep and job indices restart at 0 on every submission, so a
    # single-arm relaunch (a resume, or the second backbone submitted on its own)
    # lands in job dir 0 and writes `step_XXXXXXX.ckpt` straight over the head of
    # whichever arm was job 0 last time — same filename, different backbone, no
    # warning. The formatted run name carries the arm, so it cannot.
    ckpt_dir = Path(cfg.paths.checkpoint_dir) / _format_run_name(cfg.run_name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"step_{step:07d}.ckpt"
    fabric.save(ckpt_path, {"model": model, "optimizer": optimizer, "scheduler": scheduler, "step": step})
    log.info(f"Checkpoint saved → {ckpt_path}")


def _load_checkpoint(
    fabric: Fabric,
    model: ViTClassifier,
    cfg: DictConfig,
    optimizer: AdamW | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> int:
    """Initialise from a pretraining checkpoint, or resume a training state.

    Two different things, deliberately kept as two different config keys:

    ``trainer.resume_from_checkpoint``
        A *pretraining* checkpoint. Only its ``encoder.*`` weights are read, the
        head is left at its fresh init, and training starts at step 0. This is
        what every benchmark probe uses, and its behaviour is unchanged.

    ``trainer.resume_from_state``
        A checkpoint written by :func:`_save_checkpoint` *in this script*. Model,
        optimizer and scheduler are restored in place and training continues from
        the saved step. Needed for runs longer than the 48 h submitit timeout:
        without it a timed-out run restarts from step 0 with a fresh head.

    Set at most one. ``resume_from_state`` wins if both are given, since a
    resume already carries the encoder weights the other key would supply.
    """
    state_path = cfg.trainer.get("resume_from_state")
    if state_path:
        state: dict[str, Any] = {"model": model}
        if optimizer is not None:
            state["optimizer"] = optimizer
        if scheduler is not None:
            state["scheduler"] = scheduler
        remainder = fabric.load(state_path, state)
        step = int(remainder.get("step", 0))
        log.info(f"Resumed full training state from {state_path} at step {step}.")
        return step

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
