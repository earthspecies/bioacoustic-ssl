"""Representation eval used during MAE pretraining.

Scores the frozen encoder's features on the multilabel POW eval set, per block
and per pooling (CLS, patch-GAP), reporting cmAP / macro-AUROC — the same metric
family as the downstream BirdSet eval. Keeps *all* clips (multi-species
included). Two scorers per embedding:

* ``lp_*`` — a closed-form ridge linear probe (headline; tracks the downstream
  *trained* proto probe, which a parameter-free scorer cannot).
* ``knn_*`` — a probe-free weighted-kNN scorer (secondary). Embeddings are
  mean-centred before cosine to counter the strong anisotropy of MAE features.

NOTE: all scoring runs in fp32. MAE embeddings are extremely anisotropic
(pairwise cosine ≈ 0.999), so a bf16 similarity matmul rounds every similarity
to the same value and collapses kNN to chance — keep this path out of autocast.
"""

from __future__ import annotations

import multiprocessing as mp
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from alp_data import dataset_from_config
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision

from bioacoustic_ssl.data import Compose
from bioacoustic_ssl.models.architectures.vit.encoder import ViTEncoder

KNN_K = 20


def _build_test_collate(transforms_yaml: str | Path, num_classes: int) -> Compose:
    """Build the ``test``-stage transform pipeline, resolving ``${data.num_classes}``."""
    container = OmegaConf.create(
        {"data": {"num_classes": num_classes}, "transforms": OmegaConf.load(transforms_yaml)}
    )
    stages = []
    for item in container.transforms:
        item = OmegaConf.to_container(item, resolve=True)
        if "test" in item.pop("_stage_", ["train", "val", "test"]):
            stages.append(instantiate(item))
    return Compose(stages)


def build_pow_eval(ds_yaml: str | Path, transforms_yaml: str | Path, loader_kwargs: dict):
    """Run the test pipeline once and cache the full multilabel eval set.

    Returns (specs_fp16_cpu (N,1,128,512), targets_int (N,C) multi-hot, stats).
    Decodes audio + builds spectrograms a single time; the cached tensors are
    reused for every periodic eval during the run. No clips are dropped.
    """
    ds_cfg = OmegaConf.load(ds_yaml)
    dataset, meta = dataset_from_config(instantiate(ds_cfg.test))
    num_classes = meta["mulitlabel_from_feature"]["num_classes"]

    collate = _build_test_collate(transforms_yaml, num_classes)
    loader = DataLoader(
        dataset,
        **loader_kwargs,
        collate_fn=collate,
        multiprocessing_context=mp.get_context("spawn"),
        drop_last=False,
        shuffle=False,
    )

    specs, labels = [], []
    for batch in loader:
        specs.append(batch["spectrogram"].half().cpu())
        labels.append(batch["label"].cpu())
    specs = torch.cat(specs)
    targets = torch.cat(labels).int()

    s = targets.sum(dim=1)
    stats = {
        "n_total": int(targets.shape[0]),
        "n_nocall": int((s == 0).sum()),
        "n_multi": int((s >= 2).sum()),
        "n_present": int((targets.sum(dim=0) >= 1).sum()),
        "num_classes": int(num_classes),
    }
    return specs, targets, stats


@torch.no_grad()
def extract_layerwise_pools(
    encoder: ViTEncoder, specs: torch.Tensor, device, batch_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block CLS and patch-GAP embeddings for the cached spectrograms.

    Returns two ``(L, N, D)`` float32 CPU tensors ``(cls, gap)`` from each
    transformer block's raw (pre-final-norm) output. Run outside autocast — the
    forward is fp32 (see module docstring; scale matters for the kNN cosine).
    """
    depth = len(encoder.blocks)
    cls_acc: list[list[torch.Tensor]] = [[] for _ in range(depth)]
    gap_acc: list[list[torch.Tensor]] = [[] for _ in range(depth)]

    for i in range(0, specs.shape[0], batch_size):
        x = specs[i:i + batch_size].float().to(device)
        hiddens = ViTEncoder.forward(encoder, x, return_hidden=True)  # list of (B, N+1, D)
        for layer, h in enumerate(hiddens):
            h = h.float()
            cls_acc[layer].append(h[:, 0].cpu())
            gap_acc[layer].append(h[:, 1:].mean(dim=1).cpu())

    cls = torch.stack([torch.cat(c) for c in cls_acc])  # (L, N, D)
    gap = torch.stack([torch.cat(g) for g in gap_acc])
    return cls, gap


def knn_multilabel_scores(
    emb: torch.Tensor, targets: torch.Tensor, device, k: int = KNN_K
) -> torch.Tensor:
    """Leave-one-out weighted-kNN soft class scores, (N, C).

    Embeddings are mean-centred (to counter MAE anisotropy) then L2-normalised
    for cosine kNN. Non-negative similarity weights; each query's score for a
    class is the similarity-weighted count of that class among its top-k
    neighbours. No per-row renormalisation (AP/AUROC are per-class rank metrics).
    """
    emb = emb.to(device).float()
    emb = emb - emb.mean(dim=0, keepdim=True)
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.T
    sim.fill_diagonal_(float("-inf"))  # leave-one-out
    k = min(k, sim.shape[0] - 1)
    vals, idx = sim.topk(k, dim=1)
    w = vals.clamp(min=0.0)  # drop negative-similarity neighbours
    neigh = targets.float().to(device)[idx]  # (N, k, C)
    return (w.unsqueeze(-1) * neigh).sum(dim=1)  # (N, C)


def ridge_probe_scores(
    emb: torch.Tensor, targets: torch.Tensor, device, frac: float = 0.7, lam: float = 1e2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form ridge linear probe on a fixed train/test split of the eval set.

    Standardises features on the train split, solves ridge regression to the
    multi-hot targets, and returns ``(test_scores (Nte,C), test_targets (Nte,C))``
    for the metric. One ``torch.linalg.solve`` — no training loop.
    """
    n = emb.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    cut = int(frac * n)
    tr, te = perm[:cut], perm[cut:]

    X = emb[tr].to(device).float()
    Y = targets[tr].float().to(device)
    mu, sd = X.mean(dim=0), X.std(dim=0) + 1e-6
    Xn = (X - mu) / sd
    eye = torch.eye(Xn.shape[1], device=device)
    W = torch.linalg.solve(Xn.T @ Xn + lam * eye, Xn.T @ Y)

    scores = ((emb[te].to(device).float() - mu) / sd) @ W
    return scores, targets[te].int().to(device)


def representation_metrics(
    scores: torch.Tensor, targets: torch.Tensor, num_classes: int, device
) -> dict:
    """cmAP / macro-AUROC for kNN scores against multi-hot targets.

    Headline values average only over classes present in the eval set (BirdSet
    cmAP convention); the raw macro-over-all-classes values are also returned
    for parity with the downstream probe (which uses ``average="macro"``).
    """
    scores = scores.to(device)
    targets_i = targets.int().to(device)
    present = targets_i.sum(dim=0) >= 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # per-absent-label spam
        ap = MultilabelAveragePrecision(num_labels=num_classes, average=None).to(device)(
            scores, targets_i
        )
        auroc = MultilabelAUROC(num_labels=num_classes, average=None).to(device)(
            scores, targets_i
        )

    return {
        "cmap": float(ap[present].mean()),
        "auroc": float(auroc[present].mean()),
        "cmap_macro48": float(ap.mean()),
        "auroc_macro48": float(auroc.mean()),
    }


def run_layerwise_eval(
    encoder: ViTEncoder,
    specs: torch.Tensor,
    targets: torch.Tensor,
    device,
    num_classes: int,
    k: int = KNN_K,
    batch_size: int = 256,
) -> dict:
    """Per-layer, per-pool representation metrics (ridge probe + kNN), for wandb.

    Logs ``eval/pow/{pool}/layer_{L}/{lp,knn}_{cmap,auroc,cmap_macro48,auroc_macro48}``
    plus per-pool best-layer scalars ``eval/pow/{pool}/best_{lp,knn}_{cmap,auroc}``
    and the argmax layer index ``.../best_..._layer``.
    """
    cls, gap = extract_layerwise_pools(encoder, specs, device, batch_size)

    results: dict[str, float] = {}
    for pool, embs in (("cls", cls), ("gap", gap)):
        per_layer = []
        for layer in range(embs.shape[0]):
            knn = representation_metrics(
                knn_multilabel_scores(embs[layer], targets, device, k), targets, num_classes, device
            )
            lp_scores, lp_targets = ridge_probe_scores(embs[layer], targets, device)
            lp = representation_metrics(lp_scores, lp_targets, num_classes, device)
            m = {f"knn_{n}": v for n, v in knn.items()} | {f"lp_{n}": v for n, v in lp.items()}
            per_layer.append(m)
            for name, val in m.items():
                results[f"eval/pow/{pool}/layer_{layer}/{name}"] = val

        for metric in ("lp_cmap", "lp_auroc", "knn_cmap", "knn_auroc"):
            vals = [m[metric] for m in per_layer]
            best = max(range(len(vals)), key=lambda i: vals[i])
            results[f"eval/pow/{pool}/best_{metric}"] = vals[best]
            results[f"eval/pow/{pool}/best_{metric}_layer"] = best

    return results
