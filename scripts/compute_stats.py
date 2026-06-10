"""Estimate mean and std of the spectrogram dataset over a fixed number of samples."""

from __future__ import annotations

import multiprocessing as mp

import hydra
import pandas as pd
import torch
from esp_data.datasets import XenoCanto
from esp_data import dataset_from_config
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from soundscape_ssl.data.datasets import A2ODetections, ArbimonDetections
from soundscape_ssl.data.iterable_dataset import MixedStreamingDataset
from soundscape_ssl.data.transforms import Compose, Padding, Spectrogram, TimeShift

NUM_SAMPLES = 20_000


def build_transforms(cfg_list, stage: str) -> list:   # stage: "train", "val", or "test"
    transforms = []
    for item in cfg_list:
        item = OmegaConf.to_container(item, resolve=True)
        allowed = item.pop("_stage_", ["train", "val", "test"])
        if stage in allowed:
            transforms.append(hydra.utils.instantiate(item))
    return transforms


def estimate_stats(
    loader: DataLoader,
    num_samples: int = 1_000
) -> tuple[float, float]:

    total_sum = 0.0
    total_sq = 0.0
    total_n = 0

    num_batches = (num_samples + loader.batch_size - 1) // loader.batch_size

    for batch in tqdm(loader, total=num_batches):
        for data in batch:
            spec = data["spectrogram"]
            n = spec.numel()

            total_sum += spec.sum().item()
            total_sq += (spec ** 2).sum().item()
            total_n += n

    mean = total_sum / total_n
    var = (total_sq / total_n) - (mean ** 2)

    # Guard against tiny negative values caused by floating-point error
    std = max(var, 0.0) ** 0.5
    return mean, std


@hydra.main(version_base=None, config_path="../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    ds_configs = hydra.utils.instantiate(list(cfg.data.datasets.values()))

    datasets = [dataset_from_config(config) for config in ds_configs]

    ds = MixedStreamingDataset(datasets)

    loader = DataLoader(
        ds,
        **cfg.data.loaders.val_test,
        shuffle=True,
        collate_fn=Compose(build_transforms(cfg.data.transforms, "test")),
        multiprocessing_context=mp.get_context("spawn")
    )

    mean, std = estimate_stats(
        loader,
        num_samples=NUM_SAMPLES
    )

    print(f"\nResults over ~{NUM_SAMPLES} samples:")
    print(f"  mean = {mean:.6f}")
    print(f"  std  = {std:.6f}")


if __name__ == "__main__":
    main()
