import multiprocessing as mp
import time
import random

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from soundscape_ssl.data import Compose, MixedStreamingDataset

def build_transforms(cfg_list: ListConfig, stage: str) -> list:   # stage: "train", "val", or "test"
    transforms = []
    for item in cfg_list:
        item = OmegaConf.to_container(item, resolve=True)
        allowed = item.pop("_stage_", ["train", "val", "test"])
        if stage in allowed:
            transforms.append(instantiate(item))
    return transforms


@hydra.main(version_base=None, config_path="../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:

    datasets = [instantiate(config) for config in list(cfg.data.datasets.values())]
    for ds in datasets:
        print(ds.__class__.__name__, len(ds))

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
        worker_init_fn=worker_init_fn
    )

    num_batches = 1000
    t0 = time.perf_counter()
    for batch_idx, _ in tqdm(enumerate(loader), total=num_batches):
        if batch_idx == num_batches:
            break

    print()
    print(f"Time for {num_batches = }: {(time.perf_counter() - t0):.2f}")


def worker_init_fn(worker_id: int) -> None:
    time.sleep(worker_id * 0.2 + random.uniform(0, 0.5))


if __name__ == "__main__":
    main()
