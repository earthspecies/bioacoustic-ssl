import random
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import IterableDataset

from soundscape_ssl.data.transforms.base import Compose, Transform


class MixedStreamingDataset(IterableDataset):
    """Interleaves multiple map-style datasets with configurable weights.

    Iterates indefinitely (or for ``num_samples`` total samples) by cycling
    through each dataset in proportion to its weight and reshuffling +
    restarting a dataset whenever it is exhausted.

    Args:
        datasets: Sequence of map-style Dataset instances, each implementing
            ``__len__`` and ``__getitem__``.
        weights: Integer sampling weights. ``weights=[2, 1, 1]`` draws 2
            samples from ``datasets[0]`` for every 1 drawn from
            ``datasets[1]`` and ``datasets[2]``. ``None`` defaults to equal
            weights.
        num_samples: Total number of samples to yield across all DataLoader
            workers combined. ``None`` yields indefinitely.
        transform: Optional per-sample transform applied after retrieval.
    """

    def __init__(
        self,
        datasets: Sequence[Any],
        weights: Sequence[int] | None = None,
        num_samples: int | None = None,
        transform: Transform | Sequence[Transform] | None = None,
    ) -> None:
        if weights is None:
            weights = [1] * len(datasets)
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have the same length")

        self.datasets = list(datasets)
        self.weights = list(weights)
        self.num_samples = num_samples

        if isinstance(transform, (list, tuple)):
            self.transform = Compose(transform)
        else:
            self.transform = transform

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _make_pattern(self) -> list[int]:
        """Expand weights into one cycle of dataset indices.

        Returns:
            List of dataset indices repeated according to their weights.
        """
        pattern: list[int] = []
        for i, w in enumerate(self.weights):
            pattern.extend([i] * w)
        return pattern

    # ------------------------------------------------------------------
    # iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Any]:
        worker_info = torch.utils.data.get_worker_info()

        # Each worker gets a unique, reproducible seed so shuffles differ.
        rng = random.Random(torch.initial_seed() % (2**32))

        # Per-dataset shuffled index lists; each resets independently.
        indices = [list(range(len(ds))) for ds in self.datasets]
        for idx in indices:
            rng.shuffle(idx)
        pointers = [0] * len(self.datasets)

        # Split total steps evenly across workers.
        if self.num_samples is not None and worker_info is not None:
            n, w = worker_info.num_workers, worker_info.id
            worker_steps = self.num_samples // n + (1 if w < self.num_samples % n else 0)
        else:
            worker_steps = self.num_samples  # None → infinite

        # Cyclic sampling pattern; reshuffled after each full cycle for mixing.
        pattern = self._make_pattern()
        cycle = list(pattern)
        rng.shuffle(cycle)
        pos = 0
        steps = 0

        while worker_steps is None or steps < worker_steps:
            if pos >= len(cycle):
                pos = 0
                rng.shuffle(cycle)

            ds_idx = cycle[pos]
            pos += 1

            # Advance pointer; reset + reshuffle if the dataset is exhausted.
            if pointers[ds_idx] >= len(indices[ds_idx]):
                rng.shuffle(indices[ds_idx])
                pointers[ds_idx] = 0

            sample = self.datasets[ds_idx][indices[ds_idx][pointers[ds_idx]]]
            pointers[ds_idx] += 1

            if self.transform is not None:
                sample = self.transform(sample)

            yield sample
            steps += 1
