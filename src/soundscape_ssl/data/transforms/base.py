from typing import Callable, Sequence

import torch.nn as nn


class Transform:
    """Base class. All transforms operate on a sample dict."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, sample: dict) -> dict:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(Transform):
    """Chain transforms sequentially.

    Registers any ``nn.Module`` transforms as submodules so that calling
    ``.train()`` / ``.eval()`` on the ``Compose`` cascades to all children.
    Non-module callables (e.g. ``default_collate``) are kept in the plain
    list and are always applied regardless of training mode.
    """

    def __init__(self, transforms: Sequence) -> None:
        super().__init__()
        self.transforms = list(transforms)

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        lines = [f"  ({i}): {t!r}" for i, t in enumerate(self.transforms)]
        return "Compose(\n" + "\n".join(lines) + "\n)"


# class PerSampleTransform(Transform):
#     def __init__(self, transform: Transform) -> None:
#         self.transform = transform

#     def __call__(self, batch: list[dict]) -> list[dict]:
#         for i in range(len(batch)):
#             batch[i] = self.transform(batch[i])
#         return batch

#     def __repr__(self):
#         return self.transform.__repr__()


class Lambda(Transform):
    """Wrap an arbitrary callable as a Transform."""

    def __init__(self, fn: Callable[[dict], dict]) -> None:
        super().__init__()
        self.fn = fn

    def __call__(self, sample: dict) -> dict:
        return self.fn(sample)

    def __repr__(self) -> str:
        return f"Lambda({self.fn.__name__})"
