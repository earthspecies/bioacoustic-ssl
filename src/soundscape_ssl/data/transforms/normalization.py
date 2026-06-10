import numpy as np
import torch

from soundscape_ssl.data.transforms.base import Transform


class PeakNormalize(Transform):
    def __init__(
        self,
        target_key: str = "audio",
    ) -> None:
        super().__init__()
        self.target_key = target_key

    def __call__(self, batch: list[dict]) -> list[dict]:
        for i in range(len(batch)):
            x = batch[i][self.target_key]
            batch[i][self.target_key] = self._normalise(x)
        return batch

    def _normalise(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        peak = x.abs().max()
        if peak == 0.0:
            return x
        x = x / peak
        return x


class Normalize(Transform):
    """Normalises a spectrogram with global mean and standard deviation.

    Designed to operate on *individual* datapoints before collation.
    Accepts ``torch.Tensor``, ``numpy.ndarray``, or a ``dict`` that
    contains the spectrogram under ``spec_key``.

    Applies ``(x - mean) / (std * 2)``, consistent with the BirdMAE
    feature extractor convention.

    Args:
        mean: Global dataset mean.
        std: Global dataset standard deviation.
        spec_key: Key in the sample dict containing the spectrogram.
    """

    def __init__(
        self,
        mean: float = -7.2,
        std: float = 4.43,
        target_key: str = "spectrogram",
    ) -> None:
        super().__init__()
        self.target_key = target_key
        self._mean = mean
        self._std = std

    def __call__(self, batch: list[dict]) -> list[dict]:
        for i in range(len(batch)):
            x = batch[i][self.target_key]
            batch[i][self.target_key] = self._normalise(x)
        return batch

    def _normalise(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> torch.Tensor | np.ndarray:
        return (x - self._mean) / (self._std * 2)
