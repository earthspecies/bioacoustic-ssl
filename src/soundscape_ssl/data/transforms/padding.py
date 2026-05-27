import torch
import torch.nn.functional as F

from soundscape_ssl.data.transforms.base import Transform


class Padding(Transform):
    def __init__(
            self,
            target_shape: tuple[int, int] = (512, 320),
            pad_value: int | float = 0.0,
            target_key: str = "spectrogram",
            ) -> None:
        super().__init__()
        self.target_shape = target_shape
        self.pad_value = pad_value
        self.target_key = target_key

    def __call__(self, batch: list[dict]) -> list[dict]:
        for i in range(len(batch)):
            x = batch[i][self.target_key]
            batch[i][self.target_key] = self._pad(x)
        return batch

    def _pad(self, data: torch.Tensor) -> torch.Tensor:
        target_h, target_w = self.target_shape

        # Truncate if larger than target
        data = data[..., :target_h, :target_w]

        # Zero-pad if smaller than target
        pad_h = target_h - data.shape[-2]
        pad_w = target_w - data.shape[-1]
        if pad_h > 0 or pad_w > 0:
            data = F.pad(data, (0, pad_w, 0, pad_h), mode="constant", value=self.pad_value)

        return data


class WavePadding(Transform):
    def __init__(
            self,
            target_shape: int = 160_000,
            pad_value: int | float = 0.0,
            target_key: str = "audio",
            ) -> None:
        super().__init__()
        self.target_shape = target_shape
        self.pad_value = pad_value
        self.target_key = target_key

    def __call__(self, batch: list[dict]) -> list[dict]:
        for i in range(len(batch)):
            x = batch[i][self.target_key]
            batch[i][self.target_key] = self._pad(x)
        return batch

    def _pad(self, data: torch.Tensor) -> torch.Tensor:

        # Truncate if larger than target
        data = data[..., :self.target_shape]

        # Zero-pad if smaller than target
        pad_w = self.target_shape - data.shape[-1]
        if pad_w > 0:
            # Using (0, pad_w) ensures we only pad the last dimension,
            # supporting both 1D [length] and 2D [channels, length] waveforms.
            data = F.pad(
                data, (0, pad_w), mode="constant", value=self.pad_value
            )

        return data


class BatchPadding(Transform):
    def __init__(
            self,
            target_shape: tuple[int, int] = (512, 320),
            pad_value: int | float = 0.0,
            target_key: str = "spectrogram",
            ) -> None:
        super().__init__()
        self.target_shape = target_shape
        self.pad_value = pad_value
        self.target_key = target_key

    def __call__(self, batch: list[dict]) -> list[dict]:
        x = batch[self.target_key]
        batch[self.target_key] = self._pad(x)
        return batch

    def _pad(self, data: torch.Tensor) -> torch.Tensor:
        target_h, target_w = self.target_shape

        # Truncate if larger than target
        data = data[..., :target_h, :target_w]

        # Zero-pad if smaller than target
        pad_h = target_h - data.shape[-2]
        pad_w = target_w - data.shape[-1]
        if pad_h > 0 or pad_w > 0:
            data = F.pad(data, (0, pad_w, 0, pad_h), mode="constant", value=self.pad_value)

        return data
