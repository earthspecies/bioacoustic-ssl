from typing import Any

import torch.nn as nn

from soundscape_ssl.data.transforms.base import Transform


class TorchAudioTransform(Transform):
    """Wraps a ``torchaudio.transforms`` module for the sample-dict pipeline.

    ``torchaudio`` transforms expect a ``(..., time)`` tensor. This wrapper
    temporarily unsqueezes a channel dimension for 1-D (mono) audio so the
    transform sees ``(1, T)``, then squeezes it back.

    Example::

        import torchaudio
        resample = TorchAudioTransform(
            torchaudio.transforms.Resample(orig_freq=32_000, new_freq=16_000)
        )

    Args:
        transform: Any callable ``torchaudio`` transform (``nn.Module``).
        audio_key: Key in the sample dict containing the audio tensor.
        training_only: If ``True``, this transform is skipped when the module
            is in eval mode (``.eval()``). Defaults to ``True``.
    """

    def __init__(
        self,
        transform: Any,
        audio_key: str = "audio",
    ) -> None:
        super().__init__()
        self.transform = transform
        self.audio_key = audio_key

    def __call__(self, sample: dict) -> dict:
        audio = sample[self.audio_key]
        squeeze = audio.ndim == 1
        if squeeze:
            audio = audio.unsqueeze(0)  # (1, T)
        audio = self.transform(audio)
        if squeeze:
            audio = audio.squeeze(0)
        return {**sample, self.audio_key: audio}


class TorchAudiomentationsTransform(Transform):
    """Wraps a ``torch_audiomentations`` transform for the sample-dict pipeline.

    ``torch_audiomentations`` transforms expect ``(batch, channels, time)``
    tensors. This wrapper adds and removes the batch/channel dimensions as
    needed and passes ``sample_rate`` to the transform.

    ``sample_rate`` is resolved in this order:
    1. The ``sample_rate`` argument passed to ``__init__`` (explicit override).
    2. ``sample[sample_rate_key]`` from the sample dict.

    Example::

        from torch_audiomentations import Gain
        gain = TorchAudiomentationsTransform(
            Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5),
            sample_rate=16_000,
        )

    Args:
        transform: Any ``torch_audiomentations`` transform instance.
        sample_rate: Sample rate to pass to the transform. If ``None``,
            falls back to ``sample[sample_rate_key]``.
        audio_key: Key in the sample dict containing the audio tensor.
        sample_rate_key: Key in the sample dict containing the sample rate.
        training_only: If ``True`` (default), this transform is skipped when
            the module is in eval mode (``.eval()``). Defaults to ``True``
            since torch_audiomentations is an augmentation library.
    """

    def __init__(
        self,
        transform: Any,
        audio_key: str = "audio",
    ) -> None:
        super().__init__()
        self.transform = transform
        self.audio_key = audio_key

    def __call__(self, batch: list[dict]) -> list[dict]:
        for i in range(len(batch)):
            audio = batch[i][self.audio_key]

            # torch_audiomentations expects (batch=1, channels, time).
            ndim = audio.ndim
            if ndim == 1:
                audio = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
            elif ndim == 2:
                audio = audio.unsqueeze(0)               # (1, C, T)

            result = self.transform(audio)
            audio = result.samples  # ObjectDict with .samples

            if ndim == 1:
                audio = audio.squeeze(0).squeeze(0)
            elif ndim == 2:
                audio = audio.squeeze(0)

            batch[i][self.audio_key] = audio

        return batch

    def __repr__(self):
        return self.transform.__repr__()
