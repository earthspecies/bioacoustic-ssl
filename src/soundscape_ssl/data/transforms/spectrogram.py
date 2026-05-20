from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from soundscape_ssl.data.transforms.base import Transform


class Spectrogram(Transform):
    """Converts a mono waveform to a (log) power spectrogram via STFT.

    Accepts a ``torch.Tensor`` of shape ``(T,)`` or a ``dict`` containing
    the waveform under ``audio_key``.

    The output is always a 3-D tensor of shape
    ``(1, n_fft//2+1, time_frames)`` (channel, freq, time) suitable for
    the MAE ``PatchEmbed`` layer.

    Wraps ``torchaudio.transforms.Spectrogram`` followed by
    ``torchaudio.transforms.AmplitudeToDB``.

    Args:
        n_fft: FFT size. Frequency resolution = ``n_fft // 2 + 1`` bins.
        win_length: Window length in samples. Defaults to ``n_fft``.
        hop_length: Hop size between consecutive frames in samples.
        power: Exponent for the magnitude spectrogram (1 = amplitude,
            2 = power). Must match the ``stype`` of ``AmplitudeToDB``.
        top_db: Dynamic range clipping for the dB conversion. ``None``
            disables the dB step entirely.
        audio_key: Key in the sample dict for the input waveform.
        output_key: Key in the sample dict where the spectrogram is stored.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 1024,
        win_length: int = None,
        hop_length: int = 320,
        n_mels: int = 128,
        f_min: float = 50.0,
        f_max: float = None,
        power: float = 2.0,
        top_db: float | None = 80.0,
        audio_key: str = "audio",
        output_key: str = "spectrogram",
        drop_audio: bool = True
    ) -> None:
        super().__init__()
        self._mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length or n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
        )
        self._to_db = (
            T.AmplitudeToDB(top_db=top_db, stype="power")
            if top_db is not None
            else None
        )
        self.audio_key = audio_key
        self.output_key = output_key
        self.drop_audio = drop_audio
        self.top_db = top_db

    def __call__(self, batch: list[dict]) -> list[dict]:
        for i in range(len(batch)):
            batch[i][self.output_key] = self._compute(batch[i][self.audio_key])
            if self.drop_audio:
                del batch[i][self.audio_key]
        return batch

    def _compute(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)

        spec = self._mel_spectrogram(audio)  # (freq, time)

        if self._to_db is not None:
            # normalize between [-1, 1]
            spec = self._to_db(spec)
            spec = 2 * (spec + self.top_db) / self.top_db - 1

        spec = spec.unsqueeze(0)  # (1, freq, time)

        return spec
