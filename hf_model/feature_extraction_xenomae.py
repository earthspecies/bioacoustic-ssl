"""The mel front-end the released XenoMAE encoder was trained on.

Ships inside the HuggingFace model repo. Needs ``torchaudio`` on top of ``torch``
and ``transformers``: the released spectrogram is ``torchaudio``'s
``MelSpectrogram`` and ``AmplitudeToDB``, and calling them is the only way to
guarantee the input matches pretraining to the last bit.

The chain, which is ``soundscape_ssl.data.transforms`` at inference time:

1. peak-normalise the waveform to ``[-1, 1]``;
2. pad or truncate it to a 5 s window at 32 kHz;
3. 128-bin mel power spectrogram, 1024-point FFT, 320-sample hop, 50 Hz floor;
4. power-to-dB with an 80 dB floor, then the affine rescale
   ``2 * (dB + 80) / 80 - 1``;
5. pad or truncate to (128 mels, 512 frames).

**A note on the dB floor.** ``torchaudio``'s ``AmplitudeToDB`` takes its
``top_db`` floor relative to the maximum of the tensor it is handed, and picks
the reduction axes from the rank: hand it a 3-D ``(batch, mels, frames)`` tensor
and the floor is the maximum of the whole *batch*, which makes a clip's
spectrogram depend on what else was in its batch. This extractor adds the
channel axis before the dB step, so its floor is per sample; the training
pipeline (``BatchSpectrogram``) does the same now, but the released weights were
pretrained before that fix, under the batch-max floor. The two agree at batch
size 1 and differ only in near-silent bins otherwise.

Longer recordings are truncated to the 5 s window, not windowed: the encoder has
a fixed 512-frame position table. Window them yourself and batch the windows.
"""

from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from transformers import BatchFeature, SequenceFeatureExtractor


@lru_cache(maxsize=4)
def _mel_spectrogram(
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    f_min: float,
    f_max: float | None,
    power: float,
) -> T.MelSpectrogram:
    """Build (and cache) the mel filterbank for one set of front-end parameters.

    Cached because the filterbank is rebuilt on every construction otherwise, and
    kept out of the extractor's ``__dict__`` because that dict is what
    ``save_pretrained`` serialises to JSON.
    """
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=power,
    )


@lru_cache(maxsize=4)
def _amplitude_to_db(top_db: float) -> T.AmplitudeToDB:
    """Build (and cache) the power-to-dB conversion with a ``top_db`` floor."""
    return T.AmplitudeToDB(stype="power", top_db=top_db)


class XenoMAEFeatureExtractor(SequenceFeatureExtractor):
    """Turns raw audio into the spectrogram the released encoder expects.

    Args:
        num_mel_bins: Mel bins, and the height of the model input.
        num_frames: Frames of the model input. The spectrogram is padded or
            truncated to this many.
        sampling_rate: Sampling rate the audio must already be at. A mismatch is
            an error rather than a silent resample — resample it yourself, with
            whatever quality your data deserves.
        clip_seconds: Length of the analysis window, in seconds.
        n_fft: FFT size.
        win_length: Window length in samples. ``None`` uses ``n_fft``.
        hop_length: Hop between frames, in samples.
        f_min: Lowest mel filter edge, in Hz.
        f_max: Highest mel filter edge, in Hz. ``None`` uses Nyquist.
        power: Exponent of the magnitude spectrogram. 2 is a power spectrogram,
            which is what the dB step below assumes.
        top_db: Dynamic range kept by the dB step, and the divisor of the affine
            rescale that follows it.
        do_peak_normalize: Divide each waveform by its peak amplitude. The
            training pipeline did, and the dB chain is not scale invariant.
        padding_value: Value the waveform and the spectrogram are padded with.
        return_attention_mask: Kept for interface compatibility and ``False``:
            every clip is fitted to the same fixed window, so there is nothing
            for a mask to say.
        **kwargs: Forwarded to :class:`~transformers.SequenceFeatureExtractor`.
    """

    model_input_names = ["input_values"]

    def __init__(
        self,
        num_mel_bins: int = 128,
        num_frames: int = 512,
        sampling_rate: int = 32_000,
        clip_seconds: float = 5.0,
        n_fft: int = 1024,
        win_length: int | None = None,
        hop_length: int = 320,
        f_min: float = 50.0,
        f_max: float | None = None,
        power: float = 2.0,
        top_db: float = 80.0,
        do_peak_normalize: bool = True,
        padding_value: float = 0.0,
        return_attention_mask: bool = False,
        **kwargs: Any,
    ) -> None:
        """Store the front-end parameters, mirrored into ``preprocessor_config.json``."""
        kwargs.pop("feature_size", None)
        super().__init__(
            feature_size=num_mel_bins,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.num_mel_bins = num_mel_bins
        self.num_frames = num_frames
        self.clip_seconds = clip_seconds
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        self.top_db = top_db
        self.do_peak_normalize = do_peak_normalize

    @property
    def clip_samples(self) -> int:
        """Length of the analysis window, in samples."""
        return int(self.sampling_rate * self.clip_seconds)

    def __call__(
        self,
        audio: "torch.Tensor | np.ndarray | list",
        sampling_rate: int | None = None,
        return_tensors: str | None = "pt",
        **kwargs: Any,
    ) -> BatchFeature:
        """Featurise one clip or a batch of them.

        Args:
            audio: A mono waveform — a 1-D tensor or array, a list of them, or a
                2-D ``(batch, samples)`` tensor. Clips need not be the same
                length; each is padded or truncated to the analysis window.
            sampling_rate: Sampling rate of ``audio``, checked against the rate
                the model was trained at. Pass it; leaving it ``None`` asserts
                the audio is already at ``self.sampling_rate``.
            return_tensors: ``"pt"`` (default) or ``"np"``.
            **kwargs: Ignored, for signature compatibility with other extractors.

        Returns:
            :class:`~transformers.BatchFeature` with ``input_values`` of shape
            ``(batch, 1, num_mel_bins, num_frames)``, ready for
            :class:`XenoMAEModel`.

        Raises:
            ValueError: If ``sampling_rate`` is not the rate the model expects,
                or if ``audio`` is not a batch of mono waveforms.
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"This model was trained at a sampling rate of {self.sampling_rate} Hz "
                f"and got {sampling_rate} Hz. Resample the audio first."
            )

        waveforms = torch.stack([self._fit_window(clip) for clip in self._as_clips(audio)])
        spectrogram = _mel_spectrogram(
            self.sampling_rate,
            self.n_fft,
            self.win_length or self.n_fft,
            self.hop_length,
            self.num_mel_bins,
            self.f_min,
            self.f_max,
            self.power,
        )(waveforms).unsqueeze(1)

        # The channel axis goes on before the dB step, and that is what makes the
        # top_db floor per sample rather than per batch — see the module
        # docstring.
        spectrogram = _amplitude_to_db(self.top_db)(spectrogram)
        spectrogram = 2 * (spectrogram + self.top_db) / self.top_db - 1

        return BatchFeature(
            {"input_values": self._fit_frames(spectrogram)}, tensor_type=return_tensors
        )

    def _as_clips(self, audio: "torch.Tensor | np.ndarray | list") -> list[torch.Tensor]:
        """Normalise the accepted input forms into a list of 1-D float32 waveforms.

        Raises:
            ValueError: If a waveform is not mono, or has more than two axes.
        """
        if isinstance(audio, (list, tuple)):
            if isinstance(audio[0], (list, tuple, np.ndarray, torch.Tensor)):
                return [clip for item in audio for clip in self._as_clips(item)]
            audio = np.asarray(audio)  # a bare list of samples, i.e. one clip

        tensor = audio if isinstance(audio, torch.Tensor) else torch.as_tensor(np.asarray(audio))
        tensor = tensor.to(torch.float32)

        if tensor.ndim == 1:
            return [tensor]
        if tensor.ndim == 2:
            return list(tensor)
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            return list(tensor.squeeze(1))
        raise ValueError(
            f"Expected a mono waveform, a list of them or a (batch, samples) tensor, "
            f"got shape {tuple(tensor.shape)}."
        )

    def _fit_window(self, waveform: torch.Tensor) -> torch.Tensor:
        """Peak-normalise a waveform and fit it to the analysis window."""
        if self.do_peak_normalize:
            peak = waveform.abs().max()
            if peak > 0.0:
                waveform = waveform / peak

        waveform = waveform[..., : self.clip_samples]
        missing = self.clip_samples - waveform.shape[-1]
        if missing > 0:
            waveform = F.pad(waveform, (0, missing), value=self.padding_value)
        return waveform

    def _fit_frames(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Fit a spectrogram to the model's ``(num_mel_bins, num_frames)`` input."""
        spectrogram = spectrogram[..., : self.num_mel_bins, : self.num_frames]
        pad_mels = self.num_mel_bins - spectrogram.shape[-2]
        pad_frames = self.num_frames - spectrogram.shape[-1]
        if pad_mels > 0 or pad_frames > 0:
            spectrogram = F.pad(
                spectrogram, (0, pad_frames, 0, pad_mels), value=self.padding_value
            )
        return spectrogram


__all__ = ["XenoMAEFeatureExtractor"]
