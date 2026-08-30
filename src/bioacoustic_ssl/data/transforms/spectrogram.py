from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from torchaudio.compliance import kaldi

from bioacoustic_ssl.data.transforms.base import Transform


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


class BatchSpectrogram(Transform):
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
        batch[self.output_key] = self._compute(batch[self.audio_key])
        if self.drop_audio:
            del batch[self.audio_key]
        return batch

    def _compute(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)  # (B, 1, T) -> (B, T)

        # Channel axis before the dB step: ``AmplitudeToDB`` reads its reduction
        # axes off the rank, so a 3-D (B, n_mels, frames) input would floor
        # ``top_db`` below the *batch* maximum and make a clip's spectrogram
        # depend on what it was batched with. 4-D floors it per sample.
        spec = self._mel_spectrogram(audio).unsqueeze(1)  # (B, 1, freq, time)

        if self._to_db is not None:
            # normalize between [-1, 1]
            spec = self._to_db(spec)
            spec = 2 * (spec + self.top_db) / self.top_db - 1

        return spec


class BatchKaldiFbank(Transform):
    """Kaldi log-mel filterbank front-end for the published external backbones.

    Reproduces the input pipelines that Bird-MAE and AudioMAE were pretrained
    with, so their released weights can be probed with our heads:

    * Bird-MAE (``DBD-research-group/Bird-MAE-Base``) — 32 kHz, 512 frames,
      ``mean=-7.2``, ``std=4.43``, frame axis padded with the batch minimum.
    * AudioMAE (``vit_base_patch16_1024_128.audiomae_as2m``) — 16 kHz, 1024
      frames, ``mean=-4.2677393``, ``std=4.5689974``, zero padding.

    Unlike :class:`BatchSpectrogram` (dB mel rescaled to ``[-1, 1]``),
    ``kaldi.fbank`` returns natural-log mel energies and is *not* scale
    invariant — a gain change shifts every value by a constant. Both reference
    pipelines therefore strip the waveform DC offset rather than peak
    normalising, and fitted ``mean``/``std`` on non-peak-normalised audio, so
    configs using this transform must drop ``PeakNormalize``.

    ``kaldi.fbank`` has no batched form, so frames are computed per sample in a
    loop; measured at ~7 ms/clip for 5 s @ 32 kHz, on par with the batched
    ``MelSpectrogram`` path, i.e. not a dataloader bottleneck.

    Output keeps this repo's ``(B, 1, n_mels, frames)`` layout so the downstream
    spectrogram stages (``SpecAugment``, padding) keep masking the axes they were
    configured for. The external backbones patch-embed the transpose of that, so
    :class:`TransposeSpec` must be the final stage of the pipeline.

    Args:
        sample_rate: Sample rate of the incoming waveform. The waveform must
            already be resampled to whatever the target model expects; this
            value is only forwarded to ``kaldi.fbank``.
        target_length: Frame count the backbone was pretrained on. The frame
            axis is truncated or padded to exactly this many frames.
        mean: Global log-mel mean of the model's pretraining corpus.
        std: Global log-mel standard deviation. Normalisation is
            ``(x - mean) / (std * 2)``, as in both reference implementations.
        pad_value: Value the frame axis is padded with. ``"min"`` uses the
            minimum of the batch (what Bird-MAE's feature extractor does, so it
            is mildly batch-composition dependent); a float pads with that
            constant (AudioMAE uses ``0.0``).
        num_mel_bins, frame_length, frame_shift, htk_compat, window_type,
            use_energy, dither: forwarded to ``kaldi.fbank``. The defaults are
            the settings shared by both models.
        remove_dc: Subtract each waveform's mean before framing, as both
            reference pipelines do.
        audio_key: Key in the batch dict holding the waveform.
        output_key: Key the spectrogram is written to.
        drop_audio: Delete ``audio_key`` after computing the spectrogram.
    """

    def __init__(
        self,
        sample_rate: int,
        target_length: int,
        mean: float,
        std: float,
        pad_value: float | str = "min",
        num_mel_bins: int = 128,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        htk_compat: bool = True,
        window_type: str = "hanning",
        use_energy: bool = False,
        dither: float = 0.0,
        remove_dc: bool = True,
        audio_key: str = "audio",
        output_key: str = "spectrogram",
        drop_audio: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.mean = mean
        self.std = std
        self.pad_value = pad_value
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.htk_compat = htk_compat
        self.window_type = window_type
        self.use_energy = use_energy
        self.dither = dither
        self.remove_dc = remove_dc
        self.audio_key = audio_key
        self.output_key = output_key
        self.drop_audio = drop_audio

    def __call__(self, batch: dict) -> dict:
        batch[self.output_key] = self._compute(batch[self.audio_key])
        if self.drop_audio:
            del batch[self.audio_key]
        return batch

    def _compute(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)  # (B, 1, T) -> (B, T)

        if self.remove_dc:
            audio = audio - audio.mean(dim=-1, keepdim=True)

        spec = torch.stack([
            kaldi.fbank(
                waveform.unsqueeze(0),
                sample_frequency=self.sample_rate,
                num_mel_bins=self.num_mel_bins,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                htk_compat=self.htk_compat,
                window_type=self.window_type,
                use_energy=self.use_energy,
                dither=self.dither,
            )
            for waveform in audio
        ])  # (B, frames, n_mels)

        spec = self._fit_frames(spec)
        spec = (spec - self.mean) / (self.std * 2)

        return spec.transpose(1, 2).unsqueeze(1)  # (B, 1, n_mels, frames)

    def _fit_frames(self, spec: torch.Tensor) -> torch.Tensor:
        frames = spec.shape[1]
        if frames > self.target_length:
            return spec[:, :self.target_length]
        if frames < self.target_length:
            pad = float(spec.min()) if self.pad_value == "min" else float(self.pad_value)
            return F.pad(spec, (0, 0, 0, self.target_length - frames), value=pad)
        return spec


class BatchMinMaxSpectrogram(Transform):
    """Mel front-end with per-sample min-max scaling, as BAT's own processor does.

    Reproduces ``BatAudioProcessor`` (``processing_bat.py``): batched
    ``MelSpectrogram`` -> ``AmplitudeToDB("power", top_db=80)`` -> per-sample
    min-max to ``[0, 1]``.

    It exists separately from :class:`BatchSpectrogram` for two reasons, both
    about *where* the per-sample boundary sits:

    * :class:`BatchSpectrogram` hard-codes the affine ``2*(dB+top_db)/top_db - 1``
      rescale. Min-max is invariant to that (it is monotone affine), so the
      rescale is not the real problem — the clamp below is.
    * ``AmplitudeToDB``'s ``top_db`` clamp is relative to the maximum of the
      tensor it is handed, and ``torchaudio`` picks the reduction axes from the
      rank: a 3-D ``(B, n_mels, frames)`` input clamps against the **batch**
      maximum, a 4-D ``(B, 1, n_mels, frames)`` input clamps **per sample**. BAT
      unsqueezes before the dB step, so it gets the per-sample floor. Clamping
      is not affine, so that choice changes the min-max result — hence the
      channel axis is added here before ``_to_db``, not after.

    Consequences worth knowing:

    * The whole chain is gain invariant (a scale becomes a constant dB shift,
      which the clamp and min-max both absorb), so ``PeakNormalize`` is a no-op
      for this arm, and so is any ``Gain`` augmentation applied *after* mixing.
    * Per-sample min-max is sensitive to single loud transients: one clipped
      sample compresses the rest of the clip. That is BAT's own recipe.

    Output keeps this repo's ``(B, 1, n_mels, frames)`` layout so the downstream
    ``SpecAugment`` / padding stages keep masking the axes they were configured
    for; :class:`TransposeSpec` must be the final stage.

    Args:
        sample_rate: Sample rate of the incoming waveform, which must already be
            resampled to what the model expects (16 kHz for BAT).
        n_fft, win_length, hop_length, n_mels, f_min, f_max, power: forwarded to
            ``torchaudio.transforms.MelSpectrogram``. Defaults are BAT's.
        top_db: Dynamic-range clamp for the dB step, applied per sample.
        eps: Added to the min-max denominator, matching ``BatAudioProcessor``.
        audio_key: Key in the batch dict holding the waveform.
        output_key: Key the spectrogram is written to.
        drop_audio: Delete ``audio_key`` after computing the spectrogram.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 1024,
        win_length: int = None,
        hop_length: int = 160,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = None,
        power: float = 2.0,
        top_db: float = 80.0,
        eps: float = 1e-8,
        audio_key: str = "audio",
        output_key: str = "spectrogram",
        drop_audio: bool = True,
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
        self._to_db = T.AmplitudeToDB(stype="power", top_db=top_db)
        self.eps = eps
        self.audio_key = audio_key
        self.output_key = output_key
        self.drop_audio = drop_audio

    def __call__(self, batch: dict) -> dict:
        batch[self.output_key] = self._compute(batch[self.audio_key])
        if self.drop_audio:
            del batch[self.audio_key]
        return batch

    def _compute(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)  # (B, 1, T) -> (B, T)

        # Channel axis before the dB step: that is what makes the top_db clamp
        # per sample rather than per batch (see the class docstring).
        spec = self._mel_spectrogram(audio).unsqueeze(1)  # (B, 1, n_mels, frames)
        spec = self._to_db(spec)

        flat = spec.flatten(1)
        min_, max_ = flat.aminmax(dim=1, keepdim=True)
        flat = (flat - min_) / (max_ - min_ + self.eps)

        return flat.reshape(spec.shape)


class TransposeSpec(Transform):
    """Swap the frequency and time axes of the batched spectrogram.

    Bird-MAE and AudioMAE patch-embed a ``(B, 1, frames, n_mels)`` input — the
    transpose of this repo's ``(B, 1, n_mels, frames)`` convention. Staying in
    the house layout until the very end lets ``SpecAugment`` and the padding
    transforms keep masking the axes they were configured for, so this belongs
    last in the pipeline, right before the model.
    """

    def __init__(self, target_key: str = "spectrogram") -> None:
        super().__init__()
        self.target_key = target_key

    def __call__(self, batch: dict) -> dict:
        batch[self.target_key] = batch[self.target_key].transpose(-2, -1).contiguous()
        return batch
