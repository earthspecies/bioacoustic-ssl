from typing import Literal

import torch
from torch.distributions import Beta, Binomial, Dirichlet

from bioacoustic_ssl.data.transforms.base import Transform


def _fit_to_length(audio: torch.Tensor, length: int) -> torch.Tensor:
    T = audio.shape[-1]
    if T < length:
        pad = length - T
        return torch.nn.functional.pad(audio, (0, pad))
    offset = int(torch.randint(0, T - length + 1, (1,)).item()) if T > length else 0
    return audio[..., offset : offset + length]


def _union_labels(base: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    return torch.cat([base.unsqueeze(0), others], dim=0).sum(dim=0).clamp(max=1.0)


class Mixup(Transform):
    """Mixes audio samples within a batch using SNR-based scaling.

    Designed to be applied on a list of sample dicts where each dict contains
    ``audio_key`` as a tensor of shape ``(T,)`` or ``(C, T)``. Samples may have
    different lengths; background audio is zero-padded or randomly cropped to
    match the foreground length before mixing.

    Each sample is independently mixed with a randomly selected other sample
    from the same batch. With probability ``(1-p)`` a sample is left unchanged.

    SNR mixing:
        The background is rescaled so that its RMS is ``snr_db`` decibels
        below the foreground RMS, then added to the foreground.

    Example::

        mixup = Mixup(p=0.5, min_snr_db=0.0, max_snr_db=5.0)
        batch = mixup(batch)  # batch is list[dict]

    Args:
        p: Probability of mixing each sample in the batch.
        min_snr_db: Minimum background SNR relative to foreground (dB).
        max_snr_db: Maximum background SNR relative to foreground (dB).
        audio_key: Key in each sample dict containing the audio tensor.
        label_key: Key for multi-hot label tensors. Pass ``None`` to skip.
        mix_target: ``"union"`` ORs all labels; ``"original"`` keeps the base label.
        training_only: If ``True`` (default), this transform is skipped when
            the module is in eval mode (``.eval()``).
    """

    def __init__(
        self,
        p: float = 0.5,
        min_snr_db: float = 0.0,
        max_snr_db: float = 5.0,
        audio_key: str = "audio",
        label_key: str | None = "label",
        mix_target: Literal["union", "original"] = "union",
    ) -> None:
        super().__init__()
        self.p = p
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.audio_key = audio_key
        self.label_key = label_key
        self.mix_target = mix_target

    def __call__(self, batch: list[dict]) -> list[dict]:
        B = len(batch)
        if B < 2:
            return batch

        mix_mask = torch.rand(B) < self.p
        if not mix_mask.any():
            return batch

        offset = int(torch.randint(1, B, (1,)).item())
        bg_indices = [(i + offset) % B for i in range(B)]

        result = []
        for i, item in enumerate(batch):
            if not mix_mask[i]:
                result.append(item)
                continue

            fg = item[self.audio_key]
            bg_item = batch[bg_indices[i]]
            bg = _fit_to_length(bg_item[self.audio_key], fg.shape[-1])

            snr_db = float(torch.empty(1).uniform_(self.min_snr_db, self.max_snr_db).item())
            rms_fg = fg.pow(2).mean(dim=-1, keepdim=True).sqrt()
            rms_bg = bg.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
            bg_scaled = bg * (rms_fg / rms_bg) / (10 ** (snr_db / 20))

            new_item = {**item, self.audio_key: fg + bg_scaled}
            if self.label_key and self.mix_target == "union":
                new_item[self.label_key] = _union_labels(
                    item[self.label_key], bg_item[self.label_key].unsqueeze(0)
                )
            result.append(new_item)

        return result


class MeanMix(Transform):
    """
    Generalized audio mixup applied to a list of sample dicts before collation.

    Expects each dict to contain ``audio_key`` as a tensor of shape ``(T,)`` or
    ``(C, T)``. Samples may have different lengths; extra components are
    zero-padded or randomly cropped to match the base sample before mixing.

    For each sample:
      - Draw K extra components: K ~ BetaBin(max_samples-1, alpha, beta) + 1
        so K in [1, max_samples] and N = K+1 total components.
      - Sample mixing weights w ~ Dirichlet(omega, ..., omega) of length N.
      - Mix with gain correction: mixed = (sum_i w_i * x_i) / ||w||_2
        Extra components are optionally rolled along the time axis.
      - Targets: "original" keeps base target; "union" takes multi-hot union.

    Example::

        mixup = MeanMix(p=0.5, max_samples=2, mix_target="union")
        batch = mixup(batch)  # batch is list[dict]

    Args:
        p: Probability of applying mixup to each individual sample in the batch.
        mix_target: ``"union"`` ORs all labels; ``"original"`` keeps the base label.
        max_samples: Maximum number of extra components (K in [1, max_samples]).
        alpha: Alpha parameter of the Beta-binomial prior.
        beta: Beta parameter of the Beta-binomial prior.
        omega: Symmetric Dirichlet concentration for mixing weights.
        roll_mixed: If True, randomly roll extra components along the time axis.
        audio_key: Key for the audio tensor in each sample dict.
        label_key: Key for multi-hot label tensors. Pass ``None`` to skip.
        training_only: If ``True`` (default), this transform is skipped when
            the module is in eval mode (``.eval()``).
    """

    def __init__(
        self,
        p: float = 0.5,
        mix_target: Literal["union", "original"] = "union",
        max_samples: int = 1,
        *,
        alpha: float = 2.0,
        beta: float = 2.0,
        omega: float = 1.0,
        roll_mixed: bool = True,
        audio_key: str = "audio",
        label_key: str | None = "label",
    ) -> None:
        super().__init__()
        self.p = p
        self.mix_target = mix_target
        self.max_samples = max_samples
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.roll_mixed = roll_mixed
        self.audio_key = audio_key
        self.label_key = label_key

    def __call__(self, batch: list[dict]) -> list[dict]:
        B = len(batch)
        result = []

        for i, item in enumerate(batch):
            if torch.rand(()) > self.p:
                result.append(item)
                continue

            fg = item[self.audio_key]
            T = fg.shape[-1]

            K = self._sample_beta_binomial(self.max_samples - 1, self.alpha, self.beta) + 1
            candidates = list(range(i)) + list(range(i + 1, B))
            if K >= len(candidates):
                chosen = candidates
            else:
                perm = torch.randperm(len(candidates))[:K].tolist()
                chosen = [candidates[j] for j in perm]

            add_audios = []
            for j in chosen:
                bg = _fit_to_length(batch[j][self.audio_key], T)
                if self.roll_mixed:
                    bg = self._random_roll(bg)
                add_audios.append(bg)
            add_audios = torch.stack(add_audios)

            all_audios = torch.cat([fg.unsqueeze(0), add_audios], dim=0)
            N = all_audios.shape[0]
            w = Dirichlet(torch.full((N,), float(self.omega), device=fg.device)).sample()
            w = w.to(all_audios.dtype)
            w = w / w.pow(2).sum().sqrt().clamp(min=1e-8)
            mixed_audio = torch.tensordot(w, all_audios, dims=1)

            new_item = {**item, self.audio_key: mixed_audio}
            if self.label_key and self.mix_target == "union":
                add_targets = torch.stack([batch[j][self.label_key] for j in chosen])
                new_item[self.label_key] = _union_labels(item[self.label_key], add_targets)
            result.append(new_item)

        return result

    @staticmethod
    def _sample_beta_binomial(n: int, alpha: float, beta: float) -> int:
        if n <= 0:
            return 0
        p = Beta(alpha, beta).sample()
        return int(Binomial(total_count=n, probs=p).sample().item())

    @staticmethod
    def _random_roll(audio: torch.Tensor) -> torch.Tensor:
        shift = int(torch.randint(0, audio.shape[-1], (1,), device=audio.device).item())
        return torch.roll(audio, shifts=shift, dims=-1)
