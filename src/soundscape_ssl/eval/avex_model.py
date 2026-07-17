"""avex `ModelBase` wrapper around our soundscape-MAE ViT encoder.

avex extracts embeddings via forward hooks: ``extract_embeddings`` runs
``forward(wav, mask)`` and reads the output of a hooked layer. We expose the
mean-pooled patch tokens (GAP) through an ``nn.Identity`` hook point (``embed``)
and override ``_discover_embedding_layers`` so that ``target_layers=["embed"]``
resolves to it. GAP is the standard MAE linear-probe readout and beats the CLS
token here; the offline probe caches this 2D vector directly (its own
``aggregation`` is a no-op on an already-pooled input).

The wrapper reproduces our pretraining input pipeline (peak-normalise ->
mel-spectrogram -> pad time to 512) and loads the encoder weights from a Fabric
checkpoint itself (avex's own checkpoint loader cannot parse our
``{"model": {encoder.*, decoder.*}, ...}`` format).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from avex.models.base_model import ModelBase
from avex.models.utils.registry import register_model_class

from soundscape_ssl.data.transforms.spectrogram import BatchSpectrogram
from soundscape_ssl.models import ViTEncoder

logger = logging.getLogger(__name__)


@register_model_class
class SoundscapeMaeModel(ModelBase):
    """Soundscape-MAE ViT encoder adapted to avex's evaluation interface."""

    name = "soundscape_mae"

    def __init__(
        self,
        device: str,
        audio_config: Optional[Any] = None,
        init_config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(device, audio_config)
        cfg = dict(init_config or {})

        encoder_cfg = dict(cfg.get("encoder", {}))
        self.encoder = ViTEncoder(**encoder_cfg)

        self.target_frames = int(cfg.get("target_frames", 512))
        self._spectrogram = BatchSpectrogram(
            sample_rate=int(cfg["sample_rate"]),
            n_fft=int(cfg.get("n_fft", 1024)),
            hop_length=int(cfg.get("hop_length", 320)),
            n_mels=int(cfg.get("n_mels", 128)),
            f_min=float(cfg.get("f_min", 50.0)),
            f_max=cfg.get("f_max", None),
            power=float(cfg.get("power", 2.0)),
            top_db=cfg.get("top_db", 80.0),
        )
        # `BatchSpectrogram` is a plain Transform, not an nn.Module, so register
        # its inner torchaudio module here for `.to(device)` to move the mel
        # filterbank buffers (same object referenced inside `_compute`).
        self.mel_module = self._spectrogram._mel_spectrogram

        self.embed = nn.Identity()  # hook point -> GAP embedding (B, D)

        self._load_ckpt(cfg["checkpoint_path"])
        self.to(device)
        self.eval()

    def _load_ckpt(self, ckpt_path: str) -> None:
        """Load encoder weights from a Fabric checkpoint (strip ``encoder.``)."""
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_sd = state["model"] if "model" in state else state
        encoder_sd = {
            k[len("encoder."):]: v
            for k, v in model_sd.items()
            if k.startswith("encoder.")
        }
        if not encoder_sd:
            raise ValueError(
                f"No 'encoder.'-prefixed weights found in checkpoint {ckpt_path}; "
                "expected a soundscape-MAE / ViTClassifier checkpoint."
            )
        result = self.encoder.load_state_dict(encoder_sd, strict=False)
        logger.info("Loaded encoder from %s: %s", ckpt_path, result)

    def _discover_embedding_layers(self) -> None:
        self._layer_names = ["embed"]

    def forward(
        self,
        wav: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # wav: (B, T) raw waveform at the configured sample rate.
        peak = wav.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        wav = wav / peak
        spec = self._mel_to_db(wav)  # (B, 1, n_mels, time)
        spec = self._pad_time(spec, self.target_frames)
        tokens = self.encoder(spec)  # (B, N+1, D), no masking
        gap = tokens[:, 1:].mean(dim=1)  # mean-pool patch tokens (exclude CLS)
        return self.embed(gap)  # (B, D) -> captured by hook

    def _mel_to_db(self, wav: torch.Tensor) -> torch.Tensor:
        """Batched mel->dB->[-1, 1], matching pretraining's ``BatchSpectrogram``.

        We call the mel module directly instead of ``BatchSpectrogram._compute``
        because that helper squeezes a (1, T) input, corrupting a batch of size 1
        (avex's test loader does not drop the last partial batch).
        """
        top_db = self._spectrogram.top_db
        spec = self.mel_module(wav)  # (B, n_mels, time)
        spec = self._spectrogram._to_db(spec)
        spec = 2 * (spec + top_db) / top_db - 1
        return spec.unsqueeze(1)  # (B, 1, n_mels, time)

    @staticmethod
    def _pad_time(spec: torch.Tensor, target: int) -> torch.Tensor:
        t = spec.shape[-1]
        if t < target:
            return F.pad(spec, (0, target - t))
        if t > target:
            return spec[..., :target]
        return spec
