"""Configuration for the released BirdMAE2 models.

Ships inside the HuggingFace model repo, so it must stay importable with nothing
but ``torch`` and ``transformers`` installed — it deliberately imports nothing
from ``soundscape_ssl``.

The geometry defaults are the ones every released artifact was trained with
(``configs/module/model/backbone/vit.yaml``): a ViT-B/16 over a
(128 mel, 512 frame) input, i.e. an (8, 32) token grid and 257 tokens, with
QK-norm on and a fixed 2-D sincos position table.

The mel front-end is *not* described here. It lives in
``preprocessor_config.json`` next to this config, so the numbers that turn audio
into the model's input have exactly one home.
"""

from typing import Any

from transformers import PretrainedConfig


class BirdMAE2Config(PretrainedConfig):
    """Config for :class:`BirdMAE2Model` and its classification head.

    Args:
        num_mel_bins: Mel bins of the input spectrogram (the height of the
            ``(B, 1, num_mel_bins, num_frames)`` input).
        num_frames: Frames of the input spectrogram.
        patch_size: Square patch side. Both input axes must be divisible by it.
        num_channels: Input channels. The released models are mono, so 1.
        hidden_size: Encoder width.
        num_hidden_layers: Number of transformer blocks.
        num_attention_heads: Attention heads per block.
        mlp_ratio: Block MLP expansion factor.
        qkv_bias: Bias on the fused QKV projection.
        qkv_norm: Apply LayerNorm to queries and keys (QK-norm). ``True`` for
            every model pretrained in this project, unlike the published
            baselines it was compared against.
        layer_norm_eps: Epsilon of every LayerNorm.
        initializer_range: Std of the truncated-normal init used for freshly
            built weights. Irrelevant when loading a released checkpoint.
        num_prototypes: Prototypes per class in the classification head.
        proto_chunk: Chunk size along the prototype axis of the head's cosine
            similarity. Bounds peak activation memory and is exact — top-k
            pooling is independent per prototype. ``None`` computes in one shot,
            which at the released head's 10 799 classes would need a 28.3 GB
            intermediate.
        **kwargs: Forwarded to :class:`~transformers.PretrainedConfig`, which is
            where ``id2label`` / ``label2id`` / ``num_labels`` come from.
    """

    model_type = "birdmae2"

    def __init__(
        self,
        num_mel_bins: int = 128,
        num_frames: int = 512,
        patch_size: int = 16,
        num_channels: int = 1,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qkv_norm: bool = True,
        layer_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        num_prototypes: int = 20,
        proto_chunk: int | None = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qkv_norm = qkv_norm
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.num_prototypes = num_prototypes
        self.proto_chunk = proto_chunk

    @property
    def grid_size(self) -> tuple[int, int]:
        """Token grid as ``(mel patches, frame patches)`` — (8, 32) as released."""
        return (self.num_mel_bins // self.patch_size, self.num_frames // self.patch_size)

    @property
    def num_patches(self) -> int:
        """Patch tokens, excluding the CLS token — 256 as released."""
        grid_h, grid_w = self.grid_size
        return grid_h * grid_w


__all__ = ["BirdMAE2Config"]
