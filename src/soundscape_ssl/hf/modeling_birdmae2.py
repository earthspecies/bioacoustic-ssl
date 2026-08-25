"""The released BirdMAE2 encoder and its Xeno-Canto classification head.

Ships inside the HuggingFace model repo, so it imports nothing from
``soundscape_ssl``: ``torch`` and ``transformers`` are the whole dependency set.
Two entry points:

``BirdMAE2Model`` (``AutoModel``)
    The pretrained ViT-B/16 encoder. Takes a mel spectrogram, returns token
    embeddings — 257 tokens of 768 dims, CLS first — plus the CLS token as
    ``pooler_output``.

``BirdMAE2ForAudioClassification`` (``AutoModelForAudioClassification``)
    The same encoder, frozen during training, under a learned softmax fusion of
    all 12 block outputs and a prototypical head over 10 799 Xeno-Canto taxa.
    ``config.id2label`` carries the taxon names; the ``gbifID`` behind each
    output index — the contract you need to restrict the logits to a species
    subset — ships beside the weights as ``xc_classes.parquet``.

Every module here is a transcription of the training-time implementation
(``soundscape_ssl.models.architectures.vit``), op for op and name for name, so
that loading a converted checkpoint is a key rename rather than a
re-implementation that happens to agree. ``tests/unittests/test_hf_birdmae2.py``
asserts the two agree bit-exactly, and ``scripts/export_hf_model.py`` re-asserts
it against the real checkpoint at export time. Keep it that way: an
optimisation here that changes a single floating-point op invalidates the
published numbers.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput

from .configuration_birdmae2 import BirdMAE2Config


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """Sinusoidal embedding of one coordinate axis.

    Args:
        embed_dim: Output width. Must be even.
        pos: Positions of any shape; flattened before use.

    Returns:
        Tensor of shape ``(pos.numel(), embed_dim)``.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size_h: int, grid_size_w: int, cls_token: bool = False
) -> torch.Tensor:
    """Fixed 2-D sincos position table for a ``(grid_size_h, grid_size_w)`` grid.

    The axis order matters and is not the obvious one: positions are built with
    ``indexing="xy"`` over ``(w, h)``, which is what the released weights were
    trained with. A released checkpoint carries its own ``pos_embed`` tensor and
    overwrites this table on load, so the two can only disagree for a
    freshly-initialised model — but they must not.

    Args:
        embed_dim: Output width.
        grid_size_h: Grid height, in patches.
        grid_size_w: Grid width, in patches.
        cls_token: Prepend a zero row for the CLS token.

    Returns:
        Tensor of shape ``(grid_size_h * grid_size_w [+ 1], embed_dim)``.
    """
    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing="xy"), dim=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = torch.cat([emb_h, emb_w], dim=1)

    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed


class BirdMAE2PatchEmbed(nn.Module):
    """Non-overlapping patch projection of a mel spectrogram to token embeddings."""

    def __init__(self, config: BirdMAE2Config) -> None:
        """Build the patch projection from ``config``'s geometry."""
        super().__init__()
        self.grid_size = config.grid_size
        self.num_patches = config.num_patches
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``(B, C, mels, frames)`` to ``(B, num_patches, hidden_size)``.

        Tokens are flattened frame-major within each mel row, which is the order
        the position table above was built for.
        """
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class BirdMAE2Attention(nn.Module):
    """Multi-head self-attention with optional QK-norm, via scaled dot product."""

    def __init__(self, config: BirdMAE2Config) -> None:
        """Build the fused QKV projection, the output projection and QK norms."""
        super().__init__()
        self.num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_norm = nn.LayerNorm(head_dim) if config.qkv_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(head_dim) if config.qkv_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over the token axis of ``(B, N, hidden_size)``."""
        batch, tokens, width = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, width // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        x = (
            F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, scale=self.scale)
            .transpose(1, 2)
            .reshape(batch, tokens, width)
        )
        return self.proj(x)


class BirdMAE2MLP(nn.Module):
    """The per-block feed-forward network."""

    def __init__(self, config: BirdMAE2Config) -> None:
        """Build the two-layer GELU MLP at ``config.mlp_ratio`` expansion."""
        super().__init__()
        hidden = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP token-wise."""
        return self.fc2(self.act(self.fc1(x)))


class BirdMAE2Layer(nn.Module):
    """One pre-norm transformer block."""

    def __init__(self, config: BirdMAE2Config) -> None:
        """Build the block's two norms, attention and MLP."""
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = BirdMAE2Attention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = BirdMAE2MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual attention followed by residual MLP."""
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class BirdMAE2PreTrainedModel(PreTrainedModel):
    """Shared HuggingFace plumbing: config class, weight init, input name."""

    config_class = BirdMAE2Config
    config: BirdMAE2Config
    base_model_prefix = "encoder"
    main_input_name = "input_values"
    _no_split_modules = ["BirdMAE2Layer"]

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise a freshly built module, matching the training-time init."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear | nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, BirdMAE2Model):
            nn.init.trunc_normal_(module.cls_token, std=std)


class BirdMAE2Model(BirdMAE2PreTrainedModel):
    """The pretrained encoder: mel spectrogram in, token embeddings out.

    The input is a ``(batch, 1, num_mel_bins, num_frames)`` spectrogram, which
    :class:`BirdMAE2FeatureExtractor` produces from audio. Feeding anything
    else — a differently-scaled mel, a transposed one, a different frame count —
    is silently wrong rather than an error, so use the shipped extractor.

    Keep inference in float32. These features are strongly anisotropic — MAE
    pretraining offers no reason for them not to be — and running the encoder
    under bf16 autocast was enough, in this project's own evaluation, to take a
    cosine-kNN probe on them down to chance. Loading in a reduced dtype is
    supported and is your risk to measure.
    """

    # So that loading the classification artifact as a bare encoder — a supported
    # thing to do, it carries its own copy — does not warn about the head.
    _keys_to_ignore_on_load_unexpected = [r"^head\.", r"^layer_weights", r"^layer_norms\."]

    def __init__(self, config: BirdMAE2Config) -> None:
        """Build the patch embedding, the position table and the block stack."""
        super().__init__(config)
        self.patch_embed = BirdMAE2PatchEmbed(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.register_buffer(
            "pos_embed",
            get_2d_sincos_pos_embed(config.hidden_size, *config.grid_size, cls_token=True).unsqueeze(0),
        )
        self.blocks = nn.ModuleList(
            [BirdMAE2Layer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def forward(
        self,
        input_values: torch.Tensor,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutputWithPooling:
        """Embed a batch of spectrograms.

        Args:
            input_values: ``(batch, 1, num_mel_bins, num_frames)`` spectrogram,
                as returned by :class:`BirdMAE2FeatureExtractor`.
            output_hidden_states: Also return every block's output. These are the
                *raw* block outputs, before the final shared norm, because that
                norm is specialised for the last layer — which is why the
                classification head normalises each block on its own instead.

        Returns:
            :class:`~transformers.modeling_outputs.BaseModelOutputWithPooling`
            whose ``last_hidden_state`` is ``(batch, 1 + num_patches,
            hidden_size)`` with the CLS token first, ``pooler_output`` is that
            CLS token, and ``hidden_states`` — when asked for — is the patch
            embedding followed by each block's output.
        """
        x = self.patch_embed(input_values)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(input_values.shape[0], -1, -1), x], dim=1)

        hidden_states = (x,) if output_hidden_states else None
        for block in self.blocks:
            x = block(x)
            if output_hidden_states:
                hidden_states = hidden_states + (x,)

        x = self.norm(x)
        return BaseModelOutputWithPooling(
            last_hidden_state=x,
            pooler_output=x[:, 0],
            hidden_states=hidden_states,
        )


class BirdMAE2PrototypicalHead(nn.Module):
    """Cosine-prototype classifier over a spatial feature map.

    Each class owns ``config.num_prototypes`` prototype vectors and reads only
    its own (the standard PPNet block-diagonal formulation). The dense mixer the
    per-task probes in the paper used — every class reading every prototype — is
    quadratic in the class count and would be 2.332 **billion** parameters at
    this head's 10 799 classes, so it is not an option here and is not offered.

    Each prototype's activation is the strongest cosine similarity anywhere on
    the feature map; the class logit is a learned weighted sum of its own
    prototypes' activations.
    """

    def __init__(self, config: BirdMAE2Config) -> None:
        """Build the prototype bank and the per-class mixing weights."""
        super().__init__()
        self.num_classes = config.num_labels
        self.num_prototypes_per_class = config.num_prototypes
        self.num_prototypes_total = config.num_prototypes * config.num_labels
        self.proto_chunk = config.proto_chunk

        self.prototype_vectors = nn.Parameter(
            torch.randn(self.num_prototypes_total, config.hidden_size, 1, 1) * 0.02
        )
        bound = config.num_prototypes**-0.5
        self.class_weight = nn.Parameter(
            torch.empty(config.num_labels, config.num_prototypes).uniform_(-bound, bound)
        )
        self.class_bias = nn.Parameter(torch.empty(config.num_labels).uniform_(-bound, bound))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a ``(batch, hidden_size, grid_h, grid_w)`` feature map.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        x_norm = F.normalize(x, dim=1)
        batch = x_norm.shape[0]
        chunk = self.proto_chunk

        if chunk is None or chunk >= self.num_prototypes_total:
            p_norm = F.normalize(self.prototype_vectors, dim=1)
            act = F.conv2d(x_norm, p_norm)
            pooled = act.view(batch, act.shape[1], -1).topk(1, dim=-1).values.mean(-1)
        else:
            # Same value, one slice of the prototype axis at a time, so the
            # (batch, prototypes, grid_h, grid_w) similarity tensor is never
            # materialised in full — 28.3 GB at 10 799 classes and batch 256.
            # Exact: cosine similarity and top-k-over-space are both independent
            # per prototype.
            pooled = torch.cat(
                [
                    F.conv2d(x_norm, F.normalize(self.prototype_vectors[start : start + chunk], dim=1))
                    .view(batch, -1, x_norm.shape[-2] * x_norm.shape[-1])
                    .topk(1, dim=-1)
                    .values.mean(-1)
                    for start in range(0, self.num_prototypes_total, chunk)
                ],
                dim=1,
            )

        pooled = pooled.view(batch, self.num_classes, self.num_prototypes_per_class)
        return (pooled * self.class_weight).sum(-1) + self.class_bias


class BirdMAE2ForAudioClassification(BirdMAE2PreTrainedModel):
    """Encoder plus the layerwise-fused prototypical head.

    The head reads a learned softmax-weighted sum of all 12 block outputs rather
    than the last one: in an MAE the final blocks specialise for reconstruction,
    so the most transferable features sit further down. ``layer_weights`` after a
    softmax is that distribution, and is worth inspecting — its centroid tracks
    how well the pretraining domain matches the task.

    The released head is multi-label: a Xeno-Canto recording carries a
    foreground species and any number of background ones, so the logits are
    independent per class and ``labels`` are scored with binary cross-entropy.
    """

    def __init__(self, config: BirdMAE2Config) -> None:
        """Build the encoder, the per-block norms, the fusion weights and the head."""
        super().__init__(config)
        self.encoder = BirdMAE2Model(config)
        self.layer_weights = nn.Parameter(torch.zeros(config.num_hidden_layers))
        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.head = BirdMAE2PrototypicalHead(config)
        self.post_init()

    def forward(
        self,
        input_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Classify a batch of spectrograms.

        Args:
            input_values: ``(batch, 1, num_mel_bins, num_frames)`` spectrogram.
            labels: Optional multi-hot targets of shape ``(batch, num_labels)``,
                scored with binary cross-entropy.
            output_hidden_states: Also return the encoder's per-block outputs.

        Returns:
            :class:`~transformers.modeling_outputs.SequenceClassifierOutput` with
            ``logits`` of shape ``(batch, num_labels)``.
        """
        outputs = self.encoder(input_values, output_hidden_states=True)
        hidden = [
            norm(state) for norm, state in zip(self.layer_norms, outputs.hidden_states[1:], strict=True)
        ]

        weights = torch.softmax(self.layer_weights, dim=0)
        # Accumulated rather than stacked: a materialised (layers, batch, tokens,
        # hidden) tensor and its broadcast product are the peak allocation of the
        # whole forward pass otherwise.
        fused = weights[0] * hidden[0]
        for layer in range(1, len(hidden)):
            fused = fused + weights[layer] * hidden[layer]

        # The head scores patches by how far they stand out from the clip's own
        # CLS summary, laid back out on the token grid.
        features = fused[:, 1:] - fused[:, :1]
        features = features.permute(0, 2, 1).reshape(
            fused.shape[0], self.config.hidden_size, *self.config.grid_size
        )
        logits = self.head(features)

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
        )


__all__ = [
    "BirdMAE2ForAudioClassification",
    "BirdMAE2Model",
    "BirdMAE2PreTrainedModel",
]
