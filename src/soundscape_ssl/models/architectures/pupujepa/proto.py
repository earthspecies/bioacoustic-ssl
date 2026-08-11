"""Probing/finetuning heads on top of the vendored PupuJEPA encoder.

Ours, not vendored — kept out of ``pupujepa.py`` / ``patch_embed_rope.py`` so
those files stay byte-identical to, and diff-able against, the upstream repo
(github.com/sizigi/PupuJEPA).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from timm.layers import create_rope_embed

from ..vit.classifier import PrototypicalFloat
from .patch_embed_rope import PatchEmbedRoPE
from .pupujepa import PupuJEPAEncoder


class PupuJepaBackbone(nn.Module):
    """Vendored PupuJEPA encoder, loaded from its checkpoint and ready for a head.

    PupuJEPA does not load into :class:`ViTEncoder` (RoPE instead of a
    ``pos_embed``, SwiGLU MLPs, no CLS token), so this is a thin wrapper around
    the vendored ``PupuJEPAEncoder`` rather than another ``ViTEncoder`` subclass.
    The head-bearing arms — :class:`PupuJepaProtoFloat`,
    :class:`PupuJepaProtoLayerwise`, :class:`PupuJepaClassifier` — subclass this
    and add their own head; the backbone, its state-dict keys and its layer-decay
    grouping are identical across all three.

    Three differences from the other eval arms are worth knowing:

    * **RoPE carries no parameters** and is generated from the grid of whatever
      input arrives (``rope_encoder.get_embed(grid)``), so any clip length works
      without interpolating or slicing a positional table. At our 5 s / 24 kHz
      protocol the grid is ``(125, 8)`` = 1000 tokens against 257 for the
      ViT-B/16 arms, because the patch is 4 frames wide rather than 16 — expect
      roughly 4-6x the probing cost per step.
    * **There is no CLS token** (``num_prefix_tokens=0``), so whatever needs a
      single global reference uses the mean over patch tokens instead: the proto
      arms center by it, :class:`PupuJepaClassifier` pools by it. That mean *is*
      the clip embedding PupuJEPA's own ``embed.py`` publishes, so it is the
      model's notion of a global token — but where the other arms read a CLS
      token this is a substitute, and it is the one part of the cross-arm
      comparison that is not like-for-like.
    * **The probed weights are the EMA target encoder** (``teacher.*`` in the
      checkpoint), which is what upstream deploys at inference. The student and
      predictor are never built, so this arm holds ~113 M params rather than the
      full ~1 GB checkpoint.

    The encoder submodule **must** be called ``encoder``: ``param_groups_lrd``
    branches on ``hasattr(model, "encoder")`` to find the block count, and
    ``get_layer_id_for_vit`` strips that prefix so ``encoder.blocks.N.*`` lands
    in layer group ``N + 1`` while the top-level ``patch_embed.*`` lands in
    group 0. ``rope_encoder`` holds no parameters, so it needs no group.

    Weights arrive at construction with ``strict=True``, so this arm needs NO
    ``trainer.resume_from_checkpoint`` — set it to null, or the eval script will
    try to load an unrelated encoder over the top of it.

    Args:
        ckpt_path: Local path to PupuJEPA's ``model.safetensors``.
        img_size: Nominal input size, only used for ``patch_embed.grid_size``.
            The forward pass derives the grid from the actual input, so this does
            not constrain the clip length.
        patch_size: ``(frames, n_mels)`` patch. Fixed by the checkpoint.
        in_chans, embed_dim, depth, num_heads, mlp_ratio, use_swiglu, qk_norm,
            drop_path_rate: architecture, all fixed by the checkpoint except
            ``drop_path_rate`` (which upstream pretrained at 0.0).
    """

    def __init__(
        self,
        ckpt_path: str,
        img_size: tuple[int, int] = (500, 128),
        patch_size: tuple[int, int] = (4, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_swiglu: bool = True,
        qk_norm: bool = True,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.patch_size = tuple(patch_size)
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedRoPE(
            img_size=tuple(img_size),
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
            flatten=True,
            frequency_first=False,
        )
        self.encoder = PupuJEPAEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            drop_path_rate=drop_path_rate,
            drop_path_uniform=False,
            use_swiglu=use_swiglu,
            init_values=None,
            num_prefix_tokens=0,
            qk_norm=qk_norm,
        )
        self.rope_encoder = create_rope_embed(
            rope_type="cat",
            dim=embed_dim,
            num_heads=num_heads,
            feat_shape=None,
        )
        self._load_pretrained(ckpt_path)

    def _load_pretrained(self, ckpt_path: str) -> None:
        """Load ``patch_embed.*`` and the EMA ``teacher.*`` encoder, strictly.

        The checkpoint holds ``patch_embed`` / ``student`` / ``teacher`` /
        ``predictor``; only the first two are built here, so loading is done per
        submodule with ``strict=True`` rather than with a permissive
        ``strict=False`` over the whole thing — a renamed or reshaped backbone
        key must fail loudly, not silently leave a block at its init values.
        """
        state_dict = load_file(Path(ckpt_path))
        self.patch_embed.load_state_dict(
            {k.removeprefix("patch_embed."): v for k, v in state_dict.items() if k.startswith("patch_embed.")},
            strict=True,
        )
        self.encoder.load_state_dict(
            {k.removeprefix("teacher."): v for k, v in state_dict.items() if k.startswith("teacher.")},
            strict=True,
        )

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze (or unfreeze) everything except the head.

        Mirrors :meth:`ViTProtoFloat.freeze_backbone`, including the rule that
        the split is by the ``head`` name prefix, so ``param_groups_lrd`` sees
        only head params when probing.

        Args:
            freeze: ``True`` (default) to freeze the backbone; ``False`` for
                    full finetuning.
        """
        for name, param in self.named_parameters():
            if not name.startswith("head"):
                param.requires_grad_(not freeze)

    def _patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Patch-embed ``x`` and build the matching RoPE embedding.

        Args:
            x: Input log-mel, shape ``(B, 1, frames, n_mels)``.

        Returns:
            ``(tokens, rope, grid)`` — tokens ``(B, N, D)`` flattened
            frames-major, ``rope`` broadcast to ``(B, 1, N, head_dim)`` as
            upstream's inference path does, and ``grid`` as
            ``(frame_patches, mel_patches)``.
        """
        tokens = self.patch_embed(x)
        grid = (x.shape[-2] // self.patch_size[0], x.shape[-1] // self.patch_size[1])
        rope = self.rope_encoder.get_embed(grid)  # (N, head_dim), no parameters
        rope = rope[None, None].expand(tokens.shape[0], 1, -1, -1)
        return tokens, rope, grid


class PupuJepaProtoFloat(PupuJepaBackbone):
    """PupuJEPA backbone with the same :class:`PrototypicalFloat` head every proto arm uses.

    Args:
        num_classes: Number of output classes (logits dimension).
        num_prototypes: Prototypes per class for the head.
        head_drop: Dropout applied to the focal-similarity grid before the head.
        **backbone_kwargs: Forwarded to :class:`PupuJepaBackbone` (``ckpt_path``
            and the architecture, all read from ``configs/module/model``).
    """

    def __init__(
        self,
        num_classes: int,
        num_prototypes: int = 20,
        head_drop: float = 0.0,
        **backbone_kwargs: object,
    ) -> None:
        super().__init__(**backbone_kwargs)  # type: ignore[arg-type]

        self.head_drop = nn.Dropout(head_drop)
        self.head = PrototypicalFloat(self.embed_dim, num_prototypes, num_classes)

    def _apply_head(self, tokens: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Focal similarity against the mean patch token, then the proto head.

        Args:
            tokens: Encoder output, shape ``(B, N, D)``, flattened frames-major
                (``PatchEmbedRoPE`` with ``frequency_first=False`` permutes to
                ``(B, gh, gw, D)`` before flattening, so index ``= h * gw + w``).
            grid: ``(frame_patches, mel_patches)`` the tokens came from.

        Returns:
            Class logits, shape ``(B, num_classes)``.
        """
        z_f = tokens - tokens.mean(dim=1, keepdim=True)
        feat = z_f.permute(0, 2, 1).reshape(tokens.shape[0], self.embed_dim, *grid)
        return self.head(self.head_drop(feat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: log-mel features → class logits.

        Args:
            x: Input log-mel, shape ``(B, 1, frames, n_mels)`` — the transpose of
                this repo's house layout, produced by ``TransposeSpec`` as the
                last stage of ``configs/data/transforms/pupujepa.yaml``.

        Returns:
            Class logits, shape ``(B, num_classes)``.
        """
        tokens, rope, grid = self._patchify(x)
        tokens = self.encoder(tokens, rope=rope)  # upstream forward: blocks + final norm
        return self._apply_head(tokens, grid)


class PupuJepaProtoLayerwise(PupuJepaProtoFloat):
    """PupuJEPA with the prototypical head over a softmax-weighted sum of all
    block outputs (SUPERB-style layerwise probing).

    The layerwise counterpart of :class:`PupuJepaProtoFloat`, and the direct
    analogue of :class:`ViTProtoLayerwise`: one learnable scalar per block,
    softmax-normalised, combining every block's token sequence into a single
    representation fed to the same head. ``birdset_eval.py`` logs the softmax
    profile automatically — it looks for a ``layer_weights`` attribute.

    Like ``ViTProtoLayerwise`` this weights the block outputs *before* the
    encoder's final ``norm``, which is consequently unused on this path.

    Args:
        layer_norm: If ``True`` (default), apply an independent LayerNorm to each
            block's output before the weighted sum, so layers with different
            activation scales are comparable.
        **kwargs: Forwarded to :class:`PupuJepaProtoFloat`.
    """

    def __init__(self, layer_norm: bool = True, **kwargs: object) -> None:
        super().__init__(**kwargs)

        self.num_layers = len(self.encoder.blocks)
        # One learnable scalar per block, softmax-normalised at forward time.
        # Init at zeros -> uniform softmax -> equal-weight average to start.
        self.layer_weights = nn.Parameter(torch.zeros(self.num_layers))

        if layer_norm:
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(self.embed_dim) for _ in range(self.num_layers)]
            )
        else:
            self.layer_norms = None

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze the encoder, keeping the head AND the layerwise-fusion params
        (``layer_weights`` / ``layer_norms``) trainable.

        Overrides :meth:`PupuJepaProtoFloat.freeze_backbone`, whose ``head``-only
        rule would otherwise freeze the layer-fusion parameters too.
        """
        trainable = ("head", "layer_weights", "layer_norms")
        for name, param in self.named_parameters():
            if not name.startswith(trainable):
                param.requires_grad_(not freeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: log-mel features → class logits via a learned weighted
        sum of every block's output.

        Args:
            x: Input log-mel, shape ``(B, 1, frames, n_mels)``.

        Returns:
            Class logits, shape ``(B, num_classes)``.
        """
        tokens, rope, grid = self._patchify(x)

        # Block loop mirrors ``PupuJEPAEncoder.forward`` (minus its final norm),
        # inlined because upstream does not expose per-block hidden states and
        # ``pupujepa.py`` is kept byte-identical to upstream.
        w = torch.softmax(self.layer_weights, dim=0)  # (L,)
        fused = None
        for i, blk in enumerate(self.encoder.blocks):
            tokens = blk(tokens, rope=rope)
            h = tokens if self.layer_norms is None else self.layer_norms[i](tokens)
            fused = w[i] * h if fused is None else fused + w[i] * h

        return self._apply_head(fused, grid)


class PupuJepaClassifier(PupuJepaBackbone):
    """PupuJEPA with a single ``nn.Linear`` on the mean-pooled patch tokens.

    The linear-probing counterpart of :class:`PupuJepaProtoFloat`, and the
    PupuJEPA analogue of :class:`ViTClassifier` with ``pool="gap"``: mean over
    the encoder's (already ``norm``-ed) patch tokens, dropout, linear.

    Mean pooling is not an arbitrary choice here. PupuJEPA has no CLS token, and
    the mean over patch tokens is exactly the clip embedding upstream's own
    ``embed.py`` publishes, so this probes the representation the model itself
    deploys at inference.

    The head params are ``head.weight`` / ``head.bias``, which
    ``param_groups_lrd``'s prototype-head rule does not match: with
    ``prototype_lr: null`` (as in ``experiment=sweeps/linear/birdset/*``)
    they land in the last layer-decay group at scale 1.0, i.e. the head LR is
    exactly ``base_lr * batch_size / 256``, same as the ``linear/vit`` arm.

    Args:
        num_classes: Number of output classes (logits dimension).
        head_drop: Dropout applied to the pooled embedding before the head.
        **backbone_kwargs: Forwarded to :class:`PupuJepaBackbone`.
    """

    def __init__(
        self,
        num_classes: int,
        head_drop: float = 0.0,
        **backbone_kwargs: object,
    ) -> None:
        super().__init__(**backbone_kwargs)  # type: ignore[arg-type]

        self.head_drop = nn.Dropout(head_drop)
        self.head = nn.Linear(self.embed_dim, num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: log-mel features → class logits.

        Args:
            x: Input log-mel, shape ``(B, 1, frames, n_mels)``.

        Returns:
            Class logits, shape ``(B, num_classes)``.
        """
        tokens, rope, _ = self._patchify(x)
        tokens = self.encoder(tokens, rope=rope)  # upstream forward: blocks + final norm
        feat = tokens.mean(dim=1)                 # (B, D) — no CLS token to read
        return self.head(self.head_drop(feat))
