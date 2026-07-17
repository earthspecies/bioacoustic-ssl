from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ViTEncoder


def _pool(tokens: torch.Tensor, pool: str | None) -> torch.Tensor:
    if pool is None:
        return tokens
    elif pool == "cls":
        return tokens[:, 0]
    elif pool == "gap":
        return tokens[:, 1:].mean(dim=1)
    elif pool == "cls_gap":
        cls = tokens[:, 0]
        gap = tokens[:, 1:].mean(dim=1)
        return torch.cat([cls, gap], dim=-1)
    else:
        raise ValueError(
            f"Unknown pool strategy '{pool}'. "
            "Choose from: None, 'cls', 'gap', 'cls_gap'."
        )


class ViTClassifier(ViTEncoder):
    """ViT encoder with a linear classification head.

    Inherits all parameters and weights of :class:`ViTEncoder`, then adds a
    classification head on top of the pooled token representation.

    Checkpoint compatibility:
        Pretrained MAE weights can be loaded with the same snippet used in
        ``birdset_eval.py`` — no changes required::

            state_dict = fabric.load(path)
            # Strip the "encoder." prefix that MAE wraps around the encoder keys.
            model_sd = {
                k[len("encoder."):]: v
                for k, v in state_dict["model"].items()
                if k.startswith("encoder.")
            }
            model.load_state_dict(model_sd, strict=False)
            # strict=False silently ignores the missing head.* keys.

    Args:
        num_classes: Number of output classes (logits dimension).
        pool: Token pooling strategy fed to the head.
              ``"cls"`` — CLS token (default, standard ViT).
              ``"gap"`` — global average of patch tokens.
              ``"cls_gap"`` — concatenation of CLS and GAP (doubles head input dim).
        head_drop: Dropout applied before the linear head (default: 0.0).
        **encoder_kwargs: All keyword arguments forwarded to :class:`ViTEncoder`.
    """

    def __init__(
        self,
        num_classes: int,
        pool: str | None = "cls",
        head_drop: float = 0.0,
        **encoder_kwargs,
    ) -> None:
        super().__init__(**encoder_kwargs)

        embed_dim: int = encoder_kwargs.get("embed_dim", 768)
        head_in_dim = embed_dim * (2 if pool == "cls_gap" else 1)

        self.pool = pool
        self.head_drop = nn.Dropout(head_drop)
        self.head = nn.Linear(head_in_dim, num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze (or unfreeze) all parameters except the classification head.

        Useful for linear probing: call once after loading pretrained weights
        and before constructing the optimizer so that only ``head`` parameters
        appear in the parameter groups.

        Args:
            freeze: ``True`` (default) to freeze the backbone; ``False`` to
                    unfreeze it (e.g., to switch from linear probing to full
                    fine-tuning).
        """
        for name, param in self.named_parameters():
            if not name.startswith("head"):
                param.requires_grad_(not freeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass: spectrogram → class logits.

        Args:
            x: Input spectrogram, shape ``(B, C, H, W)``.

        Returns:
            Class logits, shape ``(B, num_classes)``.
        """
        tokens: torch.Tensor = super().forward(x)  # (B, N+1, D)
        feat = _pool(tokens, self.pool)             # (B, D) or (B, 2D)
        feat = self.head_drop(feat)
        return self.head(feat)                      # (B, num_classes)


class ViTProtoFloat(ViTEncoder):
    """ViT encoder with a linear classification head.

    Inherits all parameters and weights of :class:`ViTEncoder`, then adds a
    classification head on top of the pooled token representation.

    Checkpoint compatibility:
        Pretrained MAE weights can be loaded with the same snippet used in
        ``birdset_eval.py`` — no changes required::

            state_dict = fabric.load(path)
            # Strip the "encoder." prefix that MAE wraps around the encoder keys.
            model_sd = {
                k[len("encoder."):]: v
                for k, v in state_dict["model"].items()
                if k.startswith("encoder.")
            }
            model.load_state_dict(model_sd, strict=False)
            # strict=False silently ignores the missing head.* keys.

    Args:
        num_classes: Number of output classes (logits dimension).
        pool: Token pooling strategy fed to the head.
              ``"cls"`` — CLS token (default, standard ViT).
              ``"gap"`` — global average of patch tokens.
              ``"cls_gap"`` — concatenation of CLS and GAP (doubles head input dim).
        head_drop: Dropout applied before the linear head (default: 0.0).
        **encoder_kwargs: All keyword arguments forwarded to :class:`ViTEncoder`.
    """

    def __init__(
        self,
        num_classes: int,
        head_drop: float = 0.0,
        num_prototypes: int = 20,
        **encoder_kwargs,
    ) -> None:
        super().__init__(**encoder_kwargs)

        self.embed_dim: int = encoder_kwargs.get("embed_dim", 768)

        self.head_drop = nn.Dropout(head_drop)
        self.head = PrototypicalFloat(self.embed_dim, num_prototypes, num_classes)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze (or unfreeze) all parameters except the classification head.

        Useful for linear probing: call once after loading pretrained weights
        and before constructing the optimizer so that only ``head`` parameters
        appear in the parameter groups.

        Args:
            freeze: ``True`` (default) to freeze the backbone; ``False`` to
                    unfreeze it (e.g., to switch from linear probing to full
                    fine-tuning).
        """
        for name, param in self.named_parameters():
            if not name.startswith("head"):
                param.requires_grad_(not freeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass: spectrogram → class logits.

        Args:
            x: Input spectrogram, shape ``(B, C, H, W)``.

        Returns:
            Class logits, shape ``(B, num_classes)``.
        """
        tokens: torch.Tensor = super().forward(x)  # (B, N+1, D)
        #feat = _pool(tokens, self.pool)
        B = tokens.shape[0]

        cls_features = tokens[:, 0]
        patch_features = tokens[:, 1:]

        z_f = patch_features - cls_features.unsqueeze(1)
        feat = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, *self.patch_embed.grid_size)
        feat = self.head_drop(feat)
        return self.head(feat)                      # (B, num_classes)


class ViTProtoLayerwise(ViTProtoFloat):
    """ViT encoder with a prototypical head over a softmax-weighted sum of all
    transformer-layer hidden states (SUPERB-style layerwise probing).

    Motivation:
        In an MAE the final encoder layers specialise for reconstruction, so the
        most linearly-separable / transferable features often live in a middle
        layer rather than the last one. Instead of probing a single fixed layer,
        this head learns one scalar weight per block, softmax-normalises them,
        and combines every block's token sequence into a single representation
        that is then fed to the same :class:`PrototypicalFloat` head as
        :class:`ViTProtoFloat`. The learned weight vector also doubles as a
        readout of *where* the useful features sit.

    The encoder stays frozen during probing; only the per-layer weights, the
    optional per-layer norms, and the prototype head are trained. The learned
    ``layer_weights`` (after softmax) can be inspected post-hoc to see which
    layers the probe relied on.

    Args:
        num_classes: Number of output classes (logits dimension).
        head_drop: Dropout applied before the prototype head (default: 0.0).
        num_prototypes: Prototypes per class for the :class:`PrototypicalFloat`
            head (default: 20).
        layer_norm: If ``True`` (default), apply an independent LayerNorm to each
            block's output before the weighted sum, so layers with different
            activation scales are comparable. Set ``False`` to weight the raw
            block outputs directly.
        **encoder_kwargs: All keyword arguments forwarded to :class:`ViTEncoder`.
    """

    def __init__(
        self,
        num_classes: int,
        head_drop: float = 0.0,
        num_prototypes: int = 20,
        layer_norm: bool = True,
        **encoder_kwargs,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            head_drop=head_drop,
            num_prototypes=num_prototypes,
            **encoder_kwargs,
        )

        self.num_layers: int = encoder_kwargs.get("depth", 12)
        # One learnable scalar per block, softmax-normalised at forward time.
        # Init at zeros → uniform softmax → equal-weight average to start.
        self.layer_weights = nn.Parameter(torch.zeros(self.num_layers))

        if layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(self.embed_dim, eps=encoder_kwargs.get("norm_eps", 1e-6))
                for _ in range(self.num_layers)
            ])
        else:
            self.layer_norms = None

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze the encoder, keeping the head AND the layerwise-fusion params
        (``layer_weights`` / ``layer_norms``) trainable.

        Overrides :meth:`ViTProtoFloat.freeze_backbone`, whose ``head``-only
        rule would otherwise freeze the layer-fusion parameters too.
        """
        trainable = ("head", "layer_weights", "layer_norms")
        for name, param in self.named_parameters():
            if not name.startswith(trainable):
                param.requires_grad_(not freeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass: spectrogram → class logits via a learned weighted sum
        of every transformer block's output.

        Args:
            x: Input spectrogram, shape ``(B, C, H, W)``.

        Returns:
            Class logits, shape ``(B, num_classes)``.
        """
        # Per-block hidden states, each (B, N+1, D), before the encoder's norm.
        hidden: list[torch.Tensor] = ViTEncoder.forward(self, x, return_hidden=True)
        B = hidden[0].shape[0]

        if self.layer_norms is not None:
            hidden = [ln(h) for ln, h in zip(self.layer_norms, hidden)]

        stacked = torch.stack(hidden, dim=0)          # (L, B, N+1, D)
        w = torch.softmax(self.layer_weights, dim=0)  # (L,)
        fused = (stacked * w.view(-1, 1, 1, 1)).sum(dim=0)  # (B, N+1, D)

        cls_features = fused[:, 0]
        patch_features = fused[:, 1:]

        z_f = patch_features - cls_features.unsqueeze(1)
        feat = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, *self.patch_embed.grid_size)
        feat = self.head_drop(feat)
        return self.head(feat)                        # (B, num_classes)


class PrototypicalFloat(nn.Module):  # protofloat
    def __init__(
        self,
        dim: int,
        num_prototypes: int,
        num_classes: int,
        topk_k: int = 1
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes
        self.num_prototypes_total = num_prototypes * num_classes
        self.topk_k = topk_k

        # 1x1 convolutional kernels
        self.prototype_vectors = nn.Parameter(torch.randn(
            self.num_prototypes_total, dim, 1, 1) * 0.02)

        self.linear = nn.Linear(self.num_prototypes_total, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2: #for cls-input
            x = x[:, :, None, None]

        # standard cosine similarity (activation) - NO BINARIZATION
        x_norm = F.normalize(x, dim=1)
        p_norm = F.normalize(self.prototype_vectors, dim=1)  # directly normalize, no _binarise()
        act = F.conv2d(x_norm, p_norm)              # (B, P, H, W)

        # pooling & classification as before
        B, P, _, _ = act.shape
        act = act.view(B, P, -1)
        k = self.topk_k if self.training else 1

        # top-k pooling: taking the mean of the k strongest hits
        pooled = act.topk(k, dim=-1).values.mean(-1)
        return self.linear(pooled)
