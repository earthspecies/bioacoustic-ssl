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
        mixer: str = "dense",
        proto_chunk: int | None = None,
        **encoder_kwargs,
    ) -> None:
        super().__init__(**encoder_kwargs)

        self.embed_dim: int = encoder_kwargs.get("embed_dim", 768)

        self.head_drop = nn.Dropout(head_drop)
        self.head = PrototypicalFloat(
            self.embed_dim, num_prototypes, num_classes, mixer=mixer,
            proto_chunk=proto_chunk)

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
        mixer: How pooled prototype activations become logits — ``"dense"``
            (default, every class sees every prototype) or
            ``"block_diagonal"`` (each class sees only its own prototypes).
            See :class:`PrototypicalFloat`; the dense mixer is quadratic in
            ``num_classes`` and unusable above a few hundred classes.
        proto_chunk: Chunk size along the prototype axis for the similarity
            computation. ``None`` (default) computes it in one shot; set it when
            ``num_prototypes * num_classes`` makes the intermediate too large.
            Exact either way — see :class:`PrototypicalFloat`.
        layer_norm: If ``True`` (default), apply an independent LayerNorm to each
            block's output before the weighted sum, so layers with different
            activation scales are comparable. Set ``False`` to weight the raw
            block outputs directly.
        fusion_layers: Restrict the fusion to the LAST ``k`` blocks instead of
            all of them. ``None`` (default) fuses every block, the published
            behaviour. Set it when a long run would otherwise let the softmax
            drift onto early blocks: at 100 k steps the full-XC head moved from
            centroid 8.50 to 7.20 and w(blocks 1-4) from 0.198 to 0.351 while
            downstream cmAP stayed flat and top-1 fell 0.014, whereas every
            per-task probe (which stops after 2.5-7.5 k steps) puts 0.55-0.88 of
            its mass on blocks 9-12. Restricting the pool removes the early
            attractor instead of steering around it, and blocks 5-7 — dead in
            every head measured — go with it.
        **encoder_kwargs: All keyword arguments forwarded to :class:`ViTEncoder`.
    """

    def __init__(
        self,
        num_classes: int,
        head_drop: float = 0.0,
        num_prototypes: int = 20,
        layer_norm: bool = True,
        fusion_layers: int | None = None,
        mixer: str = "dense",
        proto_chunk: int | None = None,
        **encoder_kwargs,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            head_drop=head_drop,
            num_prototypes=num_prototypes,
            mixer=mixer,
            proto_chunk=proto_chunk,
            **encoder_kwargs,
        )

        self.depth: int = encoder_kwargs.get("depth", 12)
        if fusion_layers is not None and not 1 <= fusion_layers <= self.depth:
            raise ValueError(
                f"fusion_layers must be in [1, depth={self.depth}], got {fusion_layers}"
            )
        # Number of blocks the fusion ranges over — the last `fusion_layers` of
        # them, or all of them when unset.
        self.num_layers: int = fusion_layers or self.depth
        # One learnable scalar per fused block, softmax-normalised at forward
        # time. Init at zeros → uniform softmax → equal-weight average to start.
        self.layer_weights = nn.Parameter(torch.zeros(self.num_layers))

        if layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(self.embed_dim, eps=encoder_kwargs.get("norm_eps", 1e-6))
                for _ in range(self.num_layers)
            ])
        else:
            self.layer_norms = None

    @property
    def fusion_blocks(self) -> list[int]:
        """1-based encoder-block index of each entry in ``layer_weights``.

        The identity ``[1..depth]`` unless ``fusion_layers`` restricted the
        fusion, in which case the weights refer to the LAST ``num_layers``
        blocks and every readout of them — logs, centroid, the notebooks — has
        to be offset by this, or a run that fuses blocks 8-12 is reported as one
        that fuses blocks 1-5.
        """
        return list(range(self.depth - self.num_layers + 1, self.depth + 1))

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

        # Keep only the blocks the fusion ranges over. A no-op when
        # `fusion_layers` is unset, since `num_layers` is then the full depth.
        hidden = hidden[-self.num_layers:]

        if self.layer_norms is not None:
            hidden = [ln(h) for ln, h in zip(self.layer_norms, hidden)]

        w = torch.softmax(self.layer_weights, dim=0)  # (L,)
        # Weighted sum accumulated over the block list, avoiding a materialised
        # (L, B, N+1, D) stack and its broadcast-product temporary. Kept
        # out-of-place so gradients still flow to ``layer_weights``.
        fused = w[0] * hidden[0]
        for i in range(1, len(hidden)):
            fused = fused + w[i] * hidden[i]  # (B, N+1, D)

        cls_features = fused[:, 0]
        patch_features = fused[:, 1:]

        z_f = patch_features - cls_features.unsqueeze(1)
        feat = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, *self.patch_embed.grid_size)
        feat = self.head_drop(feat)
        return self.head(feat)                        # (B, num_classes)


class PrototypicalFloat(nn.Module):  # protofloat
    """Cosine-prototype head: per-class prototypes, top-k spatial pooling, mixer.

    ``mixer`` selects how pooled prototype activations become class logits:

    ``"dense"`` (default)
        One ``nn.Linear(num_prototypes * num_classes, num_classes)`` — every
        class's logit is a learned combination of *every* prototype. This is
        what all published probes in this repo used and it is kept as the
        default so their results stay reproducible.

    ``"block_diagonal"``
        Each class's logit reads only its own ``num_prototypes`` prototypes
        (the standard PPNet formulation), so the mixer is
        ``num_prototypes * num_classes + num_classes`` parameters instead of
        ``num_prototypes * num_classes**2``. Required above a few hundred
        classes: at Xeno-Canto's 11 737 species with 20 prototypes the dense
        mixer is 2.755 **billion** parameters, against 0.25 M here. Prototypes
        are laid out class-major, i.e. prototype ``c * num_prototypes + p``
        belongs to class ``c``.

    ``proto_chunk`` bounds peak activation memory. The cosine similarities are a
    ``(B, num_prototypes * num_classes, H, W)`` tensor — 0.3 GB in bf16 at
    PER's 132 classes and batch 256, but **30.8 GB** at Xeno-Canto's 11 737,
    which no GPU has room for. Since top-k pooling is independent per prototype,
    computing the similarities in chunks along the prototype axis is exact, not
    an approximation: ``proto_chunk=4096`` caps that tensor at 0.54 GB. Leave it
    ``None`` (default) to compute in one shot, which is what every published
    probe did.
    """

    def __init__(
        self,
        dim: int,
        num_prototypes: int,
        num_classes: int,
        topk_k: int = 1,
        mixer: str = "dense",
        proto_chunk: int | None = None,
    ) -> None:
        super().__init__()
        if mixer not in ("dense", "block_diagonal"):
            raise ValueError(f"mixer must be 'dense' or 'block_diagonal', got {mixer!r}")
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes
        self.num_prototypes_total = num_prototypes * num_classes
        self.topk_k = topk_k
        self.mixer = mixer
        self.proto_chunk = proto_chunk

        # 1x1 convolutional kernels
        self.prototype_vectors = nn.Parameter(torch.randn(
            self.num_prototypes_total, dim, 1, 1) * 0.02)

        if mixer == "dense":
            self.linear = nn.Linear(self.num_prototypes_total, num_classes)
        else:
            # Same fan-in as one row of the dense mixer would see for its own
            # class, so the initial logit scale matches nn.Linear's default.
            bound = num_prototypes ** -0.5
            self.class_weight = nn.Parameter(
                torch.empty(num_classes, num_prototypes).uniform_(-bound, bound))
            self.class_bias = nn.Parameter(
                torch.empty(num_classes).uniform_(-bound, bound))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2: #for cls-input
            x = x[:, :, None, None]

        # standard cosine similarity (activation) - NO BINARIZATION
        x_norm = F.normalize(x, dim=1)
        B = x_norm.shape[0]
        k = self.topk_k if self.training else 1

        chunk = self.proto_chunk
        if chunk is None or chunk >= self.num_prototypes_total:
            p_norm = F.normalize(self.prototype_vectors, dim=1)  # directly normalize, no _binarise()
            act = F.conv2d(x_norm, p_norm)              # (B, P, H, W)

            # pooling & classification as before
            P = act.shape[1]
            act = act.view(B, P, -1)

            # top-k pooling: taking the mean of the k strongest hits
            pooled = act.topk(k, dim=-1).values.mean(-1)
        else:
            # Same computation, one slice of the prototype axis at a time, so the
            # (B, P, H, W) similarity tensor is never materialised in full. Exact:
            # both the cosine similarity and the top-k-over-space pooling are
            # independent per prototype.
            pooled = torch.cat([
                F.conv2d(x_norm, F.normalize(self.prototype_vectors[start:start + chunk], dim=1))
                 .view(B, -1, x_norm.shape[-2] * x_norm.shape[-1])
                 .topk(k, dim=-1).values.mean(-1)
                for start in range(0, self.num_prototypes_total, chunk)
            ], dim=1)

        if self.mixer == "dense":
            return self.linear(pooled)
        # (B, C, P) * (C, P) summed over P — each class sees only its own block.
        pooled = pooled.view(B, self.num_classes, self.num_prototypes_per_class)
        return (pooled * self.class_weight).sum(-1) + self.class_bias
