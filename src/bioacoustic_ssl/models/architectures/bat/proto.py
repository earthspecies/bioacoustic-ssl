"""Prototypical probing/finetuning head on top of the vendored BAT encoder.

Ours, not vendored — kept out of ``modeling_bat.py`` so that file stays
diff-able against the pinned upstream revision.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..vit.classifier import PrototypicalFloat
from .modeling_bat import BatModel

BAT_REPO_ID = "lrauch/BAT-vit-b16-pretrainedAS2M"
BAT_REVISION = "175109327540c72f4b678b149e7cfaf0ee45d3e9"


class BatProtoFloat(nn.Module):
    """BAT backbone with the same :class:`PrototypicalFloat` head every proto arm uses.

    BAT does not load into :class:`ViTEncoder` (gated attention, post-norm
    blocks, a pre-block LayerNorm and no final norm), so this is a thin wrapper
    around the vendored :class:`BatModel` rather than another ``ViTEncoder``
    subclass. Two things make the wrapper thin:

    * BAT emits ``patch_tokens`` already shaped ``(B, D, time_patches,
      freq_patches)`` — the grid layout :meth:`ViTProtoFloat.forward` builds by
      hand — so no permute/reshape is needed.
    * Its post-norm blocks end in a LayerNorm, so ``patch_tokens`` and
      ``pooler_output`` are already normalised; there is no final ``norm`` to
      apply, unlike the ``ViTEncoder`` arms.

    The backbone submodule **must** be called ``encoder``: ``param_groups_lrd``
    branches on ``hasattr(model, "encoder")`` to find the block count, and
    ``get_layer_id_for_vit`` strips that prefix, which is what makes each
    block's ``attn.gate.*`` inherit its block's layer id.

    Weights arrive at construction from the Hub with ``strict=True``, so this arm
    needs no ``trainer.resume_from_checkpoint``.

    Args:
        num_classes: Number of output classes (logits dimension).
        repo_id: Hub repo holding ``config.json`` + ``model.safetensors``.
        revision: Pinned Hub revision. Do not float this.
        num_prototypes: Prototypes per class for the head.
        head_drop: Dropout applied to the focal-similarity grid before the head.
        grad_checkpoint: Checkpoint the encoder's block loop. Only useful when
            the backbone is trainable; a frozen backbone builds no graph.
    """

    def __init__(
        self,
        num_classes: int,
        repo_id: str = BAT_REPO_ID,
        revision: str = BAT_REVISION,
        num_prototypes: int = 20,
        head_drop: float = 0.0,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = BatModel.from_pretrained(
            repo_id,
            revision=revision,
            grad_checkpoint=grad_checkpoint,
        )
        self.embed_dim: int = self.encoder.config.hidden_size

        # BAT ships pos_embed as a *parameter* pinned to a fixed sincos grid
        # (`pos_trainable: false`), where the ViTEncoder arms use a buffer. Record
        # it so unfreezing the backbone does not quietly start training it.
        self._always_frozen: tuple[str, ...] = () if self.encoder.config.pos_trainable else ("encoder.pos_embed",)

        self.head_drop = nn.Dropout(head_drop)
        self.head = PrototypicalFloat(self.embed_dim, num_prototypes, num_classes)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze (or unfreeze) everything except the prototype head.

        Mirrors :meth:`ViTProtoFloat.freeze_backbone`, including the rule that
        the split is by the ``head`` name prefix, so ``param_groups_lrd`` sees
        only head params when probing. ``pos_embed`` stays frozen either way when
        the checkpoint says it is not trainable.

        Args:
            freeze: ``True`` (default) to freeze the backbone; ``False`` for
                    full finetuning.
        """
        for name, param in self.named_parameters():
            if name in self._always_frozen:
                param.requires_grad_(False)
            elif not name.startswith("head"):
                param.requires_grad_(not freeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: log-mel features → class logits.

        Args:
            x: BAT input features, shape ``(B, 1, frames, n_mels)``. The
                encoder's ``_canonicalize_input`` transposes the house
                ``(B, 1, n_mels, frames)`` layout automatically.

        Returns:
            Class logits, shape ``(B, num_classes)``.
        """
        out = self.encoder(input_features=x)

        # Focal similarity, as in ViTProtoFloat: subtract the CLS embedding from
        # every patch token. BAT itself does not do this; using it here is what
        # keeps the head identical across arms, which is the point of the
        # comparison.
        z_f = out.patch_tokens - out.pooler_output[:, :, None, None]
        return self.head(self.head_drop(z_f))
