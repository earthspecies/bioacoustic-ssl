"""Map training checkpoints onto the published HuggingFace state dicts.

Stays in this repository — it is not part of the model repo, because nobody
loading the weights needs it. It exists so that the key mapping is one testable
function per artifact rather than a dict comprehension buried in a script, and
so the released geometry is stated once.

The mapping is deliberately trivial. Every module in
:mod:`soundscape_ssl.hf.modeling_soundscape_mae` carries the name its
training-time counterpart carries, so converting a checkpoint moves prefixes
around and touches no tensor. That is the property worth having: a conversion
that reshapes or renames weights is a conversion that can silently be wrong.
"""

from collections.abc import Mapping

import torch

from .configuration_soundscape_mae import SoundscapeMAEConfig

# `configs/module/model/backbone/vit.yaml` — the geometry every released
# artifact was trained with. ViT-B/16 over a (128 mel, 512 frame) input, so an
# (8, 32) token grid, with QK-norm on.
RELEASE_GEOMETRY = {
    "num_mel_bins": 128,
    "num_frames": 512,
    "patch_size": 16,
    "num_channels": 1,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "qkv_norm": True,
    "layer_norm_eps": 1e-6,
}

# `configs/module/model/layerwise/vit_xc.yaml` — the released head's own
# hyperparameters, the two that change its shape or its arithmetic.
RELEASE_HEAD = {"num_prototypes": 20, "proto_chunk": 4096}

# Everything in a `ViTProtoLayerwise` checkpoint that is *not* the encoder. The
# encoder keys get the `encoder.` prefix the published classifier nests them
# under; these keep their names.
HEAD_KEY_PREFIXES = ("head.", "layer_weights", "layer_norms.")


def encoder_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Published :class:`SoundscapeMAEModel` weights from a training state dict.

    Accepts either an MAE training checkpoint's ``model`` dict, whose keys are
    prefixed ``encoder.`` and ``decoder.``, or a bare ``ViTEncoder`` state dict.
    The decoder is dropped: the released encoder artifact does not carry it.

    Args:
        state: The training-time state dict.

    Returns:
        The same tensors under the published key names.
    """
    if any(key.startswith("encoder.") for key in state):
        return {
            key.removeprefix("encoder."): value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
    return {key: value for key, value in state.items() if not key.startswith("decoder.")}


def classifier_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Published classifier weights from a ``ViTProtoLayerwise`` state dict.

    The probe checkpoint holds its frozen encoder flat alongside the head, so
    the encoder keys are the ones that move: they nest under ``encoder.``, while
    the fusion weights, the per-block norms and the prototypical head keep their
    names.

    Args:
        state: The training-time state dict.

    Returns:
        The same tensors under the published key names.
    """
    return {
        key if key.startswith(HEAD_KEY_PREFIXES) else f"encoder.{key}": value
        for key, value in state.items()
    }


def build_config(id2label: dict[int, str] | None = None) -> SoundscapeMAEConfig:
    """The released config: fixed geometry, plus a label map when there is a head.

    Args:
        id2label: Output index to taxon name. ``None`` builds the encoder-only
            config, whose label fields are meaningless.

    Returns:
        The config for the artifact being exported.
    """
    if id2label is None:
        return SoundscapeMAEConfig(**RELEASE_GEOMETRY, **RELEASE_HEAD)
    return SoundscapeMAEConfig(
        **RELEASE_GEOMETRY,
        **RELEASE_HEAD,
        id2label=id2label,
        label2id={label: index for index, label in id2label.items()},
    )


def reference_encoder_kwargs() -> dict[str, object]:
    """:data:`RELEASE_GEOMETRY` as the in-repo ``ViTEncoder`` spells it.

    The bridge between the two implementations' parameter names, so the export's
    parity check builds both sides from one description of the geometry rather
    than from two that have to be kept in step by hand.

    Returns:
        Keyword arguments for ``ViTEncoder`` / ``ViTProtoLayerwise``.
    """
    return {
        "img_size": (RELEASE_GEOMETRY["num_mel_bins"], RELEASE_GEOMETRY["num_frames"]),
        "patch_size": RELEASE_GEOMETRY["patch_size"],
        "in_chans": RELEASE_GEOMETRY["num_channels"],
        "embed_dim": RELEASE_GEOMETRY["hidden_size"],
        "depth": RELEASE_GEOMETRY["num_hidden_layers"],
        "num_heads": RELEASE_GEOMETRY["num_attention_heads"],
        "mlp_ratio": RELEASE_GEOMETRY["mlp_ratio"],
        "qkv_bias": RELEASE_GEOMETRY["qkv_bias"],
        "norm_eps": RELEASE_GEOMETRY["layer_norm_eps"],
        "qkv_norm": RELEASE_GEOMETRY["qkv_norm"],
        "pos_embed_type": "sinusoidal_2d",
    }


__all__ = [
    "HEAD_KEY_PREFIXES",
    "RELEASE_GEOMETRY",
    "RELEASE_HEAD",
    "build_config",
    "classifier_state_dict",
    "encoder_state_dict",
    "reference_encoder_kwargs",
]
