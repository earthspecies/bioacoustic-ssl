from __future__ import annotations

from typing import Any

import torch.nn as nn


def get_layer_id_for_vit(name: str, num_layers: int) -> int:
    """
    Assign a layer id to a parameter name following BEiT:
    https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33

    Handles both MAE-style names (encoder.blocks.X.*) and plain ViT names (blocks.X.*).
    """
    if name.startswith("encoder."):
        name = name[len("encoder."):]

    if name in ("cls_token", "pos_embed"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("pre_norm"):
        # BAT normalises once before block 0, so it belongs to the input stage,
        # not to the head group the fall-through below would put it in.
        return 0
    elif name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    else:
        return num_layers


def param_groups_lrd(
    model: nn.Module,
    weight_decay: float = 0.05,
    no_weight_decay_list: list[str] | None = None,
    layer_decay: float = 0.75,
    prototype_lr: float | None = None,
    layer_weights_lr: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Parameter groups for layer-wise lr decay.
    Compatible with MAE (model.encoder.blocks) and standalone ViT (model.blocks).

    Args:
        model: The model whose parameters are to be grouped.
        weight_decay: Weight decay applied to non-excluded parameters.
        no_weight_decay_list: Parameter name patterns exempt from weight decay.
        layer_decay: Multiplicative factor per layer for lr decay.
        prototype_lr: When set, all non-backbone (head-side) trainable params are
            removed from the layer-decay groups and placed in a dedicated group
            with this fixed absolute learning rate. This covers the prototype
            head (``head.prototype_vectors``, and ``head.linear.*`` or
            ``head.class_weight`` / ``head.class_bias`` depending on the mixer)
            as well as the
            layerwise-fusion params of :class:`ViTProtoLayerwise`
            (``layer_weights``, ``layer_norms.*``). The layer-decay / ``base_lr``
            groups are then left with genuine backbone params only. The head is
            split into ``head_decay`` / ``head_no_decay`` on the same rule the
            backbone uses (1-dim params exempt), plus ``prototype_vectors``,
            which is L2-normalised on every forward and so cannot be affected
            by decay except through its optimisation dynamics. Leave as
            ``None`` (default) for standard :class:`ViTClassifier` usage.
        layer_weights_lr: When set (requires ``prototype_lr``), the layerwise
            softmax-fusion weights (``layer_weights`` of
            :class:`ViTProtoLayerwise`) are split out of the head group into
            their own group at this fixed absolute learning rate — useful to run
            the 12-dim block-selection softmax cooler than the head so it does
            not collapse onto a single block prematurely. When ``None`` (default)
            ``layer_weights`` stays folded into the head group at ``prototype_lr``.
    """
    if no_weight_decay_list is None:
        no_weight_decay_list = []

    def _is_layer_weights(name: str) -> bool:
        return name.startswith("layer_weights")

    def _is_head_param(name: str) -> bool:
        return (
            "prototype_vectors" in name
            or "head.linear" in name
            # block-diagonal mixer of PrototypicalFloat — the same role
            # `head.linear` plays for the dense mixer, so the same group.
            or "head.class_weight" in name
            or "head.class_bias" in name
            or _is_layer_weights(name)
            or name.startswith("layer_norms")
        )

    param_group_names: dict[str, dict[str, Any]] = {}
    param_groups: dict[str, dict[str, Any]] = {}

    if hasattr(model, "encoder"):
        num_layers = len(model.encoder.blocks) + 1
    else:
        num_layers = len(model.blocks) + 1

    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Head-side params (prototype head + layerwise fusion) get their own
        # fixed-lr group (see below); only backbone params stay on base_lr.
        if prototype_lr is not None and _is_head_param(n):
            continue

        if p.ndim == 1 or n in no_weight_decay_list or any(n.endswith(pat) for pat in no_weight_decay_list):
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # Add a dedicated fixed-lr group for all head-side params (prototype head +
    # layerwise fusion) when prototype_lr is requested. When layer_weights_lr is
    # also set, the block-selection softmax weights are split into their own
    # group so they can run cooler than the rest of the head.
    if prototype_lr is not None:
        split_lw = layer_weights_lr is not None

        def _in_head(name: str) -> bool:
            return _is_head_param(name) and not (split_lw and _is_layer_weights(name))

        head_names = [
            n for n, p in model.named_parameters()
            if p.requires_grad and _in_head(n)
        ]
        head_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and _in_head(n)
        ]
        # Same decay convention as the backbone groups above: 1-dim params are
        # exempt. `prototype_vectors` is exempt too — it is L2-normalised on
        # every forward, so decaying it cannot change the loss, it only shrinks
        # the norm and inflates the effective step on the direction.
        def _head_decays(name: str, param: nn.Parameter) -> bool:
            return param.ndim > 1 and "prototype_vectors" not in name

        for suffix, decays in (("decay", True), ("no_decay", False)):
            sel = [
                (n, p) for n, p in zip(head_names, head_params)
                if _head_decays(n, p) is decays
            ]
            if not sel:
                continue
            this_decay = weight_decay if decays else 0.0
            param_group_names[f"head_{suffix}"] = {
                "lr": prototype_lr,
                "weight_decay": this_decay,
                "params": [n for n, _ in sel],
            }
            param_groups[f"head_{suffix}"] = {
                "lr": prototype_lr,
                "weight_decay": this_decay,
                "params": [p for _, p in sel],
            }

        if split_lw:
            lw_names = [
                n for n, p in model.named_parameters()
                if p.requires_grad and _is_layer_weights(n)
            ]
            lw_params = [
                p for n, p in model.named_parameters()
                if p.requires_grad and _is_layer_weights(n)
            ]
            if lw_params:
                # 12-dim fusion softmax: 1-dim, so no decay.
                param_group_names["layer_weights"] = {
                    "lr": layer_weights_lr,
                    "weight_decay": 0.0,
                    "params": lw_names,
                }
                param_groups["layer_weights"] = {
                    "lr": layer_weights_lr,
                    "weight_decay": 0.0,
                    "params": lw_params,
                }

    return list(param_groups.values()), param_group_names


def param_groups_lrd_pp(
    model: nn.Module,
    weight_decay: float = 0.05,
    no_weight_decay_list: list[str] | None = None,
    layer_decay: float = 0.75,
    last_layer_lr: float = 5e-4,
    prototype_lr: float = 0.05,
) -> list[dict[str, Any]]:
    """
    Parameter groups for layer-wise lr decay with a ppnet head.
    Encoder parameters get layer-wise decay; ppnet parameters are grouped manually.
    """
    if no_weight_decay_list is None:
        no_weight_decay_list = []

    param_group_names: dict[str, dict[str, Any]] = {}
    param_groups: dict[str, dict[str, Any]] = {}

    if hasattr(model, "encoder"):
        num_layers = len(model.encoder.blocks) + 1
    else:
        num_layers = len(model.blocks) + 1

    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if n.startswith("ppnet"):
            continue

        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    param_groups_list = list(param_groups.values())

    addon_params = list(model.ppnet.add_on_layers.parameters())
    param_groups_list.append({
        "params": addon_params,
        "lr": 3e-2,
        "weight_decay": 1e-4,
    })

    proto_params = [model.ppnet.prototype_vectors]
    param_groups_list.append({
        "params": proto_params,
        "lr": prototype_lr,
    })

    last_params = list(model.ppnet.last_layer.parameters())
    param_groups_list.append({
        "params": last_params,
        "lr": last_layer_lr,
        "weight_decay": 1e-4,
    })

    already_grouped = set(addon_params + proto_params + last_params)
    rest = [p for p in model.ppnet.parameters() if p not in already_grouped]
    if rest:
        param_groups_list.append({"params": rest})

    return param_groups_list
