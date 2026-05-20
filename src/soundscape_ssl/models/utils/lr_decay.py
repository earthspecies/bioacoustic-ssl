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
    elif name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    else:
        return num_layers


def param_groups_lrd(
    model: nn.Module,
    weight_decay: float = 0.05,
    no_weight_decay_list: list[str] | None = None,
    layer_decay: float = 0.75,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Parameter groups for layer-wise lr decay.
    Compatible with MAE (model.encoder.blocks) and standalone ViT (model.blocks).
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
