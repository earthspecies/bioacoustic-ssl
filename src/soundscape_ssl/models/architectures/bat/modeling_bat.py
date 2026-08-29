"""Vendored BAT encoder (ViT-B/16, gated attention, post-norm blocks).

Upstream: https://huggingface.co/lrauch/BAT-vit-b16-pretrainedAS2M/blob/main/modeling_bat.py
Revision: 175109327540c72f4b678b149e7cfaf0ee45d3e9

Vendored third-party code (MIT). Its value is being diff-able against upstream,
so do not reformat, rename or tidy it. Every deviation is marked ``# VENDOR:``:

1. ``transformers.PreTrainedModel`` -> the local ``BatPreTrainedModel`` shim and
   ``transformers.utils.ModelOutput`` -> a plain dataclass. ``transformers`` is
   unimportable under torch 2.6 in this env and is not a train-time dependency.
2. ``Attention.forward`` uses ``F.scaled_dot_product_attention``. The original
   explicit path is retained and taken when ``output_attentions=True``, which is
   what the SDPA parity test compares against.
3. Optional gradient checkpointing over the block loop
   (``config.grad_checkpoint``); BAT ships none and 513 tokens is ~2x our other
   arms.
4. ``BatModel.from_pretrained``: loads ``config.json`` + ``model.safetensors``
   from the Hub with ``load_state_dict(strict=True)``.

BAT must never be converted into ``ViTEncoder`` (its gated attention and
post-norm residual structure would be silently dropped, giving a plausible bad
number), so it is deliberately absent from ``scripts/convert_external_ckpt.py``.
"""

# ruff: noqa
# fmt: off
# (vendored: linting and formatting are off so the file stays byte-comparable
#  with upstream; do not remove these two directives)

import json  # VENDOR: for from_pretrained
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.utils.checkpoint  # VENDOR: for grad_checkpoint
from torch import Tensor, nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download  # VENDOR: for from_pretrained
from safetensors.torch import load_file  # VENDOR: for from_pretrained

from .configuration_bat import BatConfig


# VENDOR: stands in for `transformers.PreTrainedModel`; only `__init__(config)`
# and `post_init()` are used by the model below.
class BatPreTrainedModel(nn.Module):
    config_class: type = BatConfig

    def __init__(self, config: BatConfig):
        super().__init__()
        self.config = config

    def post_init(self) -> None:
        self.apply(self._init_weights)


# VENDOR: `ModelOutput` -> plain dataclass.
@dataclass
class BatModelOutput:
    last_hidden_state: Tensor = None
    pooler_output: Tensor = None
    patch_tokens: Tensor = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None


def get_2d_sincos_pos_embed_flexible(embed_dim: int, grid_size: Tuple[int, int], cls_token: bool = False) -> np.ndarray:
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.s = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.s


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep and keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PatchEmbed(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int] = (1024, 128),
        patch_size: Tuple[int, int] = (16, 16),
        in_chans: int = 1,
        dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_size[1] // patch_size[1]) * (input_size[0] // patch_size[0])
        self.patch_ft = (input_size[1] // patch_size[1], input_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        return x.transpose(1, 2)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Any = nn.GELU,
        norm_layer: Optional[Any] = None,
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_gate: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "hidden size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gate = nn.Linear(dim, dim) if use_gate else None

    def forward(self, x: Tensor, output_attentions: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # VENDOR: SDPA replaces the explicit (B, heads, N, N) attention matrix.
        # The gate is applied to the attention *output*, so this is identical.
        # The original path is kept for `output_attentions=True`.
        if output_attentions:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn_out = attn @ v
        else:
            attn = None
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scale,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        if self.gate is not None:
            attn_out = attn_out * self.gate(x).sigmoid()
        x = self.proj(attn_out)
        return self.proj_drop(x), attn


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Any = nn.GELU,
        norm_layer: Any = nn.LayerNorm,
        init_scale_values: Optional[float] = None,
        layer_norm_first: bool = False,
        use_gate: bool = True,
    ):
        super().__init__()
        self.forward_fn = self.forward_norm_first if layer_norm_first else self.forward_norm_last
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_gate=use_gate,
        )
        self.ls1 = LayerScale(dim, init_scale_values) if init_scale_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_scale_values) if init_scale_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    # VENDOR: `output_attentions` threaded through so `Attention` can pick the
    # SDPA path (see above); the residual structure is untouched.
    def forward(self, x: Tensor, output_attentions: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        return self.forward_fn(x, output_attentions)

    def forward_norm_last(self, x: Tensor, output_attentions: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        z, attn = self.attn(x, output_attentions=output_attentions)
        z = self.drop_path1(self.ls1(z))
        x = self.norm1(x + z)
        z = self.drop_path2(self.ls2(self.mlp(x)))
        x = self.norm2(x + z)
        return x, attn

    def forward_norm_first(self, x: Tensor, output_attentions: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        z, attn = self.attn(self.norm1(x), output_attentions=output_attentions)
        x = x + self.drop_path1(self.ls1(z))
        z = self.mlp(self.norm2(x))
        x = x + self.drop_path2(self.ls2(z))
        return x, attn


class BatModel(BatPreTrainedModel):
    config_class = BatConfig
    base_model_prefix = "bat"
    supports_gradient_checkpointing = True  # VENDOR: was False; see `grad_checkpoint`

    def __init__(self, config: BatConfig):
        super().__init__(config)
        input_shape = tuple(config.input_shape)
        patch_size = tuple(config.patch_size)
        self.patch_size = patch_size
        self.grad_checkpoint = config.grad_checkpoint  # VENDOR
        self.patch_embed = PatchEmbed(input_shape, patch_size, 1, config.hidden_size)
        num_freq_patches, num_time_patches = self.patch_embed.patch_ft
        pos_embed = get_2d_sincos_pos_embed_flexible(
            config.hidden_size,
            (num_freq_patches, min(config.chunk_time_patches, num_time_patches)),
            cls_token=False,
        )
        self.pos_embed = nn.Parameter(
            torch.tensor(pos_embed).float().unsqueeze(0),
            requires_grad=config.pos_trainable,
        )
        norm_layer = partial(nn.LayerNorm, eps=config.layer_norm_eps)
        self.pre_norm = norm_layer(config.hidden_size) if config.pre_norm else nn.Identity()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02).contiguous()
        if config.drop_path_rate > 0:
            dpr = [d.item() for d in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        else:
            dpr = [0.0] * config.num_hidden_layers
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    drop=config.hidden_dropout_prob,
                    attn_drop=config.attention_probs_dropout_prob,
                    drop_path=dpr[i],
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                    layer_norm_first=config.layer_norm_first,
                    init_scale_values=config.init_scale_values,
                    use_gate=config.use_gate,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.post_init()

    # VENDOR: loader replacing `PreTrainedModel.from_pretrained`. `strict=True` is
    # the correctness gate for this whole arm (every key except `pre_norm.*` and
    # `attn.gate.*` collides name-for-name with our `ViTEncoder`, so a lenient
    # load would silently drop the gates) — do not relax it.
    @classmethod
    def from_pretrained(cls, repo_id: str, revision: Optional[str] = None, **config_overrides: Any) -> "BatModel":
        with open(hf_hub_download(repo_id, "config.json", revision=revision)) as f:
            config_dict = json.load(f)
        config_dict.update(config_overrides)
        model = cls(cls.config_class(**config_dict))
        state_dict = load_file(hf_hub_download(repo_id, "model.safetensors", revision=revision))
        model.load_state_dict(state_dict, strict=True)
        return model

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

    @property
    def num_freq_patches(self) -> int:
        return self.config.input_shape[1] // self.config.patch_size[1]

    def _canonicalize_input(self, input_features: Tensor) -> Tensor:
        if input_features.ndim == 3:
            input_features = input_features.unsqueeze(1)
        if input_features.ndim != 4:
            raise ValueError(
                "BAT expects log-mel input features with shape [batch, 1, time, mel] "
                "or [batch, time, mel]. Use BatAudioProcessor for raw waveforms."
            )
        expected_mels = self.config.input_shape[1]
        if input_features.shape[1] != 1:
            raise ValueError(f"Expected a single-channel spectrogram, got shape {tuple(input_features.shape)}.")
        if input_features.shape[-1] != expected_mels and input_features.shape[-2] == expected_mels:
            input_features = input_features.transpose(-2, -1)
        if input_features.shape[-1] != expected_mels:
            raise ValueError(
                f"Expected {expected_mels} mel bins in the last dimension, got shape {tuple(input_features.shape)}."
            )
        return input_features

    def _extract_token_cache(
        self,
        input_features: Tensor,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tuple[Tensor, ...]]]:
        x = self._canonicalize_input(input_features)
        x = self.patch_embed(x)
        batch_size, num_tokens, emb_dim = x.shape
        num_freq_patches = self.num_freq_patches
        if num_tokens % num_freq_patches != 0:
            raise ValueError(
                f"Patch sequence length {num_tokens} is not compatible with {num_freq_patches} frequency patches."
            )

        x = x.reshape(batch_size, -1, num_freq_patches, emb_dim)
        chunks = torch.split(x, self.config.chunk_time_patches, dim=1)
        cls_chunks = []
        patch_chunks = []
        chunk_attentions = []

        for chunk in chunks:
            xi = chunk.flatten(1, 2)
            if xi.shape[1] > self.pos_embed.shape[1]:
                raise ValueError(
                    f"Chunk has {xi.shape[1]} patches, but positional embedding has {self.pos_embed.shape[1]}."
                )
            xi = xi + self.pos_embed[:, : xi.shape[1], :].to(dtype=xi.dtype)
            xi = torch.cat((self.cls_token.expand(batch_size, -1, -1).to(dtype=xi.dtype), xi), dim=1)
            xi = self.pre_norm(xi)

            cls_tokens = []
            patch_tokens = []
            attentions = []
            for block in self.blocks:
                # VENDOR: optional gradient checkpointing, as `TransformerBlock` does it.
                if self.grad_checkpoint and self.training:
                    xi, attn = torch.utils.checkpoint.checkpoint(
                        block, xi, output_attentions, use_reentrant=False
                    )
                else:
                    xi, attn = block(xi, output_attentions)
                cls_tokens.append(xi[:, 0])
                patch_tokens.append(xi[:, 1:])
                if output_attentions:
                    attentions.append(attn)

            cls_chunks.append(torch.stack(cls_tokens))
            patch_tokens = torch.stack(patch_tokens)
            patch_tokens = patch_tokens.transpose(2, 3)
            patch_tokens = patch_tokens.reshape(
                len(self.blocks),
                batch_size,
                emb_dim,
                -1,
                num_freq_patches,
            )
            patch_chunks.append(patch_tokens)
            if output_attentions:
                chunk_attentions.append(tuple(attentions))

        cls_by_layer = torch.stack(cls_chunks).mean(dim=0)
        patch_by_layer = torch.cat(patch_chunks, dim=3)
        attentions = chunk_attentions[0] if output_attentions and len(chunk_attentions) == 1 else None
        return cls_by_layer, patch_by_layer, attentions

    def extract_features(self, input_features: Tensor) -> Dict[str, Tensor]:
        cls_tokens, patch_tokens, _ = self._extract_token_cache(input_features, output_attentions=False)
        return {"cls_tokens": cls_tokens, "patch_tokens": patch_tokens}

    def forward_encoder(self, input_features: Tensor) -> Tensor:
        return self.forward(input_features=input_features, return_dict=True).last_hidden_state

    def forward(
        self,
        input_features: Optional[Tensor] = None,
        input_values: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if input_features is None:
            input_features = input_values
        if input_features is None:
            raise ValueError("Pass BAT log-mel features as `input_features`.")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        cls_tokens, patch_tokens, attentions = self._extract_token_cache(
            input_features,
            output_attentions=output_attentions,
        )
        final_cls = cls_tokens[-1]
        final_patch_grid = patch_tokens[-1]
        final_patch_sequence = final_patch_grid.permute(0, 2, 3, 1).reshape(final_patch_grid.shape[0], -1, final_patch_grid.shape[1])
        last_hidden_state = torch.cat((final_cls.unsqueeze(1), final_patch_sequence), dim=1)

        hidden_states = None
        if output_hidden_states:
            states = []
            for layer_idx in range(patch_tokens.shape[0]):
                patch_sequence = patch_tokens[layer_idx].permute(0, 2, 3, 1).reshape(
                    patch_tokens.shape[1],
                    -1,
                    patch_tokens.shape[2],
                )
                states.append(torch.cat((cls_tokens[layer_idx].unsqueeze(1), patch_sequence), dim=1))
            hidden_states = tuple(states)

        if not return_dict:
            output = (last_hidden_state, final_cls, final_patch_grid)
            if output_hidden_states:
                output = output + (hidden_states,)
            if output_attentions:
                output = output + (attentions,)
            return output

        return BatModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=final_cls,
            patch_tokens=final_patch_grid,
            hidden_states=hidden_states,
            attentions=attentions,
        )
