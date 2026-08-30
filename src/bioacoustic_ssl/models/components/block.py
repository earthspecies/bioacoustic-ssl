import torch
import torch.nn as nn

from .attention import Attention
from .drop import DropPath
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_eps: float = 1e-6,
        use_checkpoint: bool = False,
        rope: nn.Module | None = None,
        qkv_norm: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(
            dim, num_heads, qkv_bias, attn_drop, drop, rope=rope, qkv_norm=qkv_norm
        )
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x: torch.Tensor, pos_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), pos_ids=pos_ids))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(
        self,
        x: torch.Tensor,
        pos_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, pos_ids, use_reentrant=False
            )
        return self._forward_impl(x, pos_ids=pos_ids)
