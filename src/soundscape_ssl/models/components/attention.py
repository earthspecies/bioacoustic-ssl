import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope: nn.Module | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.rope = rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        pos_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D_head)

        if self.rope is not None:
            q = self.rope(q, pos_ids=pos_ids)
            k = self.rope(k, pos_ids=pos_ids)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SDPAttention(nn.Module):
    """
    Drop-in replacement for standard Attention using PyTorch's native
    scaled_dot_product_attention.  Automatically uses FlashAttention when
    the shapes/dtypes allow it.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.is_causal = is_causal
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape

        # (B, N, 3*C) -> (B, N, 3, H, D) -> (3, B, H, N, D)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        # Fused attention: handles scaling, softmax, and dropout internally
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=self.is_causal,
        )

        # (B, H, N, D) -> (B, N, H, D) -> (B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
