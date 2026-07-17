import torch
import torch.nn as nn

from ...components import PatchEmbed, RotaryEmbedding, TransformerBlock, get_2d_sincos_pos_embed


class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = (512, 128),
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        grad_checkpoint: bool = False,
        pos_embed_type: str = "sinusoidal_2d",
        rope_base: float = 10000.0,
        qkv_norm: bool = False,
    ) -> None:
        super().__init__()
        self.pos_embed_type = pos_embed_type

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if pos_embed_type == "learned":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.rope = None

        elif pos_embed_type == "sinusoidal_2d":
            grid_h, grid_w = self.patch_embed.grid_size
            pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, cls_token=True)
            self.register_buffer("pos_embed", pos_embed.unsqueeze(0))
            self.rope = None

        elif pos_embed_type == "rope":
            self.pos_embed = None
            head_dim = embed_dim // num_heads
            self.rope = RotaryEmbedding(
                head_dim, max_seq_len=num_patches + 1, base=rope_base
            )

        elif pos_embed_type == "none":
            self.pos_embed = None
            self.rope = None

        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.pos_drop = nn.Dropout(drop_rate)

        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rates[i],
                norm_eps=norm_eps,
                use_checkpoint=grad_checkpoint,
                rope=self.rope,
                qkv_norm=qkv_norm
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor]:
        """
        mask_ratio=None  → standard ViT forward, returns latent (B, N+1, D)
        mask_ratio given → MAE forward, returns (latent, mask, ids_restore)
                           where latent is (B, len_keep+1, D)

        return_hidden=True → instead of the final-layer latent, return the list
            of per-block hidden states ``[h_1, ..., h_depth]`` (each
            ``(B, N+1, D)``, raw block outputs *before* the final ``self.norm``).
            Used for layerwise probing, where the best transferable features may
            live in a middle layer rather than the last. When combined with
            ``mask_ratio`` the return is ``(hidden_states, mask, ids_restore)``.
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)

        # Apply spatial positions to patch tokens before masking
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:, :]

        mask, ids_restore, ids_keep = None, None, None
        if mask_ratio is not None:
            x, mask, ids_restore, ids_keep = self._random_masking(x, mask_ratio)

        # Prepend cls token with its positional embedding
        if self.pos_embed is not None:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        else:
            cls_token = self.cls_token
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # RoPE position IDs: sequential for full sequence, sparse when masked
        pos_ids = None
        if self.pos_embed_type == "rope" and mask_ratio is not None:
            pos_ids = torch.cat([
                torch.zeros(B, 1, device=x.device, dtype=torch.long),
                ids_keep + 1,
            ], dim=1)

        x = self.pos_drop(x)
        hidden_states: list[torch.Tensor] = []
        for blk in self.blocks:
            x = blk(x, pos_ids=pos_ids)
            if return_hidden:
                hidden_states.append(x)

        if return_hidden:
            # Skip the final shared self.norm: it is specialised for the last
            # layer, so the layerwise head normalises each layer on its own.
            if mask_ratio is not None:
                return hidden_states, mask, ids_restore
            return hidden_states

        x = self.norm(x)
        if mask_ratio is not None:
            return x, mask, ids_restore
        return x
