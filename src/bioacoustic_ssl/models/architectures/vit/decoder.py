import torch
import torch.nn as nn

from ...components import RotaryEmbedding, TransformerBlock, get_2d_sincos_pos_embed


class ViTDecoder(nn.Module):
    def __init__(
        self,
        num_patches: int,
        grid_size: tuple[int, int],
        encoder_dim: int,
        decoder_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        patch_pixels: int,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        grad_checkpoint: bool = False,
        pos_embed_type: str = "sinusoidal_2d",
        rope_base: float = 10000.0,
        qkv_norm: bool = False,
    ) -> None:
        super().__init__()
        self.pos_embed_type = pos_embed_type
        self.num_patches = num_patches
        self.grid_size = grid_size

        self.embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        if pos_embed_type == "learned":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, decoder_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.rope = None

        elif pos_embed_type == "sinusoidal_2d":
            grid_h, grid_w = grid_size
            pos_embed = get_2d_sincos_pos_embed(decoder_dim, grid_h, grid_w, cls_token=True)
            self.register_buffer("pos_embed", pos_embed.unsqueeze(0))
            self.rope = None

        elif pos_embed_type == "rope":
            self.pos_embed = None
            head_dim = decoder_dim // decoder_num_heads
            self.rope = RotaryEmbedding(
                head_dim, max_seq_len=num_patches + 1, base=rope_base
            )

        elif pos_embed_type == "none":
            self.pos_embed = None
            self.rope = None

        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rate,
                norm_eps=norm_eps,
                use_checkpoint=grad_checkpoint,
                rope=self.rope,
                qkv_norm=qkv_norm
            )
            for _ in range(decoder_depth)
        ])

        self.norm = nn.LayerNorm(decoder_dim, eps=norm_eps)
        self.head = nn.Linear(decoder_dim, patch_pixels)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor, pos_ids: torch.Tensor = None) -> torch.Tensor:
        x = self.embed(x)

        mask_tokens = self.mask_token.expand(
            x.shape[0],
            ids_restore.shape[1] + 1 - x.shape[1],
            -1,
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)

        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]),
        )

        x = torch.cat([x[:, :1, :], x_], dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x, pos_ids=pos_ids)

        x = self.norm(x)
        x = self.head(x)
        return x[:, 1:, :]
