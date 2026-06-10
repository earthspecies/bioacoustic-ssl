import torch
import torch.nn as nn

from .vit.decoder import ViTDecoder
from .vit.encoder import ViTEncoder

__all__ = ["MAE"]


class MAE(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 1,
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.80,
        norm_pix_loss: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        decoder_drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        grad_checkpoint: bool = False,
        pos_embed_type: str = "sinusoidal_2d",
        decoder_pos_embed_type: str = "sinusoidal_2d",
        rope_base: float = 10000.0,
        qkv_norm: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_eps=norm_eps,
            grad_checkpoint=grad_checkpoint,
            pos_embed_type=pos_embed_type,
            rope_base=rope_base,
            qkv_norm=qkv_norm
        )

        num_patches = self.encoder.patch_embed.num_patches
        grid_size = self.encoder.patch_embed.grid_size

        if isinstance(patch_size, int):
            patch_pixels = patch_size * patch_size * in_chans
        else:
            patch_pixels = patch_size[0] * patch_size[1] * in_chans

        self.decoder = ViTDecoder(
            num_patches=num_patches,
            grid_size=grid_size,
            encoder_dim=encoder_embed_dim,
            decoder_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            patch_pixels=patch_pixels,
            mlp_ratio=mlp_ratio,
            drop_path_rate=decoder_drop_path_rate,
            norm_eps=norm_eps,
            grad_checkpoint=grad_checkpoint,
            pos_embed_type=decoder_pos_embed_type,
            rope_base=rope_base,
            qkv_norm=qkv_norm
        )

        self._num_features = encoder_embed_dim

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def num_patches(self) -> int:
        return self.encoder.patch_embed.num_patches

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        if isinstance(self.patch_size, int):
            p_h = p_w = self.patch_size
        else:
            p_h, p_w = self.patch_size

        h = imgs.shape[2] // p_h
        w = imgs.shape[3] // p_w

        x = imgs.reshape(imgs.shape[0], 1, h, p_h, w, p_w)
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(imgs.shape[0], h * w, p_h * p_w * 1)
        return x

    def forward(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        latent, mask, ids_restore = self.encoder(imgs, mask_ratio=self.mask_ratio)
        pred = self.decoder(latent, ids_restore)

        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-4) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return {
            "loss": loss,
            "pred": pred,
            "mask": mask,
            "latent": latent[:, 1:, :],
            "cls_token": latent[:, 0],
        }
