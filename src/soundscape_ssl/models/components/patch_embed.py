import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        # Support tuple (h, w) or single int
        img_size: tuple[int, int] = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        patch_size: tuple[int, int] = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)           # (B, embed_dim, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x
