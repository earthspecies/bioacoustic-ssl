import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    When applied inside a residual branch:
        x = x + DropPath(attn(x))
    this randomly drops entire residual paths during training.
    """

    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True
    ) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # Generate a mask per sample, broadcast across all other dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)

        return x * random_tensor
