import torch
from torch import Tensor, nn


class AsymmetricLossMultiLabel(nn.Module):
    def __init__(self,
                 gamma_neg: int = 4,
                 gamma_pos: int = 0,
                 clip: float = 0.05,
                 eps: float = 1e-8,
                 disable_torch_grad_focal_loss: bool = True) -> None:
        super(AsymmetricLossMultiLabel, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            prev_grad = torch.is_grad_enabled()
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg, self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                # Restore the caller's grad mode; hard-setting True here would
                # clobber an outer torch.no_grad() (e.g. validation) and leak
                # the autograd graph of every subsequent forward.
                torch.set_grad_enabled(prev_grad)
            self.loss *= self.asymmetric_w

        return -self.loss.mean(dim=1).mean()
