import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            num_warmup_steps: int | float,
            num_training_steps: int,
            last_epoch: int = -1,
            min_lr: float = 1e-8
        ) -> None:
        self.warmup_steps = num_warmup_steps if isinstance(num_warmup_steps, int) else int(num_warmup_steps * num_training_steps)
        self.total_steps = num_training_steps
        self.min_lr = min_lr

        # Store initial lr, min_lr, and lr_scale for each param group
        self.init_lrs = []
        self.min_lrs = []
        self.lr_scales = []
        for param_group in optimizer.param_groups:
            self.init_lrs.append(param_group.get('initial_lr', param_group['lr']))  #could be kept for later use when doing per group lrs
            self.min_lrs.append(param_group.get('min_lr', self.min_lr))  # could be kept for later use when doing per group lrs
            self.lr_scales.append(param_group.get('lr_scale', 1.0))
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list:
        step = max(0, self.last_epoch)
        lrs = []
        for idx, (init_lr, min_lr, lr_scale) in enumerate(zip(self.init_lrs, self.min_lrs, self.lr_scales)):
            if step < self.warmup_steps:
                lr = init_lr * step / float(max(1, self.warmup_steps))
            else:
                progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                lr = min_lr + (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * progress))
            lr *= lr_scale
            lrs.append(lr)
        return lrs
