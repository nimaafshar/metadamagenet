from typing import Tuple

import torch
from torch import nn


class ComboLoss(nn.Module):
    """
    example:
    ComboLoss(
        (StableBCELoss(), 1),
        (WithSigmoid(DiceLoss(per_image=False)), 1),
        (WithSigmoid(JaccardLoss(per_image=False)), 1),
        (LovaszLoss(per_image), 1),
        (WithSigmoid(LovaszLossSigmoid(per_image)), 1),
        (WithSigmoid(FocalLoss2d()), 1)
    )
    """

    def __init__(self, *weighted_losses: Tuple[nn.Module, float]):
        super().__init__()
        self._weighted_losses: Tuple[nn.Module, float] = weighted_losses

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for loss, weight in self._weighted_losses:
            total_loss += loss(outputs, targets) * weight
        return total_loss
