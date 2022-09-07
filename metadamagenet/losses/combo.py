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
        losses: Tuple[nn.Module]
        weights: Tuple[float]
        losses, weights = zip(*weighted_losses)
        self.losses: nn.ModuleList = nn.ModuleList(losses)
        self.register_buffer("weights", torch.FloatTensor(weights))

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_items = [loss(outputs, targets) for loss in self.losses]
        return torch.dot(torch.tensor(loss_items), self.weights)
