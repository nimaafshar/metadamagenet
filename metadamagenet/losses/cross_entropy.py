from typing import Optional

import torch
from torch import nn


class SegmentationCCE(nn.Module):
    def __init__(self, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.ce: nn.CrossEntropyLoss = nn.CrossEntropyLoss(weights)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param outputs: torch.Tensor of type float and shape (N,C,H,W)
        :param targets: torch.Tensor of type int and shape (N,1,H,W) containing values [0,C)
        :return:
        """
        return self.ce(outputs, targets.squeeze(dim=1))
