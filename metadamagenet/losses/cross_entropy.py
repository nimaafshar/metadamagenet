from typing import Optional

import torch
from torch import nn


class SegmentationCCE(nn.Module):
    def __init__(self, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.ce: nn.CrossEntropyLoss = nn.CrossEntropyLoss(weights)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(outputs, targets.argmax(dim=1))
