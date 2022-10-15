from typing import Tuple

import torch
from torch import nn


class WeightedSum(nn.Module):
    """
    example:
    ```python
    loss = WeightedSum(
        (Loss1(), 1),
        (Loss2()), 10)
    )

    loss(logits,targets).backward()
    ```
    """

    def __init__(self, *weighted_losses: Tuple[nn.Module, float]):
        super().__init__()
        losses: Tuple[nn.Module]
        weights: Tuple[float]
        losses, weights = zip(*weighted_losses)
        self.losses: nn.ModuleList = nn.ModuleList(losses)
        self.register_buffer("weights", torch.FloatTensor(weights))

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weighted_loss: torch.Tensor = 0
        for i, loss in enumerate(self.losses):
            value: torch.Tensor = loss(outputs, targets)
            if value.isinf():
                print(f"{loss} encountered inf")
            weighted_loss += value * self.weights[i]
        return weighted_loss
