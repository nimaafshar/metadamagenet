from typing import Tuple

import torch
from torch import nn


class ChanneledLoss(nn.Module):
    def __init__(self, *channel_losses: Tuple[nn.Module, float]):
        super().__init__()
        losses: Tuple[nn.Module]
        weights: Tuple[float]
        losses, weights = zip(*channel_losses)
        self.losses: nn.ModuleList = nn.ModuleList(losses)
        self.register_buffer("weights", torch.FloatTensor(weights))

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert outputs.size(1) == targets.size(1) == len(self.losses), \
            "number of channels with number of losses do not match"

        loss_values = [loss(outputs[:, i, ...], targets[:, i, ...]) for i, loss in enumerate(self.losses)]
        return torch.dot(torch.tensor(loss_values), self.weights)
