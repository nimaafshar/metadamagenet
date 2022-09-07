from typing import Tuple

import torch
from torch import nn


class ChanneledLoss(nn.Module):
    def __init__(self, *channel_losses: Tuple[nn.Module, float]):
        super().__init__()
        self._channel_losses: Tuple[nn.Module, float] = channel_losses

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert outputs.size(1) == targets.size(1) == len(self._channel_losses), \
            "number of channels with number of losses do not match"
        total_loss = 0
        for i, (channel_loss, weight) in enumerate(self._channel_losses):
            total_loss += channel_loss(outputs[:, i, ...], targets[:, i, ...]) * weight
        return total_loss
