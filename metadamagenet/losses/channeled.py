from typing import Tuple

import torch
from torch import nn

from .base import MonitoredImageLoss, Monitored
from ..metrics.average import AverageMetric


class ChanneledLoss(MonitoredImageLoss):
    def __init__(self, *channel_losses: Tuple[nn.Module, float]):
        super().__init__()
        losses: Tuple[nn.Module]
        weights: Tuple[float]
        losses, weights = zip(*channel_losses)
        self.losses: nn.ModuleList[MonitoredImageLoss] = nn.ModuleList(
            [loss if isinstance(loss, MonitoredImageLoss) else Monitored(loss) for loss in losses]
        )
        self.register_buffer("weights", torch.FloatTensor(weights))
        self._total_loss_meter: AverageMetric = AverageMetric()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert outputs.size(1) == targets.size(1) == len(self.losses), \
            f"number of channels ({outputs.size(1)}) and number of losses ({len(self.losses)}) do not match"
        weighted_loss: torch.Tensor = 0
        for i, loss in enumerate(self.losses):
            weighted_loss += loss(outputs[:, i, ...], targets[:, i, ...]) * self.weights[i]
        self._total_loss_meter.update(weighted_loss.item(), outputs.size(0))
        return weighted_loss

    def till_here(self) -> float:
        return self._total_loss_meter.till_here()

    def status_till_here(self) -> str:
        return f"Weighted: {self._total_loss_meter.status_till_here()}[" + \
               ",".join((f"{loss.status_till_here()}" for loss in self.losses)) + \
               "]"

    def reset(self) -> None:
        for loss in self.losses:
            loss.reset()
        self._total_loss_meter.reset()
