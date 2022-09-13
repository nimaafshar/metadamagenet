import abc

import torch
from torch import nn

from ..metrics import Metric, AverageMetric


class MonitoredImageLoss(nn.Module, Metric, abc.ABC):
    """
    every loss that implements Metric interface is considered a monitored loss
    """
    pass


class Monitored(MonitoredImageLoss):
    """
    a wrapper to monitor usual losses
    """

    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss: nn.Module = loss
        self._average: AverageMetric = AverageMetric()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        value: torch.Tensor = self.loss(outputs, targets)
        self._average.update(value.item(), outputs.size(0))
        return value

    def till_here(self) -> float:
        return self._average.till_here

    def status_till_here(self) -> str:
        return self._average.status_till_here

    def reset(self) -> None:
        self._average.reset()
