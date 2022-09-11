from typing import Tuple, List

import torch
from torch import nn

from .monitored import MonitoredLoss
from ..metrics import AverageMeter


class ChanneledLoss(MonitoredLoss):
    def __init__(self, *channel_losses: Tuple[nn.Module, float], monitored: bool = False):
        super().__init__()
        losses: Tuple[nn.Module]
        weights: Tuple[float]
        losses, weights = zip(*channel_losses)
        self.losses: nn.ModuleList = nn.ModuleList(losses)
        self.register_buffer("weights", torch.FloatTensor(weights))
        self._monitor: bool = monitored
        if self._monitor:
            self._meters: List[AverageMeter] = [AverageMeter() for _ in len(losses)]
            self._total_loss_meter: AverageMeter = AverageMeter()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert outputs.size(1) == targets.size(1) == len(self.losses), \
            "number of channels with number of losses do not match"

        loss_items: List = []
        weighted_loss = 0
        for i, loss in enumerate(self._losses):
            value = loss(outputs[:, i, ...], targets[:, i, ...])
            weighted_loss = value * self._weights[i]
            loss_items.append(value.item())

        if self._monitor:
            for i, loss_item in enumerate(loss_items):
                self._meters[i].update(loss_item, outputs.size(0))
            self._total_loss_meter.update(weighted_loss.item(), outputs.size(0))

        return weighted_loss

    @property
    def monitored(self) -> bool:
        return self._monitor

    def last_values(self) -> str:
        if not self._monitor:
            return "not monitored"
        children_last_values = []
        for i, name in len(self._meters):
            if isinstance(self.losses[i], MonitoredLoss) and self.losses[i].monitored:
                children_last_values.append(self.losses[i].last_values)
            else:
                children_last_values.append(f"{self._meters[i].last:.4f} ({self._meters[i].avg:.4f})")
        return f"Total: {self._total_loss_meter.last:.4f} [{'; '.join(children_last_values)}]"

    def aggregate(self) -> str:
        if not self._monitor:
            return "not monitored"
        children_aggregated = []
        for i, name in len(self._meters):
            if isinstance(self.losses[i], MonitoredLoss) and self.losses[i].monitored:
                children_aggregated.append(self.losses[i].aggregate())
            else:
                children_aggregated.append(f"{self._meters[i].avg:.4f}")
        result: str = f"Total: {self._total_loss_meter.avg:.4f} [{'; '.join(children_aggregated)}]"
        for meter in self._meters:
            meter.reset()
        self._total_loss_meter.reset()
        return result
