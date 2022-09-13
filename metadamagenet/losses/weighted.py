from typing import Tuple

import torch
from torch import nn

from .base import MonitoredImageLoss, Monitored
from ..metrics.average import AverageMetric


class WeightedLoss(MonitoredImageLoss):
    """
    example:
    ComboLoss(
        ("BCE",StableBCELoss(), 1),
        ("DICE",WithSigmoid(DiceLoss(per_image=False)), 1),
        ("JACCARD",WithSigmoid(JaccardLoss(per_image=False)), 1),
        ("Lovasz",LovaszLoss(per_image), 1),
        ("Lovasz_sig",WithSigmoid(LovaszLossSigmoid(per_image)), 1),
        ("focal",WithSigmoid(FocalLoss2d()), 1)
    )
    """

    def __init__(self, *weighted_losses: Tuple[str, nn.Module, float], monitor: bool = False):
        super().__init__()
        losses: Tuple[nn.Module]
        weights: Tuple[float]
        names, losses, weights = zip(*weighted_losses)
        self.losses: nn.ModuleList[MonitoredImageLoss] = nn.ModuleList(
            [loss if isinstance(loss, MonitoredImageLoss) else Monitored(loss) for loss in losses]
        )
        self.register_buffer("weights", torch.FloatTensor(weights))
        self._names = names
        self._total_loss_meter: AverageMetric = AverageMetric()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weighted_loss: torch.Tensor = 0
        for i, loss in enumerate(self.losses):
            weighted_loss += loss(outputs, targets) * self.weights[i]
        self._total_loss_meter.update(weighted_loss.item(), outputs.size(0))
        return weighted_loss

    @property
    def till_here(self) -> float:
        return self._total_loss_meter.till_here

    @property
    def status_till_here(self) -> str:
        return f"Weighted: {self._total_loss_meter.status_till_here}[" + \
               ",".join((f"{self._names[i]}: {loss.status_till_here}" for i, loss in enumerate(self.losses))) + \
               "]"

    def reset(self) -> None:
        for loss in self.losses:
            loss.reset()
        self._total_loss_meter.reset()
