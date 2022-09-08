from typing import Tuple, List

import torch
from torch import nn

from metadamagenet.utils import AverageMeter
from .monitored import MonitoredLoss


class ComboLoss(MonitoredLoss):
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
        self.losses: nn.ModuleList = nn.ModuleList(losses)
        self.register_buffer("weights", torch.FloatTensor(weights))
        self.register_buffer("names", names)
        self._monitor: bool = monitor
        if self._monitor:
            self._meters: List[AverageMeter] = [AverageMeter() for _ in len(names)]
            self._total_loss_meter = AverageMeter()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_values = [loss(outputs, targets) for loss in self.losses]
        weighted_loss = torch.dot(torch.tensor(loss_values), self.weights)
        if self._monitor:
            for i, loss_value in enumerate(loss_values):
                self._meters[i].update(loss_value.item(), outputs.size(0))
            self._total_loss_meter.update(weighted_loss.item(), outputs.size(0))

        return weighted_loss

    @property
    def monitored(self) -> bool:
        return self._monitor

    def last_values(self) -> str:
        if not self._monitor:
            return "not monitored"
        children_last_values = []
        for i, name in len(self.names):
            if isinstance(self.losses[i], MonitoredLoss) and self.losses[i].monitored:
                children_last_values.append(f"{name}:{self.losses[i].last_values}")
            else:
                children_last_values.append(f"{name}:{self._meters[i].val:.4f} ({self._meters[i].avg:.4f})")
        return f"Total: {self._total_loss_meter.val:.4f} [{'; '.join(children_last_values)}]"

    def aggregate(self) -> str:
        if not self._monitor:
            return "not monitored"
        children_aggregated = []
        for i, name in len(self.names):
            if isinstance(self.losses[i], MonitoredLoss) and self.losses[i].monitored:
                children_aggregated.append(f"{name}:{self.losses[i].aggregate()}")
            else:
                children_aggregated.append(f"{name}:{self._meters[i].avg:.4f}")
        return f"Total: {self._total_loss_meter.avg:.4f} [{'; '.join(children_aggregated)}]"
