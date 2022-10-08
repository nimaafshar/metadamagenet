from typing import Any

import torch
from torchmetrics import Dice


class DamageLocalizationMetric(Dice):

    def __init__(self, **kwargs: Any):
        super().__init__(multiclass=True,
                         num_classes=2,
                         threshold=0.5,
                         zero_division=0,
                         ignore_index=0,
                         average='macro',
                         mdmc_average='global',
                         **kwargs)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param preds: torch.Tensor of shape (N,5,H,W)
        :param targets: torch.Tensor of shape (N,H,W)
        """
        logits_zero = preds[:, 0:1, ...]
        logits_nonzero = 1 - logits_zero
        logits = torch.cat((logits_zero, logits_nonzero), dim=1)
        targets_binary = (targets > 0).long()
        super().update(logits, targets_binary)


class DamageClassificationMetric(Dice):
    def __init__(self, **kwargs: Any):
        super().__init__(multiclass=True,
                         num_classes=4,
                         threshold=.5,
                         zero_division=0,
                         average=None,
                         **kwargs)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        :param preds: torch.Tensor of shape (N,C,H,W)
        :param targets: torch.Tensor of shape (N,H,W)
        """
        preds = preds.argmax(dim=1, keepdim=True)
        assert preds.size() == targets.size()
        preds_filtered = preds[(preds > 0) & (targets > 0)]
        targets_filtered = targets[(preds > 0) & (targets > 0)]
        super().update(preds_filtered - 1, targets_filtered - 1)

    def compute(self) -> torch.Tensor:
        return ((1 / super().compute()).mean(dim=0)) ** (-1)
