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
        super().__init__(num_classes=5,
                         ignore_index=0,
                         zero_division=1.,
                         average=None,
                         **kwargs)

    def compute(self) -> torch.Tensor:
        class_scores = super().compute()
        return 1 / ((1 / class_scores[1:]).mean())
