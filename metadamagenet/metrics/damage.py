from typing import Any

import torch
from torchmetrics import Dice


class DamageLocalizationMetric(Dice):

    def __init__(self, **kwargs: Any):
        super().__init__(multiclass=False,
                         average='micro',
                         mdmc_reduce='global',
                         **kwargs)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param preds: torch.Tensor of shape (N,C,H,W)
        :param targets: torch.Tensor of shape (N,H,W)
        """
        # convert to float so it can be detected as multi-label binary
        logits = (torch.argmax(preds, dim=1) > 0).float()
        targets_binary = (targets > 0).long()
        super().update(logits, targets_binary)


class DamageClassificationMetric(Dice):
    def __init__(self, **kwargs: Any):
        super().__init__(num_classes=5,
                         ignore_index=0,
                         zero_division=1.,
                         average=None,
                         **kwargs)

    # eps for numerical stability
    def compute(self) -> torch.Tensor:
        class_scores = super().compute()
        return 1 / ((1 / class_scores[1:]).mean())
