from typing import Any

import torch
from torchmetrics import Dice


class ScoreException(Exception):
    def __init__(self, message: str, **kwargs):
        super(ScoreException, self).__init__(message)
        self.data = kwargs


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
        # convert to float, so it can be detected as multi-label binary
        logits = (torch.argmax(preds, dim=1) > 0).float()
        targets_binary = (targets > 0).long()
        super().update(logits, targets_binary)


class DamageClassificationMetric(Dice):

    def __init__(self, eps: float = 1e-6, **kwargs: Any):
        super().__init__(num_classes=4,
                         average=None,
                         mdmc_average='global',
                         zero_division=1,  # is not working
                         **kwargs)
        self._eps: float = eps

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        :param preds: torch.Tensor of shape (N,5,H,W)
        :param targets: torch.Tensor of shape (N,H,W)
        """
        preds = preds.argmax(dim=1)
        preds = torch.where(preds != 0, preds, 1)
        super().update(preds[targets > 0] - 1, targets[targets > 0] - 1)

    def _harmonic_mean(self, scores):
        return scores.size(0) / torch.sum(1.0 / (scores + self._eps))

    def compute(self) -> torch.Tensor:
        sup = self.tp + self.fn
        scores = torch.nan_to_num(super().compute()[sup > 0], 1.)
        final_value = self._harmonic_mean(scores)
        if final_value.isnan():
            raise ScoreException("damage classification encountered nan",
                                 tp=self.tp,
                                 fn=self.fn,
                                 fp=self.fp,
                                 sup=sup,
                                 dice=super().compute(),
                                 scores=scores)
        return final_value
