import torch
from torchmetrics import F1Score as TorchMetricsF1Score

from .base import ImageMetric
from ..metrics import AverageMetric

eps = 1e-6


class DamageF1Score(ImageMetric, torch.nn.Module):
    def __init__(self):
        """
        this score filters output damage masks with targets localization mask. \
        this essentially removes the localization part of this metric and only the classification part remains. \
        this makes this score almost equal to one described at xview2 scoring
        TODO: filter output mask with a localization mask predicted by the respective localization model
        """
        super().__init__()
        self.f1_metric: torch.nn.Module = TorchMetricsF1Score(num_classes=5, average=None)
        self._overall: AverageMetric = AverageMetric()  # harmonic mean of classes
        self._undamaged: AverageMetric = AverageMetric()  # class 1
        self._minor: AverageMetric = AverageMetric()  # class 2
        self._major: AverageMetric = AverageMetric()  # class 3
        self._destroyed: AverageMetric = AverageMetric()  # class 4

    def update_batch(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size: int = outputs.size(0)
        target_labels: torch.LongTensor = targets.argmax(dim=1)
        output_labels: torch.LongTensor = outputs.argmax(dim=1)
        f1_scores: torch.FloatTensor = self.f1_metric(output_labels[target_labels > 0],
                                                      target_labels[target_labels > 0])  # returns tensor of shape (5,)
        f1_scores: torch.FloatTensor = torch.nan_to_num(f1_scores, nan=0.0)
        overall_f1_score: torch.Tensor = (4 / torch.sum(1 / (f1_scores[1:] + eps)))
        self._undamaged.update(f1_scores[1].item(), batch_size)
        self._minor.update(f1_scores[2].item(), batch_size)
        self._major.update(f1_scores[3].item(), batch_size)
        self._destroyed.update(f1_scores[4].item(), batch_size)
        self._overall.update(overall_f1_score.item(), batch_size)
        return overall_f1_score

    def till_here(self) -> float:
        return self._overall.till_here()

    def status_till_here(self) -> str:
        return f"{self._overall.status_till_here()}" \
               f"[{self._undamaged.status_till_here()},{self._minor.status_till_here()}," \
               f"{self._major.status_till_here()},{self._destroyed.status_till_here()}]"

    def reset(self) -> None:
        self.f1_metric.reset()
        self._overall.reset()
        self._undamaged.reset()
        self._minor.reset()
        self._major.reset()
        self._destroyed.reset()


class LocalizationF1Score(ImageMetric, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f1_metric: torch.nn.Module = TorchMetricsF1Score(num_classes=2, mdmc_average='samplewise')
        self._avg: AverageMetric = AverageMetric()

    def update_batch(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        out_loc_labels: torch.LongTensor = (outputs.argmax(dim=1) > 0).long()
        target_loc_labels: torch.LongTensor = (targets.argmax(dim=1) > 0).long()
        val: torch.FloatTensor = self.f1_metric(out_loc_labels.flatten(start_dim=1),
                                                target_loc_labels.flatten(start_dim=1))
        val: torch.FloatTensor = torch.nan_to_num(val, nan=0.0)
        self._avg.update(val.item(), count=outputs.size(0))
        return val

    def till_here(self) -> float:
        return self._avg.till_here()

    def status_till_here(self) -> str:
        return self._avg.status_till_here()

    def reset(self) -> None:
        self.f1_metric.reset()
        self._avg.reset()
