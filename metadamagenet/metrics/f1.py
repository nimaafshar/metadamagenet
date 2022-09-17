import torch
from torchmetrics import F1Score as TorchMetricsF1Score

from .base import ImageMetric
from ..metrics import AverageMetric


class F1Score(ImageMetric):
    def __init__(self, num_classes: int):
        super().__init__()
        self.f1_metric = TorchMetricsF1Score(num_classes=num_classes, mdmc_average='samplewise')
        self._num_classes: int = num_classes
        self._average_metric: AverageMetric = AverageMetric

    def update_batch(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        output_labels: torch.LongTensor = outputs.argmax(dim=1)
        target_labels: torch.LongTensor = targets.argmax(dim=1)
        val: torch.Tensor = self.f1_metric(output_labels.flatten(start_dim=1),
                                           target_labels.flatten(start_dim=1)).item()
        self._average_metric.update(val.item(), count=outputs.size(0))
        return val

    def till_here(self) -> float:
        return self._average_metric.till_here()

    def status_till_here(self) -> str:
        return self._average_metric.status_till_here()

    def reset(self) -> None:
        self.f1_metric.reset()
        self._average_metric.reset()
