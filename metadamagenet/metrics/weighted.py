from typing import Tuple

import torch

from .base import ImageMetric
from .average import AverageMetric


class WeightedImageMetric(ImageMetric):
    def __init__(self, *metrics: Tuple[str, ImageMetric, float]):
        super().__init__()
        self._metrics: Tuple[ImageMetric]
        self._weights: Tuple[float]
        self._names: Tuple[str]
        self._names, self._metrics, self._weights = zip(*metrics)
        self._average: AverageMetric = AverageMetric()

    def update_batch(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        metric: ImageMetric
        weighted_sum: torch.Tensor = 0
        for i, metric in enumerate(self._metrics):
            weighted_sum += metric.update_batch(outputs, targets) * self._weights[i]
        self._average.update(weighted_sum.item(), count=outputs.size(0))
        return weighted_sum

    def till_here(self) -> float:
        return self._average.till_here()

    def status_till_here(self) -> str:
        return f"Weighted: {self._average.status_till_here()}[" + \
               ",".join((f"{self._names[i]}: {metric.status_till_here()}" for i, metric in enumerate(self._metrics))) \
               + "]"

    def reset(self) -> None:
        self._average.reset()
        for metric in self._metrics:
            metric.reset()
