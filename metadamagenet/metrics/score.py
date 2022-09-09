from typing import Tuple

import torch

from ..metrics import FloatMetric


class Score(FloatMetric):
    def __init__(self, *metrics: Tuple[str, FloatMetric, float]):
        super().__init__()
        self._metrics: Tuple[FloatMetric]
        self._weights: Tuple[float]
        self._names: Tuple[str]
        self._names, self._metrics, self._weights = zip(*metrics)

    def update(self, outputs_batch: torch.Tensor, targets_batch: torch.Tensor) -> None:
        weighted_sum = 0
        metric: FloatMetric
        for i, metric in enumerate(self._metrics):
            metric.update(outputs_batch, targets_batch)
            weighted_sum += self._weights[i] * metric.last
        self._update(weighted_sum, n=outputs_batch.size(0))

    def status(self) -> str:
        return f"weighted:{self._val:.4f} ({self._avg:.4f});" \
               ";".join(f"{self._names[i]}: {metric.status}" for i, metric in enumerate(self._metrics))

    def avg_status(self) -> str:
        return f"weighted:{self._avg:.4f};" \
               ";".join(f"{self._names[i]}: {metric.avg_status}" for i, metric in enumerate(self._metrics))
