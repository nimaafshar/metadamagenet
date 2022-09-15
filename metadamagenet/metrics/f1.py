import torch

from .base import ImageMetric
from ..losses.epsilon import eps


class F1Score(ImageMetric):
    def __init__(self, start_idx: int, end_idx: int, num_classes: int):
        """
        :param start_idx: index of start channel (inclusive)
        :param end_idx: index of end channel (exclusive)
        """
        super().__init__()
        self._start_idx: int = start_idx
        self._end_idx: int = end_idx
        self._channels: int = self._end_idx - self._start_idx
        self._f1_scores_sum: torch.Tensor = torch.zeros((self._channels,))
        self._f1_scores_average: torch.Tensor = torch.zeros((self._channels,))
        self._count: int = 0
        self._num_classes: int = num_classes

    def update_batch(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.one_hot(outputs.argmax(dim=1), num_classes=5).permute(0, 3, 1, 2)
        tp = torch.logical_and(targets == 1, out == 1).flatten(start_dim=2).sum(dim=2)
        fn = torch.logical_and(targets == 1, out != 1).flatten(start_dim=2).sum(dim=2)
        fp = torch.logical_and(targets != 1, out == 1).flatten(start_dim=2).sum(dim=2)
        f1_scores = (2 * tp / (2 * tp + fp + fn)).nan_to_num(nan=0).sum(dim=0)[self._start_idx:self._end_idx]
        self._f1_scores_sum += f1_scores * outputs.size(0)
        self._count += outputs.size(0)
        self._f1_scores_average = self._f1_scores_sum / self._count
        return self._channels / torch.sum(1.0 / (f1_scores + eps))

    def till_here(self) -> float:
        return (self._channels / torch.sum(1.0 / (self._f1_scores_average + eps))).item()

    def status_till_here(self) -> str:
        return f"total: {self.till_here():.5f}, {self._f1_scores_average}"

    def reset(self) -> None:
        self._count = 0
        self._f1_scores_average = 0.0
        self._f1_scores_sum = 0.0
