import torch

from .base import ImageMetric
from ..losses.epsilon import eps


class F1Score(ImageMetric):
    def __init__(self, start_idx: int, end_idx: int):
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

    def update_batch(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        label_outputs = outputs.argmax(dim=1)
        label_targets = targets.argmax(dim=1)
        targ = label_targets * (targets[:, 0, ...] > 0)  # filter targets with masks
        # IDEA: maybe filter with preloaded loc mask
        out = label_outputs * (targets[:, 0, ...] > 0)  # filter predictions with masks
        target_one_hot = torch.nn.functional.one_hot(targ.flatten())
        out_one_hot = torch.nn.functional.one_hot(out.flatten())
        tp = torch.logical_and(out_one_hot == 1, target_one_hot == 1).sum(dim=0)[self._start_idx:self._end_idx]
        fn = torch.logical_and(out_one_hot != 1, target_one_hot == 1).sum(dim=0)[self._start_idx:self._end_idx]
        fp = torch.logical_and(out_one_hot == 1, target_one_hot != 1).sum(dim=0)[self._start_idx:self._end_idx]
        f1_scores = 2 * tp / (2 * tp + fp + fn)
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
