import torch

from .base import ImageMetric
from .average import AverageMetric
from ..losses.dice import dice_round, dice_batch


class DiceRound(ImageMetric):
    def __init__(self, channel: int = 0, inverse: bool = False):
        self._channel: int = channel
        self._inverse: bool = inverse
        self._average: AverageMetric = AverageMetric()

    def update_batch(self, output_batch: torch.Tensor, targets_batch: torch.Tensor) -> torch.Tensor:
        """
        :param output_batch: (B,C,H,W) C >= 1
        :param targets_batch: same dimensions as output_batch
        """
        out = 1 - output_batch[:, self._channel, ...] if self._inverse else output_batch[:, self._channel, ...]
        tar = 1 - targets_batch[:, self._channel, ...] if self._inverse else targets_batch[:, self._channel, ...]
        dice_sc: torch.Tensor = 1 - dice_round(out, tar, per_image=True)
        self._average.update(dice_sc.mean().item(), output_batch.size(0))
        return dice_sc.mean()

    @property
    def till_here(self) -> float:
        return self._average.till_here

    @property
    def status_till_here(self) -> str:
        return self._average.status_till_here

    def reset(self) -> None:
        self._average.reset()


class Dice(ImageMetric):
    def __init__(self, threshold: float, channel: int = 0, inverse: bool = False):
        super().__init__()
        self._threshold: float = threshold
        self._channel: int = channel
        self._inverse: bool = inverse
        self._average: AverageMetric = AverageMetric()

    def update_batch(self, output_batch: torch.Tensor, targets_batch: torch.Tensor) -> None:
        """
        :param output_batch: (B,C,H,W) C >= channel
        :param targets_batch: same dimensions as output_batch
        """
        out = 1 - output_batch[:, self._channel, ...] if self._inverse else output_batch[:, self._channel, ...]
        tar = 1 - targets_batch[:, self._channel, ...] if self._inverse else targets_batch[:, self._channel, ...]
        dice_scores: torch.Tensor = dice_batch(out, tar > self._threshold)
        self._average.update(dice_scores.mean().item(), output_batch.size(0))
        return dice_scores.mean()

    @property
    def till_here(self) -> float:
        return self._average.till_here

    @property
    def status_till_here(self) -> str:
        return self._average.status_till_here

    def reset(self) -> None:
        self._average.reset()
