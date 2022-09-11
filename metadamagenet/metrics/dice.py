import torch

from .base import FloatMetric
from ..losses.dice import dice_round, dice_batch


class DiceRound(FloatMetric):
    def __init__(self, channel: int = 0, inverse: bool = False):
        super().__init__()
        self._channel: int = channel
        self._inverse: bool = inverse

    def update(self, output_batch: torch.Tensor, targets_batch: torch.Tensor) -> None:
        """
        :param output_batch: (B,C,H,W) C >= 1
        :param targets_batch: same dimensions as output_batch
        """
        out = 1 - output_batch[:, self._channel, ...] if self._inverse else output_batch[:, self._channel, ...]
        tar = 1 - targets_batch[:, self._channel, ...] if self._inverse else targets_batch[:, self._channel, ...]
        dice_sc = 1 - dice_round(out, tar, per_image=True)
        self._update(float(dice_sc.mean()), output_batch.size(0))


class Dice(FloatMetric):
    def __init__(self, threshold: float, channel: int = 0, inverse: bool = False):
        super().__init__()
        self._threshold: float = threshold
        self._channel: int = channel
        self._inverse: bool = inverse

    def update(self, output_batch: torch.Tensor, targets_batch: torch.Tensor) -> None:
        """
        :param output_batch: (B,C,H,W) C >= channel
        :param targets_batch: same dimensions as output_batch
        """
        out = 1 - output_batch[:, self._channel, ...] if self._inverse else output_batch[:, self._channel, ...]
        tar = 1 - targets_batch[:, self._channel, ...] if self._inverse else targets_batch[:, self._channel, ...]
        dice_scores = dice_batch(out, tar > self._threshold)
        self._update(float(dice_scores.mean()), output_batch.size(0))
