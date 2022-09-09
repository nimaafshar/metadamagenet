import torch

from .base import FloatMetric
from metadamagenet.losses.dice import dice_round, dice_batch


class DiceRound(FloatMetric):
    def __init__(self, channel: int = 0):
        super().__init__()
        self._channel: int = channel

    def update(self, output_batch: torch.Tensor, targets_batch: torch.Tensor) -> None:
        """
        :param output_batch: (B,C,H,W) C >= 1
        :param targets_batch: same dimentions as output_batch
        """
        # msk_0_probs = torch.sigmoid(output_batch[:, 0, ...])
        dice_sc = 1 - dice_round(output_batch[:, self._channel, ...],
                                 targets_batch[:, self._channel, ...], per_image=True)
        self._update(float(dice_sc.mean()), output_batch.size(0))


class Dice(FloatMetric):
    def __init__(self, threshold: float, channel: int = 0):
        super().__init__()
        self._threshold: float = threshold
        self._channel: int = channel

    def update(self, output_batch: torch.Tensor, targets_batch: torch.Tensor) -> None:
        """
        :param output_batch: (B,C,H,W) C >= channel
        :param targets_batch: same dimensions as output_batch
        """
        # msk_0_probs = torch.sigmoid(output_batch[:, 0, ...])
        dice_scores = dice_batch(output_batch[:, self._channel, ...],
                                 targets_batch[:, self._channel, ...] > self._threshold)
        self._update(float(dice_scores.mean()), output_batch.size(0))
