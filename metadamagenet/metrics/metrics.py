import torch

from .base import FloatMetric
from metadamagenet.losses.dice import dice_round, dice_batch
from metadamagenet.losses.epsilon import eps


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


class F1Score(FloatMetric):
    def __init__(self):
        super().__init__()
        self._f1_scores = torch.zeros((4,))

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        label_outputs = outputs[:, 1:, ...].argmax(dim=1)
        label_targets = targets[:, 1:, ...].argmax(dim=1)
        # filter targets with masks
        targ = label_targets * (targets[:, 0, ...] > 0)
        # TODO: filter with preloaded loc mask
        # filter predictions with masks
        out = label_outputs * (targets[:, 0, ...] > 0)
        target_one_hot = torch.nn.functional.one_hot(targ.flatten())
        out_one_hot = torch.nn.functional.one_hot(out.flatten())
        tp = torch.logical_and(out_one_hot == 1, target_one_hot == 1).sum(dim=0)
        fn = torch.logical_and(out_one_hot != 1, target_one_hot == 1).sum(dim=0)
        fp = torch.logical_and(out_one_hot == 1, target_one_hot != 1).sum(dim=0)
        f1_scores = 2 * tp / (2 * tp + fp + fn)
        self._f1_scores += f1_scores * outputs.size(0)
        f1_final = 4 / torch.sum(1.0 / (f1_scores + eps))
        self._update(float(f1_final), outputs.size(0))

    def avg_status(self) -> str:
        avg_f1_scores = self._f1_scores / self.count
        return f'Total: {self.avg},' \
               f'[{avg_f1_scores[0]:.4f},' \
               f'{avg_f1_scores[1]:.4f},' \
               f'{avg_f1_scores[2]:.4f},' \
               f'{avg_f1_scores[3]:.4f}]'
