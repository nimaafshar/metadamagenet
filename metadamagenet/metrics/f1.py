from typing import Optional

import torch

from .base import FloatMetric
from metadamagenet.losses.epsilon import eps


class F1Score(FloatMetric):
    def __init__(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None):
        super().__init__()
        self._start_idx: Optional[int] = start_idx
        self._end_idx: Optional[int] = end_idx
        self._f1_scores = torch.zeros((4,))

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        label_outputs = outputs[:, self._start_idx:self._end_idx, ...].argmax(dim=1)
        label_targets = targets[:, self._start_idx:self._end_idx, ...].argmax(dim=1)
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
