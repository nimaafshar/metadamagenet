from itertools import filterfalse
from typing import Iterable

import numpy as np
import torch
from torch import nn


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def mean(iterable: Iterable, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    i = iter(iterable)
    if ignore_nan:
        i = filterfalse(np.isnan, i)
    try:
        n = 1
        acc = next(i)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(i, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class WithSigmoid(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self._loss = loss

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss(torch.sigmoid(outputs), targets)
