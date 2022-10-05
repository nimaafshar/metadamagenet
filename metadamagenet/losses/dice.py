from typing import List

import torch
from torch import nn
from torch.nn import functional as tf


def binary_dice_with_logits(logits: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    micro average soft dice score
    sigmoid activation function is applied
    loss is mean score for class 0 and 1

    :param logits: (N,1,H,W) float tensor
    :param true: (
    :param eps: epsilon for numerical stability
    :return: loss, torch.Tensor
    """
    assert logits.shape[1] == 1, "num_classes (logits.shape[1]) should be 1"
    true_1_hot = torch.eye(2)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    true_1_hot_f = true_1_hot[:, 0:1, :, :]
    true_1_hot_s = true_1_hot[:, 1:2, :, :]
    true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
    pos_prob = torch.sigmoid(logits)
    neg_prob = 1 - pos_prob
    probas = torch.cat([pos_prob, neg_prob], dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_score = (2. * intersection / (cardinality + eps)).mean()
    return 1 - dice_score


class BinaryDiceLossWithLogits(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self._eps: float = eps

    def forward(self, logits: torch.Tensor, outputs: torch.Tensor):
        return binary_dice_with_logits(logits, outputs, self._eps)


class DiceLoss(nn.Module):
    def __init__(self, class_weights: List[float]):
        super().__init__()
        raise NotImplementedError()
