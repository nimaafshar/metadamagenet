from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as tf


def binary_dice_with_logits(logits: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    micro average soft dice score
    sigmoid activation function is applied
    loss is mean score for class 0 and 1

    :param logits: (N,1,H,W) float tensor
    :param true: (N,1,H,W) with values 0 and 1
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


def dice_loss_with_logits(logits: torch.Tensor, true: torch.Tensor, class_weights: torch.Tensor, eps=1e-8):
    """
    :param logits: torch.Tensor of size (N,num_classes,H,W) with type float
    :param true: torch.Tensor of size (N,1,H,W) with type long with values 0,...,C-1
    :param class_weights: torch.Tensor of size(num_classes,)
    :param eps: epsilon for numerical stability
    :return: loss, torch.Tensor
    """
    n, num_classes, h, w = logits.size()
    assert true.max() < num_classes and true.min() >= 0, "true values should be in 0,1,...,(num_classes-1)"
    assert class_weights.ndim() == 1 and class_weights.size(0) == num_classes, \
        "class_weights should be of size (num_classes,)"
    true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = tf.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_score = (2. * intersection / (cardinality + eps))
    loss = 1 - dice_score
    return torch.dot(loss, class_weights)


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, class_weights: Optional[List[float]], eps: float = 1e-8):
        super().__init__()
        assert num_classes > 1, "Non Binary Dice Loss is used with num_classes>0"
        self._num_classes: int = num_classes
        if class_weights is None:
            self._class_weights: torch.Tensor = torch.ones(num_classes)
        else:
            assert len(class_weights) == num_classes, "len(class_weights) should be equal to num_classes"
            self._class_weights: torch.Tensor = torch.tensor(class_weights, dtype=torch.float)
        self._eps: float = eps

    def forward(self, logits: torch.Tensor, outputs: torch.Tensor):
        return dice_loss_with_logits(logits, outputs,
                                     class_weights=self._class_weights.to(device=logits.device),
                                     eps=self._eps)
