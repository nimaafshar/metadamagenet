from typing import Literal, Optional, List

import torch
from torch import nn
import torch.nn.functional as tf


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2, eps: float = 1e-8, validate_inputs: bool = False):
        super().__init__()
        self._gamma = gamma
        self._eps: float = eps
        self._validate_inputs: bool = validate_inputs

    def _validate(self, outputs: torch.Tensor, targets: torch.Tensor):
        n, c, h, w = outputs.size()
        assert c == 1, f"outputs should have be of the shape (N,1,H,W). got {outputs.shape}"
        n2, h2, w2 = targets.size()
        assert n == n2 and h == h2 and w == w2, \
            f"outputs.shape is {outputs.shape}, expected targets to have the shape of ({n},{h},{w})." \
            f" got {targets.shape}"
        assert torch.all((targets == 1) | (targets == 0)), "targets is not binary"

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        outputs: float, (N,1,H,W),
        targets: binary, (N,H,W)
        """
        if self._validate_inputs:
            self._validate(outputs, targets)
        outputs = torch.sigmoid(outputs.to(memory_format=torch.contiguous_format))
        targets = targets.unsqueeze(1).to(memory_format=torch.contiguous_format).to(dtype=outputs.dtype)
        outputs = torch.clamp(outputs, self._eps, 1. - self._eps)
        targets = torch.clamp(targets, self._eps, 1. - self._eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self._gamma * torch.log(pt)).mean()


class FocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 gamma: float = 2.,
                 class_weights: Optional[List[float]] = None,
                 activation: Literal['softmax', 'sigmoid'] = 'sigmoid',
                 eps: float = 1e-8,
                 validate_inputs: bool = False):
        super().__init__()
        assert num_classes > 1, "Non Binary Dice Loss is used with num_classes>0"
        self._num_classes: int = num_classes

        if class_weights is None:
            self.register_buffer('class_weights', torch.ones(num_classes, dtype=torch.float))
        else:
            assert len(class_weights) == num_classes, "len(class_weights) should be equal to num_classes"
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float))
        self._gamma: float = gamma
        self._eps: float = eps
        self._validate_inputs: bool = validate_inputs
        if activation not in ('softmax', 'sigmoid'):
            raise ValueError(f"unsupported activation {activation}")
        self._activation: Literal['softmax', 'sigmoid'] = activation

    def _validate(self, outputs: torch.Tensor, targets: torch.Tensor):
        n, c, h, w = outputs.size()
        n2, h2, w2 = targets.size()
        assert c == self._num_classes, "outputs.size(1) should be equal to num_classes"
        assert n == n2 and h == h2 and w == w2, \
            f"outputs.shape is {outputs.shape}, expected targets to have the shape of ({n},{h},{w})." \
            f" got {outputs.shape}"
        assert targets.dtype == torch.long \
               and targets.min() >= 0 \
               and targets.max() < c, "long is not long tensor with values (0 - C-1)"

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        outputs: float, (N,C,H,W),
        targets: long, (N,H,W) with values (0 - C-1)
        """
        if self._validate_inputs:
            self._validate(outputs, targets)

        outputs = outputs.to(memory_format=torch.contiguous_format)
        if self._activation == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        elif self._activation == 'softmax':
            outputs = torch.softmax(outputs, dim=1)

        targets = tf.one_hot(targets, num_classes=self._num_classes).permute(0, 3, 1, 2)
        targets = targets.to(memory_format=torch.contiguous_format).to(dtype=outputs.dtype)
        dims = (0, 2, 3)
        outputs = torch.clamp(outputs, self._eps, 1. - self._eps)
        targets = torch.clamp(targets, self._eps, 1. - self._eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        losses = torch.mean(-(1. - pt) ** self._gamma * torch.log(pt), dim=dims)
        return torch.dot(losses, self.class_weights.to(device=losses.device))
