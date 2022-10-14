from typing import Optional, List, Literal

import torch
from torch import nn
from torch.nn import functional as tf


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-8, validate_inputs: bool = False):
        super().__init__()
        self._smooth: float = smooth
        self._validate_inputs: bool = validate_inputs

    def _validate(self, logits: torch.Tensor, targets: torch.Tensor):
        n, c, h, w = logits.size()
        assert c == 1, f"logits should have be of the shape (N,1,H,W). got {logits.shape}"
        n2, h2, w2 = targets.size()
        assert n == n2 and h == h2 and w == w2, \
            f"logits.shape is {logits.shape}, expected targets to have the shape of ({n},{h},{w}). got {targets.shape}"
        assert torch.all((targets == 1) | (targets == 0)), "targets is not binary"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: float, (N,1,H,W),
        targets: binary, (N,H,W)
        """
        if self._validate_inputs:
            self._validate(logits, targets)
        logits_cont = torch.sigmoid(logits.to(memory_format=torch.contiguous_format))
        targets_cont = targets.unsqueeze(1).to(memory_format=torch.contiguous_format).to(dtype=logits_cont.dtype)
        intersection = torch.sum(logits_cont * targets_cont)
        union = torch.sum(logits_cont) + torch.sum(targets_cont)
        dice_coefficient = (2 * intersection + self._smooth) / (union + self._smooth)
        return 1 - dice_coefficient


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int,
                 class_weights: Optional[List[float]] = None,
                 activation: Literal['softmax', 'sigmoid'] = 'sigmoid',
                 smooth: float = 1e-8,
                 validate_inputs: bool = False):
        super().__init__()
        assert num_classes > 1, "Non Binary Dice Loss is used with num_classes>0"
        self._num_classes: int = num_classes

        if class_weights is None:
            self.register_buffer('class_weights', torch.ones(num_classes, dtype=torch.float))
        else:
            assert len(class_weights) == num_classes, "len(class_weights) should be equal to num_classes"
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float))

        self._smooth: float = smooth
        self._validate_inputs: bool = validate_inputs
        if activation not in ('softmax', 'sigmoid'):
            raise ValueError(f"unsupported activation {activation}")
        self._activation: Literal['softmax', 'sigmoid'] = activation

    def _validate(self, logits: torch.Tensor, targets: torch.Tensor):
        n, c, h, w = logits.size()
        n2, h2, w2 = targets.size()
        assert c == self._num_classes, "logits.size(1) should be equal to num_classes"
        assert n == n2 and h == h2 and w == w2, \
            f"logits.shape is {logits.shape}, expected targets to have the shape of ({n},{h},{w}). got {targets.shape}"
        assert targets.dtype == torch.long \
               and targets.min() >= 0 \
               and targets.max() < c, "long is not long tensor with values (0 - C-1)"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: float, (N,C,H,W),
        targets: long, (N,H,W) with values (0 - C-1)
        """
        if self._validate_inputs:
            self._validate(logits, targets)
        logits_cont = logits.to(memory_format=torch.contiguous_format)
        if self._activation == 'sigmoid':
            logits_cont = torch.sigmoid(logits_cont)
        elif self._activation == 'softmax':
            logits_cont = torch.softmax(logits_cont, dim=1)

        targets_onehot = tf.one_hot(targets, num_classes=self._num_classes).permute(0, 3, 1, 2)
        targets_cont = targets_onehot.to(memory_format=torch.contiguous_format).to(dtype=logits_cont.dtype)
        dims = (0, 2, 3)
        intersection = torch.sum(logits_cont * targets_cont, dim=dims)
        union = torch.sum(logits_cont, dim=dims) + torch.sum(targets_cont, dim=dims)
        dice_coefficients = (2 * intersection + self._smooth) / (union + self._smooth)
        return torch.dot((1 - dice_coefficients), self.class_weights.to(device=dice_coefficients.device))
