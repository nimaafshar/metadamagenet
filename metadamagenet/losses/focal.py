from typing import Literal, Optional
import torch
from torch import nn

import torch.nn.functional as tf


class BinaryFocalLoss2d(nn.Module):
    def __init__(self, gamma=2., eps=1e-8, reduction: Literal['sum', 'mean', None] = 'mean',
                 class_weight: Optional[torch.Tensor] = None):
        super(BinaryFocalLoss2d, self).__init__()
        self._gamma: float = gamma
        self._eps: float = eps
        self._reduction: Literal['sum', 'mean', None] = reduction
        self._class_weight: Optional[torch.Tensor] = class_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target = targets.view(-1, 1).long()
        if self._class_weight is None:
            class_weight = [1] * 2  # [1/C]*C
        else:
            class_weight = self._class_weight
        prob = torch.sigmoid(logits)
        prob = prob.view(-1, 1)
        prob = torch.cat((1 - prob, prob), 1)
        select = torch.FloatTensor(len(prob), 2).zero_().cuda()
        select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)
        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, self._eps, 1 - self._eps)
        batch_loss = -class_weight * (torch.pow((1 - prob), self._gamma)) * prob.log()
        if self._reduction is None:
            return batch_loss
        elif self._reduction == 'mean':
            return batch_loss.mean()
        elif self._reduction == 'sum':
            return batch_loss.sum()
        else:
            raise NotImplementedError(f'invalid reduction {self._reduction}')


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2., size_average=True, eps=1e-8, reduction: Literal['sum', 'mean', None] = 'mean',
                 class_weight: Optional[torch.Tensor] = None):
        super(FocalLoss2d, self).__init__()
        self._gamma: float = gamma
        self._eps: float = eps
        self._reduction: Literal['sum', 'mean', None] = reduction
        self._class_weight: Optional[torch.Tensor] = class_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        target = targets.view(-1, 1).long()
        B, C, H, W = logits.size()
        if self._class_weight is None:
            class_weight = [1] * C  # [1/C]*C
        else:
            class_weight = self._class_weight
        logit = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        prob = tf.softmax(logit, 1)
        select = torch.FloatTensor(len(prob), C).zero_().cuda()
        select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)
        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, self._eps, 1 - self._eps)
        batch_loss = -class_weight * (torch.pow((1 - prob), self._gamma)) * prob.log()
        if self._reduction is None:
            return batch_loss
        elif self._reduction == 'mean':
            return batch_loss.mean()
        elif self._reduction == 'sum':
            return batch_loss.sum()
        else:
            raise NotImplementedError(f'invalid reduction {self._reduction}')
