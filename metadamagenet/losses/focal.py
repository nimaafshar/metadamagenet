from torch import Tensor
from kornia.losses import BinaryFocalLossWithLogits


class BinaryFocalLoss(BinaryFocalLossWithLogits):
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return super(BinaryFocalLoss, self).forward(inputs.squeeze(1), targets)


"""
Focal Loss is removed because of numerical instability in fp16 training
use kornia.losses.focal
"""
