import torch

from .bce import StableBCELoss
from .dice import DiceLoss
from .jaccard import JaccardLoss
from .lovasz import LovaszLoss, LovaszLossSigmoid
from .focal import FocalLoss2d


class ComboLoss(torch.nn.Module):
    def __init__(self, weights, per_image=False):
        super().__init__()
        self.weights = weights
        self.bce = StableBCELoss()
        self.dice = DiceLoss(per_image=False)
        self.jaccard = JaccardLoss(per_image=False)
        self.lovasz = LovaszLoss(per_image=per_image)
        self.lovasz_sigmoid = LovaszLossSigmoid(per_image=per_image)
        self.focal = FocalLoss2d()
        self.mapping = {'bce': self.bce,
                        'dice': self.dice,
                        'focal': self.focal,
                        'jaccard': self.jaccard,
                        'lovasz': self.lovasz,
                        'lovasz_sigmoid': self.lovasz_sigmoid}
        self.expect_sigmoid = {'dice', 'focal', 'jaccard', 'lovasz_sigmoid'}
        self.values = {}

    def forward(self, outputs, targets):
        loss = 0
        weights = self.weights
        sigmoid_input = torch.sigmoid(outputs)
        for k, v in weights.items():
            if not v:
                continue
            val = self.mapping[k](sigmoid_input if k in self.expect_sigmoid else outputs, targets)
            self.values[k] = val
            loss += self.weights[k] * val
        return loss
