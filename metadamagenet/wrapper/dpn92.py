import abc
from typing import Tuple

import torch
from torch import nn

from .wrapper import ModelWrapper, LocalizerModelWrapper, ClassifierModelWrapper
from metadamagenet.models.unet import Unet
from metadamagenet.models.unet import Dpn92Unet
from metadamagenet.models.dpn import DPN, dpn92


class Dpn92Wrapper(ModelWrapper, abc.ABC):
    input_size = 512, 512

    def empty_unet(self) -> Unet:
        return Dpn92Unet(dpn92(pretrained=None))

    def unet_with_pretrained_backbone(self, backbone: nn.Module) -> Unet:
        return Dpn92Unet(dpn92())


class Dpn92LocalizerWrapper(Dpn92Wrapper, LocalizerModelWrapper):
    model_name = "Dpn92UnetLocalizer"
    data_parallel = False


class Dpn92ClassifierWrapper(Dpn92Wrapper, ClassifierModelWrapper):
    model_name = "Dpn92UnetClassifier"
    data_parallel = True

    def apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)
