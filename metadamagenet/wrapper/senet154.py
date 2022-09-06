import abc
from typing import Tuple

import torch
from torch import nn

from .wrapper import ModelWrapper, LocalizerModelWrapper, ClassifierModelWrapper
from metadamagenet.models.metadata import Metadata
from metadamagenet.models.unet import Unet
from metadamagenet.models.unet import SeNet154Unet
from metadamagenet.models.senet import SENet, senet154


class SeNet154Wrapper(ModelWrapper, abc.ABC):
    data_parallel = True

    def empty_unet(self) -> Unet:
        return SeNet154Unet(senet154(pretrained=None))

    def unet_with_pretrained_backbone(self, backbone: nn.Module) -> Unet:
        return SeNet154Unet(senet154())


class SeNet154LocalizerWrapper(SeNet154Wrapper, LocalizerModelWrapper):
    model_name = "SeNet154UnetLocalizer"
    input_size = (480, 480)


class SeNet154ClassifierWrapper(SeNet154Wrapper, ClassifierModelWrapper):
    model_name = "SeNet154UnetClassifier"
    input_size = 448, 448

    def apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)
