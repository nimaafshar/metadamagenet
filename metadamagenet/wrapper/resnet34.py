import abc
from typing import Tuple

import torch
from torchvision.models import resnet34
from torch import nn

from .wrapper import ModelWrapper, LocalizerModelWrapper, ClassifierModelWrapper
from metadamagenet.models.unet import Unet
from metadamagenet.models.metadata import Metadata
from metadamagenet.models.checkpoint import Checkpoint
from metadamagenet.models.manager import Manager
from metadamagenet.models.unet import Resnet34Unet
from torchvision.models import ResNet34_Weights


class Resnet34Wrapper(ModelWrapper, abc.ABC):
    unet_class = Resnet34Unet
    data_parallel = True

    def empty_unet(self) -> Unet:
        return Resnet34Unet(resnet34(weights=None))

    def unet_with_pretrained_backbone(self, backbone: nn.Module) -> Unet:
        return Resnet34Unet(resnet34(weights=ResNet34_Weights.DEFAULT))


class Resnet34LocalizerWrapper(Resnet34Wrapper, LocalizerModelWrapper):
    model_name = "Resnet34UnetLocalizer"
    input_size = (736, 736)


class Resnet34ClassifierWrapper(Resnet34Wrapper, ClassifierModelWrapper):
    model_name = "Resnet34UnetClassifier"
    input_size = (608, 608)

    def apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
