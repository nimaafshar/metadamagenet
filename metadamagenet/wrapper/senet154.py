import abc

import torch
from torch import nn

from .wrapper import ModelWrapper, LocalizerModelWrapper, ClassifierModelWrapper
from ..models.unet import Unet
from ..models.unet import SeNet154Unet
from ..models.senet import SENet, senet154
from ..metrics import Score, F1Score


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
    default_score = Score(
        ("LocF1", F1Score(start_idx=0, end_idx=1), 0.3),
        ("F1", F1Score(start_idx=1), 0.7)
    )

    def apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)
