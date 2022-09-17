import abc
from typing import Optional

import torch

from .wrapper import ModelWrapper, LocalizerModelWrapper, ClassifierModelWrapper
from ..models.unet import Unet
from ..models.unet import SeNet154Unet
from ..models.senet import SENet, senet154
from ..metrics import xview2, ImageMetric


class SeNet154Wrapper(ModelWrapper, abc.ABC):
    data_parallel = True

    def empty_unet(self) -> Unet:
        return SeNet154Unet(senet154(pretrained=None))

    def unet_with_pretrained_backbone(self, backbone: Optional[SENet] = None) -> Unet:
        if backbone is not None:
            return SeNet154Unet(backbone)
        return SeNet154Unet(senet154())


class SeNet154LocalizerWrapper(SeNet154Wrapper, LocalizerModelWrapper):
    model_name = "SeNet154UnetLocalizer"
    input_size = 480, 480


class SeNet154ClassifierWrapper(SeNet154Wrapper, ClassifierModelWrapper):
    model_name = "SeNet154UnetClassifier"
    input_size = 448, 448
    default_score: ImageMetric = xview2.classification_score
