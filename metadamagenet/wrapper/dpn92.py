import abc
from typing import Optional

import torch

from .wrapper import ModelWrapper, LocalizerModelWrapper, ClassifierModelWrapper
from ..models.unet import Unet
from ..models.unet import Dpn92Unet
from ..models.dpn import DPN, dpn92
from ..metrics import WeightedImageMetric, F1Score


class Dpn92Wrapper(ModelWrapper, abc.ABC):
    input_size = 512, 512

    def empty_unet(self) -> Unet:
        return Dpn92Unet(dpn92(pretrained=None))

    def unet_with_pretrained_backbone(self, backbone: Optional[DPN] = None) -> Unet:
        if backbone is not None:
            return Dpn92Unet(backbone)
        return Dpn92Unet(dpn92())


class Dpn92LocalizerWrapper(Dpn92Wrapper, LocalizerModelWrapper):
    model_name = "Dpn92UnetLocalizer"
    data_parallel = False


class Dpn92ClassifierWrapper(Dpn92Wrapper, ClassifierModelWrapper):
    model_name = "Dpn92UnetClassifier"
    data_parallel = True
    default_score = WeightedImageMetric(
        ("LocF1", F1Score(num_classes=5), 0.3),
        ("F1", F1Score(num_classes=5), 0.7)
    )
