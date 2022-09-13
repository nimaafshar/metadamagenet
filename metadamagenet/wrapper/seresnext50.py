import abc
from typing import Optional

import torch

from .wrapper import ModelWrapper, LocalizerModelWrapper, ClassifierModelWrapper
from metadamagenet.models.unet import Unet
from metadamagenet.models.unet import SeResnext50Unet
from metadamagenet.models.senet import se_resnext50_32x4d, SENet
from ..metrics import WeightedImageMetric, F1Score


class SeResnext50Wrapper(ModelWrapper, abc.ABC):
    input_size = 512, 512

    def empty_unet(self) -> Unet:
        return SeResnext50Unet(se_resnext50_32x4d(pretrained=None))

    def unet_with_pretrained_backbone(self, backbone: Optional[SENet] = None) -> Unet:
        if backbone is not None:
            return SeResnext50Unet(backbone)
        return SeResnext50Unet(se_resnext50_32x4d())


class SeResnext50LocalizerWrapper(SeResnext50Wrapper, LocalizerModelWrapper):
    model_name = "SeResnext50UnetLocalizer"
    data_parallel = False


class SeResnext50ClassifierWrapper(SeResnext50Wrapper, ClassifierModelWrapper):
    data_parallel = True
    model_name = "SeResnext50UnetClassifier"
    default_score = WeightedImageMetric(
        ("LocF1", F1Score(start_idx=0, end_idx=1), 0.3),
        ("F1", F1Score(start_idx=1, end_idx=5), 0.7)
    )

    def apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)
