import abc
from typing import Optional

import torch
from torchvision.models import resnet34, ResNet
from torchvision.models import ResNet34_Weights

from .wrapper import ModelWrapper, LocalizerModelWrapper, ClassifierModelWrapper
from ..models.unet import Unet
from ..models.unet import Resnet34Unet
from ..metrics import WeightedImageMetric, DamageF1Score, Dice, ImageMetric


class Resnet34Wrapper(ModelWrapper, abc.ABC):
    unet_class = Resnet34Unet
    data_parallel = True

    def empty_unet(self) -> Unet:
        return Resnet34Unet(resnet34(weights=None))

    def unet_with_pretrained_backbone(self, backbone: Optional[ResNet] = None) -> Unet:
        if backbone is not None:
            return Resnet34Unet(backbone)
        return Resnet34Unet(resnet34(weights=ResNet34_Weights.DEFAULT))


class Resnet34LocalizerWrapper(Resnet34Wrapper, LocalizerModelWrapper):
    model_name = "Resnet34UnetLocalizer"
    input_size = (736, 736)


class Resnet34ClassifierWrapper(Resnet34Wrapper, ClassifierModelWrapper):
    model_name = "Resnet34UnetClassifier"
    input_size = (608, 608)
    default_score: ImageMetric = WeightedImageMetric(
        ("LocDice", Dice(threshold=0.5, channel=0, inverse=True), 0.3),
        ("F1", DamageF1Score(clip_localization_mask=True), 0.7)
    )
