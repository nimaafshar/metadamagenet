from typing import Tuple
from torchvision.models import resnet34
from torch import nn

from .wrapper import ModelWrapper, LocalizerModelWrapper, ClassifierModelWrapper
from ..unet.base import Unet, Localizer, Classifier
from ..metadata import Metadata
from ..checkpoint import Checkpoint
from ..manager import Manager
from ..unet.resnet34 import Resnet34Unet
from torchvision.models import ResNet


class Resnet34Wrapper(LocalizerModelWrapper):
    @property
    def model_name(self) -> str:
        return "Resnet34UnetLocalizer"

    def from_checkpoint(self, version: str, seed: int) -> Tuple[nn.Module, Metadata]:
        checkpoint = Checkpoint(
            model_name=self.model_name,
            version=version,
            seed=seed
        )
        manager = Manager.get_instance()
        state_dict, metadata = manager.load_checkpoint(checkpoint)
        empty_model: nn.Module = nn.DataParallel(self.unet_type(Resnet34Unet(resnet34())))
        empty_model.load_state_dict(state_dict, strict=True)
        return empty_model, metadata

    def from_unet(self, unet: Unet) -> Localizer:
        return nn.DataParallel(self.unet_type(unet)), Metadata()

    def from_backbone(self, backbone: ResNet) -> Localizer:
        return nn.DataParallel((self.unet_type(Resnet34Unet(backbone))))


class Resnet34ClassifierWrapper(ModelWrapper, LocalizerModelWrapper):

    @property
    def model_name(self) -> str:
        return "Resnet34UnetClassifier"

    def from_checkpoint(self, version: str, seed: int) -> Tuple[nn.Module, Metadata]:
        checkpoint = Checkpoint(
            model_name=self.model_name,
            version=version,
            seed=seed
        )
        manager = Manager.get_instance()
        state_dict, metadata = manager.load_checkpoint(checkpoint)
        empty_model: nn.Module = nn.DataParallel(self.unet_type(Resnet34Unet(resnet34())))
        empty_model.load_state_dict(state_dict, strict=True)
        return empty_model, metadata

    def from_unet(self, unet: Unet) -> Localizer:
        return nn.DataParallel(self.unet_type(unet)), Metadata()

    def from_backbone(self, backbone: ResNet) -> Localizer:
        return nn.DataParallel((self.unet_type(Resnet34Unet(backbone))))
