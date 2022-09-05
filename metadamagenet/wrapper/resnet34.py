import abc
from typing import Tuple
from torchvision.models import resnet34
from torch import nn

from .wrapper import ModelWrapper, LocalizerModelWrapper, ClassifierModelWrapper
from metadamagenet.models.unet import Unet
from metadamagenet.models.metadata import Metadata
from metadamagenet.models.checkpoint import Checkpoint
from metadamagenet.models.manager import Manager
from metadamagenet.models.unet import Resnet34Unet
from torchvision.models import ResNet


class Resnet34Wrapper(ModelWrapper, abc.ABC):
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

    def from_unet(self, unet: Resnet34Unet) -> Tuple[nn.Module, Metadata]:
        return nn.DataParallel(self.unet_type(unet)), Metadata()

    def from_backbone(self, backbone: ResNet) -> Tuple[nn.Module, Metadata]:
        """
        :param backbone: a resnet34 model
        """
        return nn.DataParallel((self.unet_type(Resnet34Unet(backbone)))), Metadata()


class Resnet34LocalizerWrapper(Resnet34Wrapper, LocalizerModelWrapper):
    @property
    def model_name(self) -> str:
        return "Resnet34UnetLocalizer"


class Resnet34ClassifierWrapper(Resnet34Wrapper, ClassifierModelWrapper):

    @property
    def model_name(self) -> str:
        return "Resnet34UnetClassifier"
