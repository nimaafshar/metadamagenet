import abc
from typing import Tuple

from torch import nn

from metadamagenet.models.metadata import Metadata
from metadamagenet.models.unet import Unet, Localizer, Classifier


class ModelWrapper(abc.ABC):

    @abc.abstractmethod
    @property
    def unet_type(self) -> nn.Module:
        pass

    @abc.abstractmethod
    @property
    def model_name(self) -> str:
        pass

    @abc.abstractmethod
    def from_checkpoint(self, version: str, seed: int) -> Tuple[nn.Module, Metadata]:
        pass

    @abc.abstractmethod
    def from_unet(self, unet: Unet) -> Tuple[nn.Module, Metadata]:
        pass

    @abc.abstractmethod
    def from_backbone(self, backbone: nn.Module) -> Tuple[nn.Module, Metadata]:
        pass


class ClassifierModelWrapper(ModelWrapper, abc.ABC):
    @property
    def unet_type(self) -> nn.Module:
        return Classifier


class LocalizerModelWrapper(ModelWrapper, abc.ABC):
    @property
    def unet_type(self) -> nn.Module:
        return Localizer
