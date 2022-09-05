import abc
from typing import Tuple

import torch
from torch import nn

from .wrapper import ModelWrapper, LocalizerModelWrapper, ClassifierModelWrapper
from metadamagenet.models.unet import Unet
from metadamagenet.models.metadata import Metadata
from metadamagenet.models.checkpoint import Checkpoint
from metadamagenet.models.manager import Manager
from metadamagenet.models.unet import SeResnext50Unet
from metadamagenet.models.senet import se_resnext50_32x4d, SENet


class SeResnext50Wrapper(ModelWrapper, abc.ABC):
    pass


class SeResnext50LocalizerWrapper(SeResnext50Wrapper, LocalizerModelWrapper):
    @property
    def model_name(self) -> str:
        return "SeResnext50UnetLocalizer"

    def from_checkpoint(self, version: str, seed: int) -> Tuple[nn.Module, Metadata]:
        checkpoint = Checkpoint(
            model_name=self.model_name,
            version=version,
            seed=seed
        )
        manager = Manager.get_instance()
        state_dict, metadata = manager.load_checkpoint(checkpoint)
        empty_model: nn.Module = self.unet_type(SeResnext50Unet(se_resnext50_32x4d(pretrained=None)))
        empty_model.load_state_dict(state_dict, strict=True)
        return empty_model, metadata

    def from_unet(self, unet: SeResnext50Unet) -> Tuple[nn.Module, Metadata]:
        return self.unet_type(unet), Metadata()

    def from_backbone(self, backbone: SENet) -> Tuple[nn.Module, Metadata]:
        """
        :param backbone: a se_resnext50_32x4d module
        """
        return self.unet_type(SeResnext50Unet(backbone)), Metadata()


class SeResnext50ClassifierWrapper(SeResnext50Wrapper, ClassifierModelWrapper):

    @property
    def model_name(self) -> str:
        return "SeResnext50UnetClassifier"

    def from_checkpoint(self, version: str, seed: int) -> Tuple[nn.Module, Metadata]:
        checkpoint = Checkpoint(
            model_name=self.model_name,
            version=version,
            seed=seed
        )
        manager = Manager.get_instance()
        state_dict, metadata = manager.load_checkpoint(checkpoint)
        empty_model: nn.Module = nn.DataParallel(self.unet_type(SeResnext50Unet(se_resnext50_32x4d(pretrained=None))))
        empty_model.load_state_dict(state_dict, strict=True)
        return empty_model, metadata

    def from_unet(self, unet: SeResnext50Unet) -> Tuple[nn.Module, Metadata]:
        return nn.DataParallel(self.unet_type(unet)), Metadata()

    def from_backbone(self, backbone: SENet) -> Tuple[nn.Module, Metadata]:
        """
        :param backbone: a se_resnext50_32x4d module
        """
        return nn.DataParallel((self.unet_type(SeResnext50Unet(backbone)))), Metadata()

    def apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)
