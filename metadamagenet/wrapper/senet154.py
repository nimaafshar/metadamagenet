import abc
from typing import Tuple

import torch
from torch import nn


from .wrapper import ModelWrapper, LocalizerModelWrapper, ClassifierModelWrapper
from metadamagenet.models.metadata import Metadata
from metadamagenet.models.checkpoint import Checkpoint
from metadamagenet.models.manager import Manager
from metadamagenet.models.unet import SeNet154Unet
from metadamagenet.models.senet import SENet, senet154


class SeNet154Wrapper(ModelWrapper, abc.ABC):
    def from_checkpoint(self, version: str, seed: int) -> Tuple[nn.Module, Metadata]:
        checkpoint = Checkpoint(
            model_name=self.model_name,
            version=version,
            seed=seed
        )
        manager = Manager.get_instance()
        state_dict, metadata = manager.load_checkpoint(checkpoint)
        empty_model: nn.Module = nn.DataParallel(self.unet_type(SeNet154Unet(senet154(pretrained=False))))
        empty_model.load_state_dict(state_dict, strict=True)
        return empty_model, metadata

    def from_unet(self, unet: SeNet154Unet) -> Tuple[nn.Module, Metadata]:
        return nn.DataParallel(self.unet_type(unet)), Metadata()

    def from_backbone(self, backbone: SENet) -> Tuple[nn.Module, Metadata]:
        """
        :param backbone: a senet154 module
        """
        return nn.DataParallel((self.unet_type(SeNet154Unet(backbone)))), Metadata()


class SeNet154LocalizerWrapper(SeNet154Wrapper, LocalizerModelWrapper):
    @property
    def model_name(self) -> str:
        return "SeNet154UnetLocalizer"


class SeNet154ClassifierWrapper(SeNet154Wrapper, ClassifierModelWrapper):

    @property
    def model_name(self) -> str:
        return "SeNet154UnetClassifier"

    def apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x,dim=1)