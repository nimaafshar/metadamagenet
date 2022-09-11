import abc
from typing import Tuple, Type, Union

import torch
from torch import nn

from ..models import Metadata
from ..models.unet import Unet, Localizer, Classifier
from ..models.checkpoint import Checkpoint
from ..models.manager import Manager as ModelManager
from ..metrics import Score, Dice
from ..logging import log


class ModelWrapper(abc.ABC):
    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def input_size(self) -> Tuple[int, int]:
        """
        :return: (height,width)
        """
        pass

    @property
    @abc.abstractmethod
    def unet_type(self) -> Union[Type[Localizer], Type[Classifier]]:
        pass

    @property
    @abc.abstractmethod
    def data_parallel(self) -> bool:
        pass

    @abc.abstractmethod
    def empty_unet(self) -> Unet:
        pass

    @abc.abstractmethod
    def unet_with_pretrained_backbone(self, backbone: nn.Module) -> Unet:
        pass

    def from_checkpoint(self, version: str, seed: int) -> Tuple[nn.Module, Metadata]:
        log(":eyes: loading from checkpoint")
        checkpoint = Checkpoint(
            model_name=self.model_name,
            version=version,
            seed=seed
        )
        manager = ModelManager.get_instance()
        state_dict, metadata = manager.load_checkpoint(checkpoint)
        empty_model: nn.Module = self.unet_type(self.empty_unet)
        if self.data_parallel:
            empty_model = nn.DataParallel(empty_model)
        empty_model.load_state_dict(state_dict, strict=True)
        return empty_model, metadata

    def from_unet(self, unet: Unet) -> Tuple[nn.Module, Metadata]:
        log(":eyes: creating from unet")
        model: nn.Module = self.unet_type(unet)
        if self.data_parallel:
            model = nn.DataParallel(model)
        return model, Metadata()

    def from_backbone(self, backbone: nn.Module) -> Tuple[nn.Module, Metadata]:
        log(":eyes: creating from backbone")
        model: nn.Module = self.unet_type(self.unet_with_pretrained_backbone(backbone))
        if self.data_parallel:
            model = nn.DataParallel(model)
        return model, Metadata()

    @abc.abstractmethod
    def apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def default_score(self) -> Score:
        pass


class ClassifierModelWrapper(ModelWrapper, abc.ABC):
    unet_type = Classifier


class LocalizerModelWrapper(ModelWrapper, abc.ABC):
    unet_type = Localizer
    default_score = Score(
        ("Dice", Dice(threshold=0.5, channel=0), 1)
    )

    def apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x[:, 0, ...])
