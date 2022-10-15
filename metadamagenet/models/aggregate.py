import abc
from typing import Dict, Tuple

import torch
from torch import nn

from .base import BaseModel


class ModelAggregator(nn.Module, metaclass=abc.ABCMeta):

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        pass

    @abc.abstractmethod
    def preprocess(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        models forward + activate + aggregate
        :return:
        """


class Mean(ModelAggregator):
    @classmethod
    def name(cls) -> str:
        return "ModelsMean"

    def __init__(self, *models: BaseModel):
        super().__init__()
        self.models: nn.ModuleList[BaseModel] = nn.ModuleList(models)
        assert len(models) > 0, "len(models) == 0"

    def preprocess(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.models[0].preprocess(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward + activate + mean
        """
        outputs_sum: torch.Tensor = 0
        for i, model in enumerate(self.models):
            outputs: torch.Tensor = model.activate(model(x))
            if i == 0:
                outputs_sum = torch.zeros_like(outputs, device=outputs.device)
            outputs_sum += outputs
        return outputs_sum / len(self.models)
