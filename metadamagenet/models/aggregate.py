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
            outputs: torch.Tensor = model(x)
            if i == 0:
                outputs_sum = torch.zeros_like(outputs, device=outputs.device)
            outputs_sum += outputs
        return self.models[0].activate(outputs_sum / len(self.models))


class FourFlips(ModelAggregator):
    @classmethod
    def name(cls) -> str:
        return "FourFlips"

    def __init__(self, model: BaseModel):
        super().__init__()
        assert isinstance(model, BaseModel), "model should be an instance of BaseModel"
        self.model: BaseModel = model

    def preprocess(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.preprocess(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward + activate + mean
        """

        def forward_transform(model, inputs, dims):
            return torch.flip(model(torch.flip(inputs, dims=dims)), dims=dims)

        outputs_sum: torch.Tensor = self.model(x)  # original
        outputs_sum += forward_transform(self.model, x, (2,))  # top-down
        outputs_sum += forward_transform(self.model, x, (3,))  # left-right
        outputs_sum += forward_transform(self.model, x, (2, 3))  # top-down and left-right
        return self.model.activate(outputs_sum / 4)


class FourRotations(ModelAggregator):
    @classmethod
    def name(cls) -> str:
        return "FourRotations"

    def __init__(self, model: BaseModel):
        super().__init__()
        assert isinstance(model, BaseModel), "model should be an instance of BaseModel"
        self.model: BaseModel = model

    def preprocess(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.preprocess(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward + activate + mean
        """

        def forward_transform(model, inputs, k):
            return torch.rot90(model(torch.rot90(inputs, k=k, dims=(2, 3))), k=(4 - k), dims=(2, 3))

        outputs_sum: torch.Tensor = self.model(x)  # original
        outputs_sum += forward_transform(self.model, x, 1)  # 90 degree
        outputs_sum += forward_transform(self.model, x, 2)  # 180 degree
        outputs_sum += forward_transform(self.model, x, 3)  # 270 degree
        return self.model.activate(outputs_sum / 4)
