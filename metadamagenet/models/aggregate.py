from typing import Dict, Tuple
from typing_extensions import Self

import torch
from torch import Tensor, nn

from .base import BaseModel
from torchmetrics.aggregation import MeanMetric


class Mean(BaseModel):
    def __init__(self, *models: BaseModel):
        super().__init__()
        self.models: nn.ModuleList[BaseModel] = nn.ModuleList(models)
        self.mean: MeanMetric = MeanMetric()
        assert len(models) > 0, "len(models) == 0"

    def activate(self, outputs: Tensor) -> Tensor:
        return self.models[0].activate(outputs)

    def preprocess(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.models[0].preprocess(data)

    @classmethod
    def name(cls) -> str:
        return "ModelsMean"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean.reset()
        for model in self.models:
            self.mean.update(model(x))
        return self.mean.compute()

    @classmethod
    def from_pretrained(cls, version: str, seed: int, data_parallel: bool = False) -> Self:
        raise NotImplementedError("loading pretrained is not supported for Mean aggregator")
