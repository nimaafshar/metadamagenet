from typing import Dict, Tuple
from typing_extensions import Self

import torch
from torch import Tensor, nn

from .base import BaseModel


class Mean(BaseModel):
    def __init__(self, *models: BaseModel):
        super().__init__()
        self.models: nn.ModuleList[BaseModel] = nn.ModuleList(models)
        assert len(models) > 0, "len(models) == 0"

    def activate(self, outputs: Tensor) -> Tensor:
        return self.models[0].activate(outputs)

    def preprocess(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.models[0].preprocess(data)

    @classmethod
    def name(cls) -> str:
        return "ModelsMean"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs_sum: torch.Tensor = 0
        for i, model in enumerate(self.models):
            outputs: torch.Tensor = model(x)
            if i == 0:
                outputs_sum = torch.zeros_like(outputs, device=outputs.device)
            outputs_sum += outputs
        return outputs_sum / len(self.models)

    @classmethod
    def from_pretrained(cls, version: str, seed: int, data_parallel: bool = False) -> Self:
        raise NotImplementedError("loading pretrained is not supported for Mean aggregator")
