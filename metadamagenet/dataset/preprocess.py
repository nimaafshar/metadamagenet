import abc
from typing import Dict, Tuple

import torch
from torch import nn


class ImagePreprocessor(nn.Module, abc.ABC):
    def __init__(self, transforms: nn.Sequential):
        super().__init__()
        self.transforms: nn.Sequential = transforms

    @abc.abstractmethod
    def forward(self, data: Dict[str, torch.FloatTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class LocalizationPreprocessor(ImagePreprocessor):
    def forward(self, data: Dict[str, torch.FloatTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        result: Dict[str, torch.FloatTensor] = self.transforms(data)

        return (result['img'] * 2 - 1), result['msk'].long()


class ClassificationPreprocessor(ImagePreprocessor):
    def forward(self, data: Dict[str, torch.FloatTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        result: Dict[str, torch.FloatTensor] = self.transforms(data)

        return (torch.cat((result['img_pre'] * 2 - 1, result['img_post'] * 2 - 1), dim=1),
                torch.nn.functional.one_hot(result['msk'].long(), num_classes=5))
