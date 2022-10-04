from typing import List
import abc

import torch
from torch import nn


class UnetBase(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, pretrained_backbone: bool):
        super().__init__()

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: torch.Tensor of shape (N,C,H,H)
        :return: torch.Tensor of shape (N,C',H',H')
        """
        pass

    @property
    def out_channels(self) -> int:
        return self.decoder_filters[0]

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @property
    @abc.abstractmethod
    def encoder_filters(self) -> List[int]:
        """
        :return: 5 encoder filters
        """
        pass

    @property
    @abc.abstractmethod
    def decoder_filters(self) -> List[int]:
        """
        :return: 5 decoder filters
        """
        pass
