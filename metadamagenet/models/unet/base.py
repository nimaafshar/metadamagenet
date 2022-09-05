import abc
from typing import Optional, List

import torch
from torch import nn


class Unet(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    @property
    @abc.abstractmethod
    def out_channels(self) -> int:
        pass

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Localizer(nn.Module):
    def __init__(self, unet: Unet):
        super(Localizer, self).__init__()
        self.unet: nn.Module = unet
        self.res: nn.Conv2d = nn.Conv2d(in_channels=unet.out_channels,
                                        out_channels=1,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.unet(x))

    def _initialize_weights(self) -> None:
        """
        initialize model weights assuming that unet weights are initialized
        :return:
        """
        self.res.weight.data = nn.init.kaiming_normal_(self.res.weight.data)
        if self.res.bias is not None:
            self.res.bias.data.zero_()


class Classifier:
    def __init__(self, unet: Unet):
        self.unet: Unet = unet
        self.res: nn.Conv2d = nn.Conv2d(in_channels=self.unet.out_channels * 2,
                                        out_channels=5,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        dec10_0 = self.unet(x[:, :3, :, :])
        dec10_1 = self.unet(x[:, 3:, :, :])
        dec10 = torch.cat([dec10_0, dec10_1], dim=1)
        return self.res(dec10)

    def _initialize_weights(self) -> None:
        """
        initialize model weights assuming that unet weights are initialized
        :return:
        """
        self.res.weight.data = nn.init.kaiming_normal_(self.res.weight.data)
        if self.res.bias is not None:
            self.res.bias.data.zero_()
