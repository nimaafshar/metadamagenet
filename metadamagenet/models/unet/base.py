import abc

import torch
from torch import nn


class Unet(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    @abc.abstractmethod
    def initialize_weights(self):
        pass

    @property
    @abc.abstractmethod
    def out_channels(self) -> int:
        pass


class Localizer(nn.Module):
    def __init__(self, unet: Unet):
        super(Localizer, self).__init__()
        self.unet: nn.Module = unet
        self.res: nn.Conv2d = nn.Conv2d(in_channels=unet.out_channels,
                                        out_channels=1,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.unet(x))

    def initialize_weights(self) -> None:
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

    def forward(self, x: torch.Tensor):
        dec10_0 = self.unet(x[:, :3, :, :])
        dec10_1 = self.unet(x[:, 3:, :, :])
        dec10 = torch.cat([dec10_0, dec10_1], 1)
        return self.res(dec10)

    def initialize_weights(self) -> None:
        """
        initialize model weights assuming that unet weights are initialized
        :return:
        """
        self.res.weight.data = nn.init.kaiming_normal_(self.res.weight.data)
        if self.res.bias is not None:
            self.res.bias.data.zero_()
