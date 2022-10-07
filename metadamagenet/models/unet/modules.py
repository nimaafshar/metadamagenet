from typing import ClassVar, Type
import torch
from torch import nn
import torch.nn.functional as tf

from ..senet import SCSEModule


class ConvReluBN(nn.Module):
    """
    Conv2d + BatchNorm2d + ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvReluBN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class ConvRelu(nn.Module):
    """
    Conv2d + ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class DecoderModule(nn.Module):
    """
    Simple Decoder Module In Unet
    """
    ConvType: ClassVar[Type[nn.Module]] = ConvReluBN

    def __init__(self, in_channels: int, injected_channels: int, out_channels: int):
        super().__init__()
        self.in_channels: int = in_channels
        self.injected_channels: int = injected_channels
        self.out_channels: int = out_channels

        self.conv1: nn.Module = self.ConvType(in_channels, out_channels)
        self.conv2: nn.Module = self.ConvType(out_channels + injected_channels, out_channels)

    def forward(self, inputs: torch.Tensor, injected: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: torch.Tensor of shape (N,in_channels,H/2,H/2)
        :param injected: torch.Tensor of shape (N,injected_channels,H,H)
        :return: torch.Tensor of shape (N,out_channels,H,H)
        """
        out1: torch.Tensor = self.conv1(tf.interpolate(inputs, scale_factor=2))
        return self.conv2(torch.cat((out1, injected), dim=1))


class SCSEDecoderModule(DecoderModule):
    """
    Spatial and Channel Squeeze and Excitation Decoder Module In Unet
    No Concat
    """

    def __init__(self, in_channels: int, injected_channels: int, out_channels: int):
        super(DecoderModule, self).__init__()
        self.conv1 = self.ConvType(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            self.ConvType(out_channels + injected_channels, out_channels),
            SCSEModule(out_channels, reduction=16, concat=False)
        )


class FinalDecoderModule(nn.Module):
    """
    Final Decoder Module In Unet Which only uses one conv
    """
    ConvType: ClassVar[Type[nn.Module]] = ConvReluBN

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.conv: nn.Module = self.ConvType(in_channels, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: torch.Tensor of shape (N,in_channels,H/2,H/2)
        :return: torch.Tensor of shape (N,out_channels,H,H)
        """
        return self.conv(tf.interpolate(inputs, scale_factor=2))
