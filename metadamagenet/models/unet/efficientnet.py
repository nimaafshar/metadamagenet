import torch
from torch import nn
import torch.nn.functional as F

from .base import Unet
from .modules import ConvRelu


class EfficientUnetB0(Unet):
    def __init__(self, backbone: nn.Module):
        """
        :param backbone: EfficientNet B0 instance from
        (https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet)
        """
        super().__init__()
        encoder_filters = [32, 16, 40, 112, 320]
        decoder_filters = [32, 16, 32, 96, 256]
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])

        # res

        self._initialize_weights()

        self.conv1 = backbone.stem
        self.conv2 = backbone.layers[0]
        self.conv3 = nn.Sequential(
            backbone.layers[1],
            backbone.layers[2]
        )
        self.conv4 = nn.Sequential(
            backbone.layers[3],
            backbone.layers[4]
        )
        self.conv5 = nn.Sequential(
            backbone.layers[5],
            backbone.layers[6]
        )

    def forward(self, x: torch.Tensor):
        """
                :param x: batch_size, C, H, W = x.shape
                :return:
                """
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc1], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10

    @property
    def out_channels(self) -> int:
        return self.decoder_filters[-5]


def efficientnet_b0(pretrained: bool = False):
    return torch.hub.load(repo='NVIDIA/DeepLearningExamples:torchhub',
                          model='nvidia_efficientnet_b0',
                          pretrained=pretrained)
