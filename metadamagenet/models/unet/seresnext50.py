import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .base import Unet
from .modules import ConvRelu
from ..senet import SENet


class SeResnext50Unet(Unet):

    def __init__(self, se_resnext: SENet):
        super().__init__()
        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2
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

        self.conv1 = nn.Sequential(
            se_resnext.layer0.conv1,
            se_resnext.layer0.bn1,
            se_resnext.layer0.relu1)  # se_resnext.layer0.conv1
        self.conv2 = nn.Sequential(
            se_resnext.pool,
            se_resnext.layer1)
        self.conv3 = se_resnext.layer2
        self.conv4 = se_resnext.layer3
        self.conv5 = se_resnext.layer4

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
