import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from .modules import ConvRelu
from .base import Unet


class Resnet34Unet(Unet):
    def __init__(self, resnet: torchvision.models.ResNet):
        super(Resnet34Unet, self).__init__()
        encoder_filters = [64, 64, 128, 256, 512]
        decoder_filters = [48, 64, 96, 160, 320]
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

        self._initialize_weights()

        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu)
        self.conv2 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1)
        self.conv3 = resnet.layer2
        self.conv4 = resnet.layer3
        self.conv5 = resnet.layer4

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
