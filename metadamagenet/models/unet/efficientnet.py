from typing import ClassVar, Type
import abc
import torch
from torch import nn

from .base import UnetBase
from .modules import (DecoderModule, FinalDecoderModule, SCSEDecoderModule)


class EfficientUnet(UnetBase, metaclass=abc.ABCMeta):
    DecoderModuleType: ClassVar[Type[DecoderModule]] = DecoderModule
    FinalDecoderModuleType: ClassVar[Type[FinalDecoderModule]] = FinalDecoderModule

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__(pretrained_backbone)

        encoder_filters = self.encoder_filters
        decoder_filters = self.decoder_filters

        self.conv6 = self.DecoderModuleType(in_channels=encoder_filters[-1],
                                            injected_channels=encoder_filters[-2],
                                            out_channels=decoder_filters[-1])
        self.conv7 = self.DecoderModuleType(in_channels=decoder_filters[-1],
                                            injected_channels=encoder_filters[-3],
                                            out_channels=decoder_filters[-2])

        self.conv8 = self.DecoderModuleType(in_channels=decoder_filters[-2],
                                            injected_channels=encoder_filters[-4],
                                            out_channels=decoder_filters[-3])
        self.conv9 = self.DecoderModuleType(in_channels=decoder_filters[-3],
                                            injected_channels=encoder_filters[-5],
                                            out_channels=decoder_filters[-4])

        self.conv10 = self.FinalDecoderModuleType(in_channels=decoder_filters[-4],
                                                  out_channels=decoder_filters[-5])

        self._initialize_weights()

        backbone: nn.Module = self.get_backbone(pretrained_backbone)

        self.conv1 = nn.Sequential(
            backbone.stem,
            backbone.layers[0]
        )
        self.conv2 = backbone.layers[1]
        self.conv3 = backbone.layers[2]
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

        dec = self.conv6(enc5, enc4)
        dec = self.conv7(dec, enc3)
        dec = self.conv8(dec, enc2)
        dec = self.conv9(dec, enc1)
        dec = self.conv10(dec)

        return dec

    @abc.abstractmethod
    def get_backbone(self, pretrained: bool) -> nn.Module:
        pass


class EfficientUnetSCSE(EfficientUnet, metaclass=abc.ABCMeta):
    DecoderModuleType = SCSEDecoderModule
    FinalDecoderModuleType = FinalDecoderModule


"""
models based on EfficientNetB0
"""


class EfficientUnetB0(EfficientUnet):
    encoder_filters = [16, 24, 40, 112, 320]
    decoder_filters = [48, 64, 96, 160, 320]  # same as Resnet34Unet

    def get_backbone(self, pretrained: bool) -> nn.Module:
        return torch.hub.load(repo_or_dir='NVIDIA/DeepLearningExamples:torchhub',
                              model='nvidia_efficientnet_b0',
                              pretrained=pretrained,
                              trust_repo=True)


class EfficientUnetB0SCSE(EfficientUnetSCSE):
    encoder_filters = [16, 24, 40, 112, 320]
    decoder_filters = [48, 64, 96, 160, 320]  # same as Resnet34Unet

    def get_backbone(self, pretrained: bool) -> nn.Module:
        return torch.hub.load(repo_or_dir='NVIDIA/DeepLearningExamples:torchhub',
                              model='nvidia_efficientnet_b0',
                              pretrained=pretrained,
                              trust_repo=True)


class EfficientUnetWideSEB0(EfficientUnet):
    """
    EfficientUnet with Wide SE modules in encoder
    """
    encoder_filters = [16, 24, 40, 112, 320]
    decoder_filters = [48, 64, 96, 160, 320]  # same as Resnet34Unet

    def get_backbone(self, pretrained: bool) -> nn.Module:
        return torch.hub.load(repo_or_dir='NVIDIA/DeepLearningExamples:torchhub',
                              model='nvidia_efficientnet_widese_b0',
                              pretrained=pretrained,
                              trust_repo=True)


"""
models based on EfficientUnetB4
"""


class EfficientUnetB4(EfficientUnet):
    encoder_filters = [24, 32, 56, 160, 448]
    decoder_filters = [48, 64, 96, 160, 320]  # same as Resnet34Unet

    def get_backbone(self, pretrained: bool):
        return torch.hub.load(repo_or_dir='NVIDIA/DeepLearningExamples:torchhub',
                              model='nvidia_efficientnet_b4',
                              pretrained=pretrained,
                              trust_repo=True)


class EfficientUnetB4SCSE(EfficientUnetSCSE):
    encoder_filters = [24, 32, 56, 160, 448]
    decoder_filters = [48, 64, 96, 160, 320]  # same as Resnet34Unet

    def get_backbone(self, pretrained: bool):
        return torch.hub.load(repo_or_dir='NVIDIA/DeepLearningExamples:torchhub',
                              model='nvidia_efficientnet_b4',
                              pretrained=pretrained,
                              trust_repo=True)
