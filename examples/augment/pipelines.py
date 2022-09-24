from typing import Tuple

from torch import nn

from metadamagenet.augment import (
    OneOf,
    Random,
    VFlip,
    Rotate90,
    Shift,
    RotateAndScale,
    RGBShift,
    HSVShift,
    BestCrop,
    ElasticTransform,
    GaussianNoise,
    Clahe,
    Blur,
    Saturation,
    Brightness,
    Contrast,
    Dilation
)


def dpn92_unet_double(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.9999),
        Random(Shift(y=(.2, .8), x=(.2, .8)), p=.5),
        Random(RotateAndScale(center_y=(0.3, 0.7), center_x=(0.3, 0.7), angle=(-10., 10.), scale=(.9, 1.1)), p=0.95),
        BestCrop(samples=10, dsize=(input_size, input_size), size_range=(0.4, 0.6)),
        OneOf(
            (RGBShift().only_on('img_pre'), 0.1),
            (RGBShift().only_on('img_post'), 0.1),
        ),
        OneOf(
            (HSVShift().only_on('img_pre'), 0.1),
            (HSVShift().only_on('img_post'), 0.1),
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_pre'), 0.1),
                (GaussianNoise().only_on('img_pre'), 0.1),
                (Blur().only_on('img_pre'), 0.1)), 0.1),
            (OneOf(
                (Saturation().only_on('img_pre'), 0.1),
                (Brightness().only_on('img_pre'), 0.1),
                (Contrast().only_on('img_pre'), 0.1)), 0.1)
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_post'), 0.1),
                (GaussianNoise().only_on('img_post'), 0.1),
                (Blur().only_on('img_post'), 0.1)), 0.1),
            (OneOf(
                (Saturation().only_on('img_post'), 0.1),
                (Brightness().only_on('img_post'), 0.1),
                (Contrast().only_on('img_post'), 0.1)), 0.1)
        ),
        Random(ElasticTransform().only_on('img_pre'), p=0.1),
        Random(ElasticTransform().only_on('img_post'), p=0.1),
        Random(Dilation().only_on('msk'), p=0.9)
    )


def resnet34_unet_double(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.95),
        Random(Shift(), p=.1),
        Random(RotateAndScale(), p=0.4),
        BestCrop(samples=10, dsize=(input_size, input_size), size_range=(0.65, 0.85)),
        OneOf(
            (RGBShift().only_on('img_pre'), 0.015),
            (RGBShift().only_on('img_post'), 0.015),
        ),
        OneOf(
            (HSVShift().only_on('img_pre'), 0.015),
            (HSVShift().only_on('img_post'), 0.015),
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_pre'), 0.015),
                (GaussianNoise().only_on('img_pre'), 0.015),
                (Blur().only_on('img_pre'), 0.1)), 0.015),
            (OneOf(
                (Saturation().only_on('img_pre'), 0.015),
                (Brightness().only_on('img_pre'), 0.015),
                (Contrast().only_on('img_pre'), 0.015)), 0.02)
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_post'), 0.015),
                (GaussianNoise().only_on('img_post'), 0.015),
                (Blur().only_on('img_post'), 0.1)), 0.015),
            (OneOf(
                (Saturation().only_on('img_post'), 0.015),
                (Brightness().only_on('img_post'), 0.015),
                (Contrast().only_on('img_post'), 0.015)), 0.02)
        ),
        Random(ElasticTransform().only_on('img_pre'), p=0.017),
        Random(ElasticTransform().only_on('img_post'), p=0.017),
        Random(Dilation().only_on('msk'), p=0.9)
    )


def senet154_unet_double(input_size: int) -> nn.Module:
    return dpn92_unet_double(input_size)


def seresnext50_unet_double(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.05),
        Random(Shift(), p=.2),
        Random(RotateAndScale(), p=0.8),
        BestCrop(samples=10, dsize=(input_size, input_size), size_range=(0.4, 0.6)),
        OneOf(
            (RGBShift().only_on('img_pre'), 0.04),
            (RGBShift().only_on('img_post'), 0.04),
        ),
        OneOf(
            (HSVShift().only_on('img_pre'), 0.04),
            (HSVShift().only_on('img_post'), 0.04),
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_pre'), 0.04),
                (GaussianNoise().only_on('img_pre'), 0.04),
                (Blur().only_on('img_pre'), 0.1)), 0.04),
            (OneOf(
                (Saturation().only_on('img_pre'), 0.04),
                (Brightness().only_on('img_pre'), 0.04),
                (Contrast().only_on('img_pre'), 0.04)), 0.1)
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_post'), 0.04),
                (GaussianNoise().only_on('img_post'), 0.04),
                (Blur().only_on('img_post'), 0.1)), 0.04),
            (OneOf(
                (Saturation().only_on('img_post'), 0.04),
                (Brightness().only_on('img_post'), 0.04),
                (Contrast().only_on('img_post'), 0.04)), 0.1)
        ),
        Random(ElasticTransform().only_on('img_pre'), p=0.04),
        Random(ElasticTransform().only_on('img_post'), p=0.04),
        Random(Dilation().only_on('msk'), p=0.9)
    )


def dpn92_unet_localization(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.95),
        Random(Shift(y=(.2, .8), x=(.2, .8)), p=.1),
        Random(RotateAndScale(center_y=(0.3, 0.7), center_x=(0.3, 0.7), angle=(-10., 10.), scale=(.9, 1.1)), p=0.1),
        BestCrop(samples=5, dsize=(input_size, input_size), size_range=(0.45, 0.55)),
        Random(RGBShift().only_on('img'), p=0.01),
        Random(HSVShift().only_on('img'), p=0.01),
        OneOf((
            OneOf(
                (Clahe().only_on('img'), 0.01),
                (GaussianNoise().only_on('img'), 0.01),
                (Blur().only_on('img'), 0.01)), 0.01),
            (OneOf(
                (Saturation().only_on('img'), 0.01),
                (Brightness().only_on('img'), 0.01),
                (Contrast().only_on('img'), 0.01)), 0.01)
        ),
        Random(ElasticTransform(), p=0.001)
    )


def resnet34_unet_localization(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.95),
        Random(Shift(), p=.2),
        Random(RotateAndScale(), p=0.8),
        BestCrop(samples=5, dsize=(input_size, input_size), size_range=(0.6, 0.9)),
        OneOf(
            (RGBShift().only_on('img'), 0.03),
            (HSVShift().only_on('img'), 0.03)
        ),
        OneOf((
            OneOf(
                (Clahe().only_on('img'), 0.03),
                (GaussianNoise().only_on('img'), 0.03),
                (Blur().only_on('img'), 0.02)), 0.07),
            (OneOf(
                (Saturation().only_on('img'), 0.03),
                (Brightness().only_on('img'), 0.03),
                (Contrast().only_on('img'), 0.03)), 0.07)
        ),
        Random(ElasticTransform().only_on('img'), p=0.03)
    )


def senet154_unet_localization(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.4),
        Random(Rotate90(), p=0.9),
        Random(Shift(y=(.2, .8), x=(.2, .8)), p=.3),
        Random(RotateAndScale(center_y=(0.3, 0.7), center_x=(0.3, 0.7), angle=(-10., 10.), scale=(.9, 1.1)), p=0.6),
        BestCrop(samples=5, dsize=(input_size, input_size), size_range=(0.42, 0.52)),
        Random(RGBShift().only_on('img'), p=0.05),
        Random(HSVShift().only_on('img'), p=0.04),
        OneOf((
            OneOf(
                (Clahe().only_on('img'), 0.08),
                (GaussianNoise().only_on('img'), 0.08),
                (Blur().only_on('img'), 0.08)), 0.08),
            (OneOf(
                (Saturation().only_on('img'), 0.08),
                (Brightness().only_on('img'), 0.08),
                (Contrast().only_on('img'), 0.08)), 0.08)
        ),
        Random(ElasticTransform(), p=0.05)
    )


def seresnext50_unet_localization(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.95),
        Random(Shift(), p=.1),
        Random(RotateAndScale(), p=0.1),
        BestCrop(samples=5, dsize=(input_size, input_size), size_range=(0.45, 0.55)),
        Random(RGBShift().only_on('img'), p=0.01),
        Random(HSVShift().only_on('img'), p=0.01),
        OneOf(
            (OneOf(
                (Clahe().only_on('img'), 0.01),
                (GaussianNoise().only_on('img'), 0.01),
                (Blur().only_on('img'), 0.01)), 0.01),
            (OneOf(
                (Saturation().only_on('img'), 0.01),
                (Brightness().only_on('img'), 0.01),
                (Contrast().only_on('img'), 0.01)), 0.01)
        ),
        Random(ElasticTransform().only_on('img'), p=0.001)
    )


def dpn92_unet_double_tune(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.3),
        Random(Rotate90(), p=0.7),
        Random(Shift(), p=.01),
        Random(RotateAndScale(), p=0.5),
        BestCrop(samples=10, dsize=(input_size, input_size), size_range=(0.4, 0.6)),
        OneOf(
            (RGBShift().only_on('img_pre'), 0.01),
            (RGBShift().only_on('img_post'), 0.01),
        ),
        OneOf(
            (HSVShift().only_on('img_pre'), 0.01),
            (HSVShift().only_on('img_post'), 0.01),
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_pre'), 0.01),
                (GaussianNoise().only_on('img_pre'), 0.01),
                (Blur().only_on('img_pre'), 0.01)), 0.01),
            (OneOf(
                (Saturation().only_on('img_pre'), 0.01),
                (Brightness().only_on('img_pre'), 0.01),
                (Contrast().only_on('img_pre'), 0.01)), 0.01)
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_post'), 0.01),
                (GaussianNoise().only_on('img_post'), 0.01),
                (Blur().only_on('img_post'), 0.01)), 0.01),
            (OneOf(
                (Saturation().only_on('img_post'), 0.01),
                (Brightness().only_on('img_post'), 0.01),
                (Contrast().only_on('img_post'), 0.01)), 0.01)
        ),
        Random(ElasticTransform().only_on('img_pre'), p=0.01),
        Random(ElasticTransform().only_on('img_post'), p=0.01),
        Random(Dilation().only_on('msk'), p=0.9)
    )


def resnet34_unet_double_tune(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.3),
        Random(Rotate90(), p=0.7),
        Random(Shift(), p=.02),
        Random(RotateAndScale(), p=0.5),
        BestCrop(samples=10, dsize=(input_size, input_size), size_range=(0.65, 0.85)),
        OneOf(
            (RGBShift().only_on('img_pre'), 0.01),
            (RGBShift().only_on('img_post'), 0.01),
        ),
        OneOf(
            (HSVShift().only_on('img_pre'), 0.01),
            (HSVShift().only_on('img_post'), 0.01),
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_pre'), 0.01),
                (GaussianNoise().only_on('img_pre'), 0.01),
                (Blur().only_on('img_pre'), 0.01)), 0.01),
            (OneOf(
                (Saturation().only_on('img_pre'), 0.01),
                (Brightness().only_on('img_pre'), 0.01),
                (Contrast().only_on('img_pre'), 0.01)), 0.01)
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_post'), 0.015),
                (GaussianNoise().only_on('img_post'), 0.015),
                (Blur().only_on('img_post'), 0.015)), 0.01),
            (OneOf(
                (Saturation().only_on('img_post'), 0.01),
                (Brightness().only_on('img_post'), 0.01),
                (Contrast().only_on('img_post'), 0.01)), 0.01)
        ),
        Random(ElasticTransform().only_on('img_pre'), p=0.01),
        Random(ElasticTransform().only_on('img_post'), p=0.01),
        Random(Dilation().only_on('msk'), p=0.9)
    )


def senet154_unet_double_tune(input_size: int) -> nn.Sequential:
    return dpn92_unet_double_tune(input_size)


def seresnext50_unet_double_tune(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.3),
        Random(Rotate90(), p=0.7),
        Random(Shift(), p=.01),
        Random(RotateAndScale(), p=0.5),
        BestCrop(samples=10, dsize=(input_size, input_size), size_range=(0.4, 0.6)),
        OneOf(
            (RGBShift().only_on('img_pre'), 0.01),
            (RGBShift().only_on('img_post'), 0.01),
        ),
        OneOf(
            (HSVShift().only_on('img_pre'), 0.01),
            (HSVShift().only_on('img_post'), 0.01),
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_pre'), 0.04),
                (GaussianNoise().only_on('img_pre'), 0.04),
                (Blur().only_on('img_pre'), 0.04)), 0.01),
            (OneOf(
                (Saturation().only_on('img_pre'), 0.01),
                (Brightness().only_on('img_pre'), 0.01),
                (Contrast().only_on('img_pre'), 0.01)), 0.01)
        ),
        OneOf(
            (OneOf(
                (Clahe().only_on('img_post'), 0.01),
                (GaussianNoise().only_on('img_post'), 0.01),
                (Blur().only_on('img_post'), 0.01)), 0.01),
            (OneOf(
                (Saturation().only_on('img_post'), 0.01),
                (Brightness().only_on('img_post'), 0.01),
                (Contrast().only_on('img_post'), 0.01)), 0.01)
        ),
        Random(ElasticTransform().only_on('img_pre'), p=0.017),
        Random(ElasticTransform().only_on('img_post'), p=0.017),
        Random(Dilation().only_on('msk'), p=0.9)
    )


def dpn92_unet_localization_tune(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.45),
        Random(Rotate90(), p=0.9),
        Random(Shift(), p=.05),
        Random(RotateAndScale(), p=0.05),
        BestCrop(samples=5, dsize=(input_size, input_size), size_range=(0.45, 0.55)),
        OneOf(
            (RGBShift().only_on('img'), 0.01),
            (HSVShift().only_on('img'), 0.01)
        ),
        OneOf((
            OneOf(
                (Clahe().only_on('img'), 0.01),
                (GaussianNoise().only_on('img'), 0.01),
                (Blur().only_on('img'), 0.01)), 0.01),
            (OneOf(
                (Saturation().only_on('img'), 0.01),
                (Brightness().only_on('img'), 0.01),
                (Contrast().only_on('img'), 0.01)), 0.01)
        ),
        Random(ElasticTransform().only_on('img'), p=0.001)
    )


def seresnext50_unet_localization_tune(input_size: Tuple[int, int]) -> nn.Sequential:
    return dpn92_unet_double_tune(input_size)
