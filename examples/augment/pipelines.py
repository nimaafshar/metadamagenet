from typing import Tuple

from torch import nn

from metadamagenet.augment import (
    OneOf,
    Random,
    VFlip,
    Rotate90,
    Shift,
    RotateAndScale,
    Resize,
    RGBShift,
    HSVShift,
    RandomCrop,
    ElasticTransform,
    GaussianNoise,
    Clahe,
    Blur,
    Saturation,
    Brightness,
    Contrast,
    Dilation
)


# TODO: move dilation to augments

def dpn92_unet_double(input_shape: Tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.9999),
        Random(Shift(y=(.2, .8), x=(.2, .8)), p=.5),
        Random(RotateAndScale(center_y=(0.3, 0.7), center_x=(0.3, 0.7), angle=(-10., 10.), scale=(.9, 1.1)), p=0.95),
        RandomCrop(  # TODO: replace
            default_crop_size=input_shape[0],
            size_change_probability=0.05,
            crop_size_range=(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85)),
            try_range=(1, 10),
            scoring_weights={'msk2': 5, 'msk3': 5, 'msk4': 2, 'msk1': 1}
        ),
        Resize(  # TODO: replace
            height=input_shape[0],
            width=input_shape[1],
        ),
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
        (ElasticTransform().only_on('img_pre'), 0.1),
        (ElasticTransform().only_on('img_post'), 0.1),
        (Dilation().only_on('msk'), 0.9)
    )


def resnet34_unet_double(input_shape: Tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.95),
        Random(Shift(), p=.1),
        Random(RotateAndScale(), p=0.4),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.2,
            crop_size_range=(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85)),
            try_range=(1, 10),
            scoring_weights={'msk2': 5, 'msk3': 5, 'msk4': 2, 'msk1': 1}
        ),
        Resize(
            height=input_shape[0],
            width=input_shape[1],
        ),
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
        (ElasticTransform().only_on('img_pre'), 0.017),
        (ElasticTransform().only_on('img_post'), 0.017),
        (Dilation().only_on('msk'), 0.9)
    )


def senet154_unet_double(input_shape: Tuple[int, int]) -> nn.Module:
    return dpn92_unet_double(input_shape)


def seresnext50_unet_double(input_shape: Tuple[int, int]) -> nn.Sequential:
    return nn.Sequential((
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.05),
        Random(Shift(), p=.2),
        Random(RotateAndScale(), p=0.8),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.1,
            crop_size_range=(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85)),
            try_range=(1, 10),
            scoring_weights={'msk2': 5, 'msk3': 5, 'msk4': 2, 'msk1': 1}
        ),
        Resize(
            height=input_shape[0],
            width=input_shape[1],
        ),
        OneOf(
            (RGBShift().only_on('img_pre'), 0.04),
            (RGBShift().only_on('img_post'), 0.04),
        ),
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
        (ElasticTransform().only_on('img_pre'), 0.04),
        (ElasticTransform().only_on('img_post'), 0.04),
        (Dilation().only_on('msk'), 0.9)
    )


def dpn92_unet_localization(input_shape: Tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.95),
        Random(Shift(y=(.2, .8), x=(.2, .8)), p=.1),
        Random(RotateAndScale(center_y=(0.3, 0.7), center_x=(0.3, 0.7), angle=(-10., 10.), scale=(.9, 1.1)), p=0.1),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.5,
            crop_size_range=(int(input_shape[0] / 1.1), int(input_shape[0] / 0.9)),
            try_range=(1, 5)
        ),
        Resize(*input_shape),
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
                (Contrast().only_on('img'), 0.01), 0.01))
        ),
        Random(ElasticTransform(), p=0.001)
    )


def resnet34_unet_localization(input_shape: Tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.95),
        Random(Shift(), p=.2),
        Random(RotateAndScale(), p=0.8),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.3,
            crop_size_range=(int(input_shape[0] / 1.2), int(input_shape[0] / 0.8)),
            try_range=(1, 5)
        ),
        Resize(*input_shape),
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
                (Contrast().only_on('img'), 0.03), 0.07))
        ),
        Random(ElasticTransform().only_on('img'), p=0.03)
    )


def senet154_unet_localization(input_shape: Tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.4),
        Random(Rotate90(), p=0.9),
        Random(Shift(y=(.2, .8), x=(.2, .8)), p=.3),
        Random(RotateAndScale(center_y=(0.3, 0.7), center_x=(0.3, 0.7), angle=(-10., 10.), scale=(.9, 1.1)), p=0.6),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.2,
            crop_size_range=(int(input_shape[0] / 1.1), int(input_shape[0] / 0.9)),
            try_range=(1, 5)
        ),
        Resize(*input_shape),
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
                (Contrast().only_on('img'), 0.08), 0.08))
        ),
        Random(ElasticTransform(), p=0.05)
    )


def seresnext50_unet_localization(input_shape: Tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.5),
        Random(Rotate90(), p=0.95),
        Random(Shift(), p=.1),
        Random(RotateAndScale(), p=0.1),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.3,
            crop_size_range=(int(input_shape[0] / 1.1), int(input_shape[0] / 0.9)),
            try_range=(1, 5)
        ),
        Resize(*input_shape),
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
                (Contrast().only_on('img'), 0.01), 0.01))
        ),
        Random(ElasticTransform().only_on('img'), p=0.001)
    )


def dpn92_unet_double_tune(input_shape: Tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.3),
        Random(Rotate90(), p=0.7),
        Random(Shift(), p=.01),
        Random(RotateAndScale(), p=0.5),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.5,
            crop_size_range=(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85)),
            try_range=(1, 10),
            scoring_weights={'msk2': 5, 'msk3': 5, 'msk4': 2, 'msk1': 1}
        ),
        Resize(
            height=input_shape[0],
            width=input_shape[1],
        ),
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
        (ElasticTransform().only_on('img_pre'), 0.01),
        (ElasticTransform().only_on('img_post'), 0.01),
        (Dilation().only_on('msk'), 0.9)
    )


def resnet34_unet_double_tune(input_shape: Tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.3),
        Random(Rotate90(), p=0.7),
        Random(Shift(), p=.02),
        Random(RotateAndScale(), p=0.5),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.5,
            crop_size_range=(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85)),
            try_range=(1, 10),
            scoring_weights={'msk2': 5, 'msk3': 5, 'msk4': 2, 'msk1': 1}
        ),
        Resize(
            height=input_shape[0],
            width=input_shape[1],
        ),
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
        (ElasticTransform().only_on('img_pre'), 0.01),
        (ElasticTransform().only_on('img_post'), 0.01),
        (Dilation().only_on('msk'), 0.9)
    )


def senet154_unet_double_tune(input_shape: Tuple[int, int]) -> Pipeline:
    return dpn92_unet_double_tune(input_shape)


def seresnext50_unet_double_tune(input_shape: Tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.3),
        Random(Rotate90(), p=0.7),
        Random(Shift(), p=.01),
        Random(RotateAndScale(), p=0.5),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.5,
            crop_size_range=(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85)),
            try_range=(1, 10),
            scoring_weights={'msk2': 5, 'msk3': 5, 'msk4': 2, 'msk1': 1}
        ),
        Resize(
            height=input_shape[0],
            width=input_shape[1],
        ),
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
        (ElasticTransform().only_on('img_pre'), 0.017),
        (ElasticTransform().only_on('img_post'), 0.017),
        (Dilation().only_on('msk'), 0.9)
    )


def dpn92_unet_localization_tune(input_shape: Tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        Random(VFlip(), p=0.45),
        Random(Rotate90(), p=0.9),
        Random(Shift(), p=.05),
        Random(RotateAndScale(), p=0.05),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.6,
            crop_size_range=(int(input_shape[0] / 1.1), int(input_shape[0] / 0.9)),
            try_range=(1, 5)
        ),
        Resize(*input_shape),
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
                (Contrast().only_on('img'), 0.01), 0.01))
        ),
        Random(ElasticTransform().only_on('img'), p=0.001)
        )


def seresnext50_unet_localization_tune(input_shape: Tuple[int, int]) -> Pipeline:
    return dpn92_unet_double_tune(input_shape)
