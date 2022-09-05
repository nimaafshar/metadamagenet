from typing import Tuple
from metadamagenet.augment import (
    OneOf,
    Pipeline,
    TopDownFlip,
    Rotation90Degree,
    Shift,
    RotateAndScale,
    Resize,
    ShiftRGB,
    ShiftHSV,
    RandomCrop,
    ElasticTransformation,
    GaussianNoise,
    Clahe,
    Blur,
    Saturation,
    Brightness,
    Contrast
)


def dpn92_unet_double(input_shape: Tuple[int, int]) -> Pipeline:
    return Pipeline((
        TopDownFlip(
            probability=0.5
        ),
        Rotation90Degree(
            probability=0.0001
        ),
        Shift(probability=0.5,
              y_range=(-320, 320),
              x_range=(-320, 320)),
        RotateAndScale(
            probability=0.05,
            center_x_range=(-320, 320),
            center_y_range=(-320, 320),
            scale_range=(0.9, 1.1),
            angle_range=(-10, 10)
        ),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.05,
            crop_size_range=(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85)),
            try_range=(1, 10),
            scoring_weights={'msk2': 5, 'msk3': 5, 'msk4': 2, 'msk1': 1}
        ),
        Resize(
            height=input_shape[0],
            width=input_shape[1],
        ),
        OneOf((
            ShiftRGB(
                probability=0.9,
                r_range=(-5, 5),
                g_range=(-5, 5),
                b_range=(-5, 5),
                apply_to=('img_pre',)
            ),
            ShiftRGB(
                probability=0.9,
                r_range=(-5, 5),
                g_range=(-5, 5),
                b_range=(-5, 5),
                apply_to=('img_post',)
            ),
        ), probability=0
        ),
        OneOf((
            ShiftHSV(
                probability=0.9,
                h_range=(-5, 5),
                s_range=(-5, 5),
                v_range=(-5, 5),
                apply_to=('img_pre',)
            ),
            ShiftHSV(
                probability=0.9,
                h_range=(-5, 5),
                s_range=(-5, 5),
                v_range=(-5, 5),
                apply_to=('img_post',)
            ),
        ), probability=0
        ),
        OneOf((
            OneOf((
                Clahe(
                    probability=0.9,
                    apply_to=('img_pre',)
                ),
                GaussianNoise(
                    probability=0.9,
                    apply_to=('img_pre',)
                ),
                Blur(
                    probability=0.9,
                    apply_to=('img_post',)
                )
            ), probability=0.9
            ),
            OneOf((
                Saturation(
                    probability=0.9,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                ),
                Brightness(
                    probability=0.9,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                ),
                Contrast(
                    probability=0.9,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                )
            ), probability=0.9
            )
        ), probability=0
        ),
        OneOf((
            OneOf((
                Clahe(
                    probability=0.9,
                    apply_to=('img_post',)
                ),
                GaussianNoise(
                    probability=0.9,
                    apply_to=('img_post',)
                ),
                Blur(
                    probability=0.9,
                    apply_to=('img_post',)
                )
            ),
                probability=0.9
            ),
            OneOf((
                Saturation(
                    probability=0.9,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                ),
                Brightness(
                    probability=0.9,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                ),
                Contrast(
                    probability=0.9,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                )
            ),
                probability=0.9
            )
        ), probability=0),
        ElasticTransformation(
            probability=0.9,
            apply_to=('img_pre',)
        ),
        ElasticTransformation(
            probability=0.9,
            apply_to=('img_post',)
        )
    ))


def resnet34_unet_double(input_shape: Tuple[int, int]) -> Pipeline:
    return Pipeline((
        TopDownFlip(
            probability=0.5
        ),
        Rotation90Degree(
            probability=0.05
        ),
        Shift(probability=0.9,
              y_range=(-320, 320),
              x_range=(-320, 320)),
        RotateAndScale(
            probability=0.6,
            center_x_range=(-320, 320),
            center_y_range=(-320, 320),
            scale_range=(0.9, 1.1),
            angle_range=(-10, 10)
        ),
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
        OneOf((
            ShiftRGB(
                probability=0.985,
                r_range=(-5, 5),
                g_range=(-5, 5),
                b_range=(-5, 5),
                apply_to=('img_pre',)
            ),
            ShiftRGB(
                probability=0.985,
                r_range=(-5, 5),
                g_range=(-5, 5),
                b_range=(-5, 5),
                apply_to=('img_post',)
            ),
        ), probability=0
        ),
        OneOf((
            ShiftHSV(
                probability=0.985,
                h_range=(-5, 5),
                s_range=(-5, 5),
                v_range=(-5, 5),
                apply_to=('img_pre',)
            ),
            ShiftHSV(
                probability=0.985,
                h_range=(-5, 5),
                s_range=(-5, 5),
                v_range=(-5, 5),
                apply_to=('img_post',)
            ),
        ), probability=0
        ),
        OneOf((
            OneOf((
                Clahe(
                    probability=0.985,
                    apply_to=('img_pre',)
                ),
                GaussianNoise(
                    probability=0.985,
                    apply_to=('img_pre',)
                ),
                Blur(
                    probability=0.985,
                    apply_to=('img_post',)
                )
            ), probability=0.98
            ),
            OneOf((
                Saturation(
                    probability=0.985,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                ),
                Brightness(
                    probability=0.985,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                ),
                Contrast(
                    probability=0.985,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                )
            ), probability=0.98
            )
        ), probability=0
        ),
        OneOf((
            OneOf((
                Clahe(
                    probability=0.985,
                    apply_to=('img_post',)
                ),
                GaussianNoise(
                    probability=0.985,
                    apply_to=('img_post',)
                ),
                Blur(
                    probability=0.985,
                    apply_to=('img_post',)
                )
            ),
                probability=0.98
            ),
            OneOf((
                Saturation(
                    probability=0.985,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                ),
                Brightness(
                    probability=0.985,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                ),
                Contrast(
                    probability=0.985,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                )
            ),
                probability=0.98
            )
        ), probability=0),
        ElasticTransformation(
            probability=0.983,
            apply_to=('img_pre',)
        ),
        ElasticTransformation(
            probability=0.983,
            apply_to=('img_post',)
        )
    ))


def senet154_unet_double(input_shape: Tuple[int, int]) -> Pipeline:
    return dpn92_unet_double(input_shape)


def seresnext50_unet_double(input_shape: Tuple[int, int]) -> Pipeline:
    return Pipeline((
        TopDownFlip(
            probability=0.5
        ),
        Rotation90Degree(
            probability=0.05
        ),
        Shift(probability=0.8,
              y_range=(-320, 320),
              x_range=(-320, 320)),
        RotateAndScale(
            probability=0.2,
            center_x_range=(-320, 320),
            center_y_range=(-320, 320),
            scale_range=(0.9, 1.1),
            angle_range=(-10, 10)
        ),
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
        OneOf((
            ShiftRGB(
                probability=0.96,
                r_range=(-5, 5),
                g_range=(-5, 5),
                b_range=(-5, 5),
                apply_to=('img_pre',)
            ),
            ShiftRGB(
                probability=0.96,
                r_range=(-5, 5),
                g_range=(-5, 5),
                b_range=(-5, 5),
                apply_to=('img_post',)
            ),
        ), probability=0
        ),
        OneOf((
            ShiftHSV(
                probability=0.96,
                h_range=(-5, 5),
                s_range=(-5, 5),
                v_range=(-5, 5),
                apply_to=('img_pre',)
            ),
            ShiftHSV(
                probability=0.96,
                h_range=(-5, 5),
                s_range=(-5, 5),
                v_range=(-5, 5),
                apply_to=('img_post',)
            ),
        ), probability=0
        ),
        OneOf((
            OneOf((
                Clahe(
                    probability=0.96,
                    apply_to=('img_pre',)
                ),
                GaussianNoise(
                    probability=0.96,
                    apply_to=('img_pre',)
                ),
                Blur(
                    probability=0.96,
                    apply_to=('img_post',)
                )
            ), probability=0.9
            ),
            OneOf((
                Saturation(
                    probability=0.96,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                ),
                Brightness(
                    probability=0.96,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                ),
                Contrast(
                    probability=0.96,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                )
            ), probability=0.9
            )
        ), probability=0
        ),
        OneOf((
            OneOf((
                Clahe(
                    probability=0.96,
                    apply_to=('img_post',)
                ),
                GaussianNoise(
                    probability=0.96,
                    apply_to=('img_post',)
                ),
                Blur(
                    probability=0.96,
                    apply_to=('img_post',)
                )
            ),
                probability=0.9
            ),
            OneOf((
                Saturation(
                    probability=0.96,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                ),
                Brightness(
                    probability=0.96,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                ),
                Contrast(
                    probability=0.96,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                )
            ),
                probability=0.9
            )
        ), probability=0),
        ElasticTransformation(
            probability=0.96,
            apply_to=('img_pre',)
        ),
        ElasticTransformation(
            probability=0.96,
            apply_to=('img_post',)
        )
    ))


def dpn92_unet_localization(input_shape: Tuple[int, int]) -> Pipeline:
    return Pipeline((
        TopDownFlip(probability=0.5),
        Rotation90Degree(probability=0.05),
        Shift(probability=0.9,
              y_range=(-320, 320),
              x_range=(-320, 320)),
        RotateAndScale(
            probability=0.9,
            center_y_range=(-320, 320),
            center_x_range=(-320, 320),
            angle_range=(-10, 10),
            scale_range=(0.9, 1.1)
        ),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.5,
            crop_size_range=(int(input_shape[0] / 1.1), int(input_shape[0] / 0.9)),
            try_range=(1, 5)
        ),
        Resize(*input_shape),
        ShiftRGB(probability=0.99,
                 r_range=(-5, 5),
                 g_range=(-5, 5),
                 b_range=(-5, 5)),
        ShiftHSV(probability=0.99,
                 h_range=(-5, 5),
                 s_range=(-5, 5),
                 v_range=(-5, 5)),
        OneOf((
            OneOf((
                Clahe(0.99),
                GaussianNoise(0.99),
                Blur(0.99)),
                probability=0.99),
            OneOf((
                Saturation(0.99, (0.9, 1.1)),
                Brightness(0.99, (0.9, 1.1)),
                Contrast(0.99, (0.9, 1.1))),
                probability=0.99)), probability=0),
        ElasticTransformation(0.999)
    ))


def resnet34_unet_localization(input_shape: Tuple[int, int]) -> Pipeline:
    return Pipeline((
        TopDownFlip(probability=0.5),
        Rotation90Degree(probability=0.05),
        Shift(probability=0.8,
              y_range=(-320, 320),
              x_range=(-320, 320)),
        RotateAndScale(
            probability=0.2,
            center_y_range=(-320, 320),
            center_x_range=(-320, 320),
            angle_range=(-10, 10),
            scale_range=(0.9, 1.1)
        ),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.3,
            crop_size_range=(int(input_shape[0] / 1.2), int(input_shape[0] / 0.8)),
            try_range=(1, 5)
        ),
        Resize(*input_shape),
        OneOf((
            ShiftRGB(probability=0.97,
                     r_range=(-5, 5),
                     g_range=(-5, 5),
                     b_range=(-5, 5)),

            ShiftHSV(probability=0.97,
                     h_range=(-5, 5),
                     s_range=(-5, 5),
                     v_range=(-5, 5))), probability=0),
        OneOf((
            OneOf((
                Clahe(0.97),
                GaussianNoise(0.97),
                Blur(0.98)),
                probability=0.93),
            OneOf((
                Saturation(0.97, (0.9, 1.1)),
                Brightness(0.97, (0.9, 1.1)),
                Contrast(0.97, (0.9, 1.1))),
                probability=0.93)), probability=0),
        ElasticTransformation(0.97)
    ))


def senet154_unet_localization(input_shape: Tuple[int, int]) -> Pipeline:
    return Pipeline((
        TopDownFlip(probability=0.6),
        Rotation90Degree(probability=0.1),
        Shift(probability=0.7,
              y_range=(-320, 320),
              x_range=(-320, 320)),
        RotateAndScale(
            probability=0.4,
            center_y_range=(-320, 320),
            center_x_range=(-320, 320),
            angle_range=(-10, 10),
            scale_range=(0.9, 1.1)
        ),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.2,
            crop_size_range=(int(input_shape[0] / 1.1), int(input_shape[0] / 0.9)),
            try_range=(1, 5)
        ),
        Resize(*input_shape),
        ShiftRGB(probability=0.95,
                 r_range=(-5, 5),
                 g_range=(-5, 5),
                 b_range=(-5, 5)),
        ShiftHSV(probability=0.9597,
                 h_range=(-5, 5),
                 s_range=(-5, 5),
                 v_range=(-5, 5)),
        OneOf((
            OneOf((
                Clahe(0.92),
                GaussianNoise(0.92),
                Blur(0.92)),
                probability=0.92),
            OneOf((
                Saturation(0.92, (0.9, 1.1)),
                Brightness(0.92, (0.9, 1.1)),
                Contrast(0.92, (0.9, 1.1))),
                probability=0.92)), probability=0),
        ElasticTransformation(0.95)
    ))


def seresnext50_unet_localization(input_shape: Tuple[int, int]) -> Pipeline:
    return Pipeline((
        TopDownFlip(probability=0.5),
        Rotation90Degree(probability=0.05),
        Shift(probability=0.9,
              y_range=(-320, 320),
              x_range=(-320, 320)),
        RotateAndScale(
            probability=0.9,
            center_y_range=(-320, 320),
            center_x_range=(-320, 320),
            angle_range=(-10, 10),
            scale_range=(0.9, 1.1)
        ),
        RandomCrop(
            default_crop_size=input_shape[0],
            size_change_probability=0.3,
            crop_size_range=(int(input_shape[0] / 1.1), int(input_shape[0] / 0.9)),
            try_range=(1, 5)
        ),
        Resize(*input_shape),
        ShiftRGB(probability=0.99,
                 r_range=(-5, 5),
                 g_range=(-5, 5),
                 b_range=(-5, 5)),
        ShiftHSV(probability=0.99,
                 h_range=(-5, 5),
                 s_range=(-5, 5),
                 v_range=(-5, 5)),
        OneOf((
            OneOf((
                Clahe(0.99),
                GaussianNoise(0.99),
                Blur(0.99)),
                probability=0.99),
            OneOf((
                Saturation(0.99, (0.9, 1.1)),
                Brightness(0.99, (0.9, 1.1)),
                Contrast(0.99, (0.9, 1.1))),
                probability=0.99)), probability=0),
        ElasticTransformation(0.999)
    ))


def dpn92_unet_double_tune(input_shape: Tuple[int, int]) -> Pipeline:
    return Pipeline((
        TopDownFlip(
            probability=0.7
        ),
        Rotation90Degree(
            probability=0.3
        ),
        Shift(probability=0.99,
              y_range=(-320, 320),
              x_range=(-320, 320)),
        RotateAndScale(
            probability=0.5,
            center_x_range=(-320, 320),
            center_y_range=(-320, 320),
            scale_range=(0.9, 1.1),
            angle_range=(-10, 10)
        ),
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
        OneOf((
            ShiftRGB(
                probability=0.99,
                r_range=(-5, 5),
                g_range=(-5, 5),
                b_range=(-5, 5),
                apply_to=('img_pre',)
            ),
            ShiftRGB(
                probability=0.99,
                r_range=(-5, 5),
                g_range=(-5, 5),
                b_range=(-5, 5),
                apply_to=('img_post',)
            ),
        ), probability=0
        ),
        OneOf((
            ShiftHSV(
                probability=0.99,
                h_range=(-5, 5),
                s_range=(-5, 5),
                v_range=(-5, 5),
                apply_to=('img_pre',)
            ),
            ShiftHSV(
                probability=0.99,
                h_range=(-5, 5),
                s_range=(-5, 5),
                v_range=(-5, 5),
                apply_to=('img_post',)
            ),
        ), probability=0
        ),
        OneOf((
            OneOf((
                Clahe(
                    probability=0.99,
                    apply_to=('img_pre',)
                ),
                GaussianNoise(
                    probability=0.99,
                    apply_to=('img_pre',)
                ),
                Blur(
                    probability=0.99,
                    apply_to=('img_post',)
                )
            ), probability=0.99
            ),
            OneOf((
                Saturation(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                ),
                Brightness(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                ),
                Contrast(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                )
            ), probability=0.99
            )
        ), probability=0
        ),
        OneOf((
            OneOf((
                Clahe(
                    probability=0.99,
                    apply_to=('img_post',)
                ),
                GaussianNoise(
                    probability=0.99,
                    apply_to=('img_post',)
                ),
                Blur(
                    probability=0.99,
                    apply_to=('img_post',)
                )
            ),
                probability=0.99
            ),
            OneOf((
                Saturation(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                ),
                Brightness(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                ),
                Contrast(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                )
            ),
                probability=0.99
            )
        ), probability=0),
        ElasticTransformation(
            probability=0.99,
            apply_to=('img_pre',)
        ),
        ElasticTransformation(
            probability=0.99,
            apply_to=('img_post',)
        )
    ))


def resnet34_unet_double_tune(input_shape: Tuple[int, int]) -> Pipeline:
    return Pipeline((
        TopDownFlip(
            probability=0.7
        ),
        Rotation90Degree(
            probability=0.3
        ),
        Shift(probability=0.98,
              y_range=(-320, 320),
              x_range=(-320, 320)),
        RotateAndScale(
            probability=0.5,
            center_x_range=(-320, 320),
            center_y_range=(-320, 320),
            scale_range=(0.9, 1.1),
            angle_range=(-10, 10)
        ),
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
        OneOf((
            ShiftRGB(
                probability=0.99,
                r_range=(-5, 5),
                g_range=(-5, 5),
                b_range=(-5, 5),
                apply_to=('img_pre',)
            ),
            ShiftRGB(
                probability=0.99,
                r_range=(-5, 5),
                g_range=(-5, 5),
                b_range=(-5, 5),
                apply_to=('img_post',)
            ),
        ), probability=0
        ),
        OneOf((
            ShiftHSV(
                probability=0.99,
                h_range=(-5, 5),
                s_range=(-5, 5),
                v_range=(-5, 5),
                apply_to=('img_pre',)
            ),
            ShiftHSV(
                probability=0.99,
                h_range=(-5, 5),
                s_range=(-5, 5),
                v_range=(-5, 5),
                apply_to=('img_post',)
            ),
        ), probability=0
        ),
        OneOf((
            OneOf((
                Clahe(
                    probability=0.99,
                    apply_to=('img_pre',)
                ),
                GaussianNoise(
                    probability=0.99,
                    apply_to=('img_pre',)
                ),
                Blur(
                    probability=0.99,
                    apply_to=('img_post',)
                )
            ), probability=0.99
            ),
            OneOf((
                Saturation(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                ),
                Brightness(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                ),
                Contrast(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                )
            ), probability=0.99
            )
        ), probability=0
        ),
        OneOf((
            OneOf((
                Clahe(
                    probability=0.985,
                    apply_to=('img_post',)
                ),
                GaussianNoise(
                    probability=0.985,
                    apply_to=('img_post',)
                ),
                Blur(
                    probability=0.985,
                    apply_to=('img_post',)
                )
            ),
                probability=0.99
            ),
            OneOf((
                Saturation(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                ),
                Brightness(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                ),
                Contrast(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                )
            ),
                probability=0.99
            )
        ), probability=0),
        ElasticTransformation(
            probability=0.99,
            apply_to=('img_pre',)
        ),
        ElasticTransformation(
            probability=0.99,
            apply_to=('img_post',)
        )
    ))


def senet154_unet_double_tune(input_shape: Tuple[int, int]) -> Pipeline:
    return dpn92_unet_double_tune(input_shape)


def seresnext50_unet_double_tune(input_shape: Tuple[int, int]) -> Pipeline:
    return Pipeline((
        TopDownFlip(
            probability=0.7
        ),
        Rotation90Degree(
            probability=0.3
        ),
        Shift(probability=0.99,
              y_range=(-320, 320),
              x_range=(-320, 320)),
        RotateAndScale(
            probability=0.5,
            center_x_range=(-320, 320),
            center_y_range=(-320, 320),
            scale_range=(0.9, 1.1),
            angle_range=(-10, 10)
        ),
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
        OneOf((
            ShiftRGB(
                probability=0.99,
                r_range=(-5, 5),
                g_range=(-5, 5),
                b_range=(-5, 5),
                apply_to=('img_pre',)
            ),
            ShiftRGB(
                probability=0.99,
                r_range=(-5, 5),
                g_range=(-5, 5),
                b_range=(-5, 5),
                apply_to=('img_post',)
            ),
        ), probability=0
        ),
        OneOf((
            ShiftHSV(
                probability=0.99,
                h_range=(-5, 5),
                s_range=(-5, 5),
                v_range=(-5, 5),
                apply_to=('img_pre',)
            ),
            ShiftHSV(
                probability=0.99,
                h_range=(-5, 5),
                s_range=(-5, 5),
                v_range=(-5, 5),
                apply_to=('img_post',)
            ),
        ), probability=0
        ),
        OneOf((
            OneOf((
                Clahe(
                    probability=0.96,
                    apply_to=('img_pre',)
                ),
                GaussianNoise(
                    probability=0.96,
                    apply_to=('img_pre',)
                ),
                Blur(
                    probability=0.96,
                    apply_to=('img_post',)
                )
            ), probability=0.99
            ),
            OneOf((
                Saturation(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                ),
                Brightness(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                ),
                Contrast(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_pre',)
                )
            ), probability=0.99
            )
        ), probability=0
        ),
        OneOf((
            OneOf((
                Clahe(
                    probability=0.99,
                    apply_to=('img_post',)
                ),
                GaussianNoise(
                    probability=0.99,
                    apply_to=('img_post',)
                ),
                Blur(
                    probability=0.99,
                    apply_to=('img_post',)
                )
            ),
                probability=0.99
            ),
            OneOf((
                Saturation(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                ),
                Brightness(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                ),
                Contrast(
                    probability=0.99,
                    alpha_range=(0.9, 1.1),
                    apply_to=('img_post',)
                )
            ),
                probability=0.99
            )
        ), probability=0),
        ElasticTransformation(
            probability=0.99,
            apply_to=('img_pre',)
        ),
        ElasticTransformation(
            probability=0.99,
            apply_to=('img_post',)
        )
    ))


def dpn92_unet_localization_tune(input_shape: Tuple[int, int]) -> Pipeline:
    return Pipeline(
        (
            TopDownFlip(probability=0.55),
            Rotation90Degree(probability=0.1),
            Shift(probability=0.95,
                  y_range=(-320, 320),
                  x_range=(-320, 320)),
            RotateAndScale(
                probability=0.95,
                center_y_range=(-320, 320),
                center_x_range=(-320, 320),
                angle_range=(-10, 10),
                scale_range=(0.9, 1.1)
            ),
            RandomCrop(
                default_crop_size=input_shape[0],
                size_change_probability=0.6,
                crop_size_range=(int(input_shape[0] / 1.1), int(input_shape[0] / 0.9)),
                try_range=(1, 5)
            ),
            Resize(*input_shape),
            ShiftRGB(probability=0.99,
                     r_range=(-5, 5),
                     g_range=(-5, 5),
                     b_range=(-5, 5)),
            ShiftHSV(probability=0.99,
                     h_range=(-5, 5),
                     s_range=(-5, 5),
                     v_range=(-5, 5)),
            OneOf((
                OneOf((
                    Clahe(0.99),
                    GaussianNoise(0.99),
                    Blur(0.99)),
                    probability=0.99),
                OneOf((
                    Saturation(0.99, (0.9, 1.1)),
                    Brightness(0.99, (0.9, 1.1)),
                    Contrast(0.99, (0.9, 1.1))),
                    probability=0.99)), probability=0),
            ElasticTransformation(0.999)
        ))


def seresnext50_unet_localization_tune(input_shape: Tuple[int, int]) -> Pipeline:
    return dpn92_unet_double_tune(input_shape)
