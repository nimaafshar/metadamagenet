import sys
import timeit

from src.train.dataset import ClassificationDataset, ClassificationValidationDataset
from src.file_structure import Dataset as ImageDataset
from src import configs

from src.augment import (
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

if __name__ == '__main__':
    t0 = timeit.default_timer()

    seed = int(sys.argv[1])

    input_shape = (608, 608)

    train_image_dataset = ImageDataset((configs.TRAIN_SPLIT,))
    train_image_dataset.discover()

    valid_image_data = ImageDataset((configs.VALIDATION_SPLIT,))
    valid_image_data.discover()

    train_dataset = ClassificationDataset(
        image_dataset=train_image_dataset,
        augmentations=Pipeline((
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
            )
                , probability=0),
            ElasticTransformation(
                probability=0.983,
                apply_to=('img_pre',)
            ),
            ElasticTransformation(
                probability=0.983,
                apply_to=('img_post',)
            )
        ))
        , do_dilation=True)

    validation_dataset = ClassificationValidationDataset(
        image_dataset=valid_image_data,
    )
