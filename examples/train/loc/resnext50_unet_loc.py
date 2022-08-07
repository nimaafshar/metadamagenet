import os
import sys
import numpy as np

import random
import timeit
import cv2

from src.zoo.models import SeResNext50_Unet_Loc
from src.train.dataset import Dataset
from src.train.loc import LocalizationTrainer
from src.train.trainer import TrainingConfig
from src.model_config import ModelConfig
from src.file_structure import Dataset as ImageDataset
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
from src import configs
from src.logs import log

random.seed(1)
np.random.seed(1)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

input_shape = (736, 736)

if __name__ == '__main__':
    t0 = timeit.default_timer()

    seed = int(sys.argv[1])

    input_shape = (512, 512)

    train_image_dataset = ImageDataset(configs.TRAIN_DIRS)
    train_image_dataset.discover()

    train_data = Dataset(
        image_dataset=train_image_dataset,
        augmentations=Pipeline(
            (
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
    )
    #
    # valid_image_data = ImageDataset((configs.TEST_DIR,))
    # valid_image_data.discover()
    #
    # vali_data = Dataset(
    #     image_dataset=valid_image_data,
    #     augmentations=None,
    #     post_version_prob=1
    # )
    #
    # model_config = ModelConfig(
    #     name='res34_loc',
    #     model_type=Res34_Unet_Loc,
    #     version='1',
    #     seed=seed
    # )
    #
    # config = TrainingConfig(
    #     model_config=model_config,
    #     input_shape=input_shape,
    #     epochs=55,
    #     batch_size=16,
    #     val_batch_size=8,
    #     train_dataset=train_data,
    #     validation_dataset=vali_data,
    #     evaluation_interval=2
    # )
    #
    # trainer = LocalizationTrainer(config)
    #
    # trainer.train()
    #
    # elapsed = timeit.default_timer() - t0
    # log(':hourglass: : {:.3f} min'.format(elapsed / 60))
