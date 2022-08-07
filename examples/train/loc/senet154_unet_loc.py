import os
import sys
import numpy as np

import random
import timeit
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from apex import amp

from src.zoo.models import SeNet154_Unet_Loc

from src.train.dataset import Dataset
from src.optim import AdamW
from src.train.loc import LocalizationTrainer, LocalizationRequirements
from src.train.trainer import TrainingConfig
from src.model_config import ModelConfig
from src.losses import ComboLoss
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


class SEResNext50UnetLocTrainer(LocalizationTrainer):
    def _setup(self):
        super(SEResNext50UnetLocTrainer, self)._setup()
        np.random.seed(self._config.model_config.seed + 321)
        random.seed(self._config.model_config.seed + 321)

    def _get_dataloaders(self) -> (DataLoader, DataLoader):
        return (DataLoader(self._config.train_dataset,
                           batch_size=self._config.batch_size,
                           num_workers=6,
                           shuffle=True,
                           pin_memory=False,
                           drop_last=True),
                DataLoader(self._config.validation_dataset,
                           batch_size=self._config.val_batch_size,
                           num_workers=6,
                           shuffle=False,
                           pin_memory=False))

    def _get_requirements(self) -> LocalizationRequirements:
        model = self._get_model().cuda()
        optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.00015,
                                     weight_decay=1e-6)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        lr_scheduler = MultiStepLR(optimizer,
                                   milestones=[3, 7, 11, 15, 19, 23, 27, 33, 41, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190],
                                   gamma=0.5)
        model = nn.DataParallel(model).cuda()
        seg_loss = ComboLoss({'dice': 1.0, 'focal': 14.0}, per_image=False).cuda()
        return LocalizationRequirements(
            model,
            optimizer,
            lr_scheduler,
            seg_loss
        )

    def _update_weights(self, loss: torch.Tensor) -> None:
        self._optimizer.zero_grad()

        with amp.scale_loss(loss, self._optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(self._optimizer), 0.999)
        self._optimizer.step()


if __name__ == '__main__':
    t0 = timeit.default_timer()

    seed = int(sys.argv[1])

    input_shape = (480, 480)

    train_image_dataset = ImageDataset(configs.TRAIN_DIRS)
    train_image_dataset.discover()

    train_data = Dataset(
        image_dataset=train_image_dataset,
        augmentations=Pipeline(
            (
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
            )),
        post_version_prob=0.96
    )

    valid_image_data = ImageDataset((configs.TEST_DIR,))
    valid_image_data.discover()

    vali_data = Dataset(
        image_dataset=valid_image_data,
        augmentations=None,
        post_version_prob=1
    )

    model_config = ModelConfig(
        name='se154_loc',
        model_type=SeNet154_Unet_Loc,
        version='1',
        seed=seed
    )

    config = TrainingConfig(
        model_config=model_config,
        input_shape=input_shape,
        epochs=30,
        batch_size=14,
        val_batch_size=4,
        train_dataset=train_data,
        validation_dataset=vali_data,
        evaluation_interval=1
    )

    trainer = SEResNext50UnetLocTrainer(config)

    trainer.train()

    elapsed = timeit.default_timer() - t0
    log(':hourglass: : {:.3f} min'.format(elapsed / 60))
