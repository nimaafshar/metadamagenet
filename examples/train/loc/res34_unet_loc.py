import os
import sys
import timeit
import random

import numpy as np
import cv2
import torch
from torch import nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.cuda import amp

from src.zoo.models import Res34_Unet_Loc

from src.losses import ComboLoss
from src.train.dataset import LocalizationDataset
from src.train.loc import LocalizationRequirements, LocalizationTrainer
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
from src.configs import GeneralConfig
from src.setup import set_random_seeds
from src.logs import log

set_random_seeds()

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"


class Resnet34UnetLocTrainer(LocalizationTrainer):

    def _setup(self):
        super(Resnet34UnetLocTrainer, self)._setup()
        np.random.seed(self._config.model_config.seed + 545)
        random.seed(self._config.model_config.seed + 454)

    def _get_dataloaders(self) -> (DataLoader, DataLoader):
        return (DataLoader(self._config.train_dataset,
                           batch_size=self._config.batch_size,
                           num_workers=6,
                           shuffle=True,
                           pin_memory=True,
                           drop_last=True),

                DataLoader(self._config.validation_dataset,
                           batch_size=self._config.val_batch_size,
                           num_workers=6,
                           shuffle=False,
                           pin_memory=True))

    def _get_requirements(self) -> LocalizationRequirements:
        model: nn.Module = self._get_model()
        optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.00015,
                                     weight_decay=1e-6)
        lr_scheduler: MultiStepLR = MultiStepLR(optimizer,
                                                milestones=[5, 11, 17, 25, 33, 47, 50,
                                                            60, 70, 90, 110, 130, 150,
                                                            170, 180, 190],
                                                gamma=0.5)
        seg_loss: ComboLoss = ComboLoss({'dice': 1.0, 'focal': 10.0}, per_image=False).cuda()
        return LocalizationRequirements(
            model,
            optimizer,
            lr_scheduler,
            seg_loss,
            amp.GradScaler()
        )

    def _update_weights(self, loss: torch.Tensor) -> None:
        self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()


if __name__ == '__main__':
    t0 = timeit.default_timer()

    GeneralConfig.load()

    config = GeneralConfig.get_instance()

    seed = int(sys.argv[1])

    input_shape = (736, 736)

    train_image_dataset = ImageDataset(config.train_dirs)
    train_image_dataset.discover()

    valid_image_data = ImageDataset(config.test_dirs)
    valid_image_data.discover()

    train_data = LocalizationDataset(
        image_dataset=train_image_dataset,
        augmentations=Pipeline(
            (
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
    )

    vali_data = LocalizationDataset(
        image_dataset=valid_image_data,
        augmentations=None,
        post_version_prob=1
    )

    model_config = ModelConfig(
        name='res34_loc',
        empty_model=torch.nn.DataParallel(Res34_Unet_Loc().cuda()).cuda(),
        version='1',
        seed=seed
    )

    config = TrainingConfig(
        model_config=model_config,
        input_shape=input_shape,
        epochs=55,
        batch_size=16,
        val_batch_size=8,
        train_dataset=train_data,
        validation_dataset=vali_data,
        evaluation_interval=2
    )

    trainer = Resnet34UnetLocTrainer(config)

    trainer.train()

    elapsed = timeit.default_timer() - t0
    log(':hourglass: : {:.3f} min'.format(elapsed / 60))
