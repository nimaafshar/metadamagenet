import os
import sys
import numpy as np
from typing import Optional

import random
import timeit
import cv2
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda import amp
from torch import nn

from src.zoo.models import Dpn92_Unet_Loc

from src.train.dataset import LocalizationDataset
from src.train.loc import LocalizationTrainer, Requirements
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
from src.configs import GeneralConfig
from src.logs import log

random.seed(1)
np.random.seed(1)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"


class Dpn92UnetLocTuner(LocalizationTrainer):
    def _setup(self):
        super(Dpn92UnetLocTuner, self)._setup()
        np.random.seed(self._config.model_config.seed + 156)
        random.seed(self._config.model_config.seed + 156)

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

    def _get_requirements(self) -> Requirements:
        model: nn.Module
        best_score: Optional[float]
        start_epoch: int
        model, best_score, start_epoch = self._get_model()
        model = model.cuda()

        optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.00004,
                                     weight_decay=1e-6)

        lr_scheduler = MultiStepLR(optimizer,
                                   milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130,
                                               150, 170, 180, 190],
                                   gamma=0.5)
        seg_loss = ComboLoss({'dice': 1.0, 'focal': 6.0}, per_image=False).cuda()
        return Requirements(
            model,
            optimizer,
            lr_scheduler,
            seg_loss,
            grad_scaler=amp.GradScaler(),
            model_score=best_score,
            start_epoch=start_epoch
        )

    def _update_weights(self, loss: torch.Tensor) -> None:
        self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.1)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()


if __name__ == '__main__':
    t0 = timeit.default_timer()

    GeneralConfig.load()
    config = GeneralConfig.get_instance()

    seed = int(sys.argv[1])

    input_shape = (512, 512)

    # using full train dataset to tune the model
    train_image_dataset = ImageDataset(config.train_dirs)
    train_image_dataset.discover()

    valid_image_data = ImageDataset(config.test_dirs)
    valid_image_data.discover()

    train_data = LocalizationDataset(
        image_dataset=train_image_dataset,
        augmentations=Pipeline(
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
            )),
        post_version_prob=1
    )

    vali_data = LocalizationDataset(
        image_dataset=valid_image_data,
        augmentations=None,
        post_version_prob=1
    )

    model_config = ModelConfig(
        name='dpn92_loc',
        empty_model=Dpn92_Unet_Loc().cuda(),
        version='tuned',
        seed=seed
    )

    start_checkpoint = ModelConfig(
        name='dpn92_loc',
        empty_model=Dpn92_Unet_Loc().cuda(),
        version='0',
        seed=0
    )

    config = TrainingConfig(
        model_config=model_config,
        input_shape=input_shape,
        epochs=8,
        batch_size=10,
        val_batch_size=4,
        train_dataset=train_data,
        validation_dataset=vali_data,
        evaluation_interval=1,
        start_checkpoint=start_checkpoint
    )

    trainer = Dpn92UnetLocTuner(config)

    trainer.train()

    elapsed = timeit.default_timer() - t0
    log(':hourglass: : {:.3f} min'.format(elapsed / 60))
