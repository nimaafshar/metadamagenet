import os
import sys
import numpy as np
from typing import Optional

import random
import timeit
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda import amp

from src.zoo.models import SeNet154_Unet_Loc

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

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


class SENet154UnetLocTrainer(LocalizationTrainer):
    def _setup(self):
        super(SENet154UnetLocTrainer, self)._setup()
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

    def _get_requirements(self) -> Requirements:
        model: nn.Module
        best_score: Optional[float]
        start_epoch: int
        model, best_score, start_epoch = self._get_model()
        model = model.cuda()

        optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.00015,
                                     weight_decay=1e-6)
        lr_scheduler = MultiStepLR(optimizer,
                                   milestones=[3, 7, 11, 15, 19, 23, 27, 33, 41, 50, 60, 70, 90, 110, 130, 150, 170,
                                               180, 190],
                                   gamma=0.5)

        return Requirements(
            model,
            optimizer,
            lr_scheduler,
            seg_loss,
            amp.GradScaler(),
            model_score=best_score,
            start_epoch=start_epoch
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

    input_shape = (480, 480)

    train_image_dataset = ImageDataset(config.train_dirs)
    train_image_dataset.discover()

    valid_image_data = ImageDataset(config.test_dirs)
    valid_image_data.discover()

    train_data = LocalizationDataset(
        image_dataset=train_image_dataset,
        augmentations=,
        post_version_prob=0.96
    )

    vali_data = LocalizationDataset(
        image_dataset=valid_image_data,
        augmentations=None,
        post_version_prob=1
    )

    model_config = ModelConfig(
        name='se154_loc',
        empty_model=torch.nn.DataParallel(SeNet154_Unet_Loc().cuda()).cuda(),
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

    trainer = SENet154UnetLocTrainer(config)

    trainer.train()

    elapsed = timeit.default_timer() - t0
    log(':hourglass: : {:.3f} min'.format(elapsed / 60))
