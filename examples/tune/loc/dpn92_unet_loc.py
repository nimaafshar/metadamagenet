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
        augmentations=,
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
