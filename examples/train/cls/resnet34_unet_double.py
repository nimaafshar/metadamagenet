import sys
import timeit
import random
import os
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda import amp

from src.train.dataset import ClassificationDataset, ClassificationValidationDataset
from src.file_structure import Dataset as ImageDataset
from src.configs import GeneralConfig
from src.train.cls import ClassificationTrainer, ClassificationRequirements
from src.train.trainer import TrainingConfig
from src.model_config import ModelConfig
from src.zoo.models import Res34_Unet_Double, Res34_Unet_Loc
from src.losses import ComboLoss
from src.setup import set_random_seeds

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

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

set_random_seeds()

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def train():
    np.random.seed(self._config.model_config.seed + 321)
    random.seed(self._config.model_config.seed + 321)

    (DataLoader(self._config.train_dataset,
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



    def _apply_activation(self, model_out: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(model_out)
    def _update_weights(self, loss: torch.Tensor) -> None:
        self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()

    t0 = timeit.default_timer()

    GeneralConfig.load()
    config = GeneralConfig.get_instance()

    seed = int(sys.argv[1])

    input_shape = (608, 608)

    # TODO: count images with 2,3 damage level 2 times
    train_image_dataset = ImageDataset(config.train_dirs)
    train_image_dataset.discover()

    valid_image_data = ImageDataset(config.test_dirs)
    valid_image_data.discover()

    train_dataset = ClassificationDataset(
        image_dataset=train_image_dataset,
        augmentations=, do_dilation=True)

    validation_dataset = ClassificationValidationDataset(
        image_dataset=valid_image_data,
    )

    model_config: ModelConfig = ModelConfig(
        name='res34_cls2',
        empty_model=torch.nn.DataParallel(Res34_Unet_Double().cuda()).cuda(),
        version='0',
        seed=seed
    )

    training_config: TrainingConfig = TrainingConfig(
        model_config=model_config,
        input_shape=input_shape,
        epochs=20,
        batch_size=16,
        val_batch_size=8,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        evaluation_interval=2,
        start_checkpoint=ModelConfig(
            name='res34_loc',
            empty_model=torch.nn.DataParallel(Res34_Unet_Loc().cuda()),
            version='1',
            seed=seed,
        ),
    )

    trainer = Resnet34UnetDoubleTrainer(training_config,
                                        use_cce_loss=False)

    trainer.train()


