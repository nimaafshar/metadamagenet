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
from src.zoo.models import Dpn92_Unet_Loc, Dpn92_Unet_Double
from src.losses import ComboLoss
from src.setup import set_random_seeds
from src.train.metrics import F1ScoreCalculator

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
    np.random.seed(self._config.model_config.seed + 54321)
    random.seed(self._config.model_config.seed + 54321)

    t0 = timeit.default_timer()

    GeneralConfig.load()
    config = GeneralConfig.get_instance()

    seed = int(sys.argv[1])

    input_shape = (512, 512)

    train_image_dataset = ImageDataset(config.train_dirs)
    train_image_dataset.discover()

    valid_image_data = ImageDataset(config.test_dirs)
    valid_image_data.discover()

    DataLoader(self._config.train_dataset,
               batch_size=self._config.batch_size,
               num_workers=6,
               shuffle=True,
               pin_memory=False,
               drop_last=True),

    DataLoader(self._config.validation_dataset,
               batch_size=self._config.val_batch_size,
               num_workers=6,
               shuffle=False,
               pin_memory=False)

    train_dataset = ClassificationDataset(
        image_dataset=train_image_dataset,
        inverse_msk0=True,
        augmentations=, do_dilation=True)

    validation_dataset = ClassificationValidationDataset(
        image_dataset=valid_image_data,
    )

    model_config: ModelConfig = ModelConfig(
        name='dpn92_cls_cce',
        empty_model=torch.nn.DataParallel(Dpn92_Unet_Double().cuda()),
        version='1',
        seed=seed
    )



    training_config: TrainingConfig = TrainingConfig(
        model_config=model_config,
        input_shape=input_shape,
        epochs=10,
        batch_size=12,
        val_batch_size=4,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        evaluation_interval=2,
        start_checkpoint=ModelConfig(
            name='dpn92_loc',
            empty_model=Dpn92_Unet_Loc().cuda(),
            version='0',
            seed=seed,
        ),
    )

    trainer = Dpn92UnetDoubleTrainer(
        training_config,
        use_cce_loss=True
    )
    # use_cce_loss means inverse the msk0 because in order to use cross-entropy loss
    # for 0-5 masks we have to flag pixels that are not in any damage boundary with
    # value 1 in mask 0. so that the cross entropy loss forces other values (mask 1-4) to be 0

    def _apply_activation(self, model_out: torch.Tensor) -> torch.Tensor:
        return torch.softmax(model_out, dim=1)


    dice_metric_calculator = F1ScoreCalculator()

    def _update_weights(self, loss: torch.Tensor) -> None:
        self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        self._grad_scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.999)
        self._grad_scaler.step(self._optimizer)
        self._grad_scaler.update()
