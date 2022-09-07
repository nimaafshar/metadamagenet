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
from src.zoo.models import Res34_Unet_Double
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


class Resnet34UnetDoubleTrainer(ClassificationTrainer):

    def _apply_activation(self, model_out: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(model_out)

    def _setup(self):
        super(Resnet34UnetDoubleTrainer, self)._setup()
        np.random.seed(self._config.model_config.seed + 357)
        random.seed(self._config.model_config.seed + 357)

    def _get_requirements(self) -> ClassificationRequirements:
        model: nn.Module
        best_score: Optional[float]
        start_epoch: int
        model, best_score, start_epoch = self._get_model()
        model = model.cuda()




        return ClassificationRequirements(
            model,
            optimizer,
            lr_scheduler,
            seg_loss,
            grad_scaler=amp.GradScaler(),
            model_score=best_score,
            start_epoch=start_epoch,
            ce_loss=ce_loss,
            label_loss_weights=np.array([0.05, 0.2, 0.8, 0.7, 0.4])
        )



if __name__ == '__main__':
    t0 = timeit.default_timer()

    GeneralConfig.load()
    config = GeneralConfig.get_instance()

    seed = int(sys.argv[1])

    input_shape = (608, 608)

    # in tuning, it's better to train with all available train images
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
        version='tuned',
        seed=seed
    )

    training_config: TrainingConfig = TrainingConfig(
        model_config=model_config,
        input_shape=input_shape,
        epochs=3,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        evaluation_interval=1,
        start_checkpoint=ModelConfig(
            name='res34_cls2',
            empty_model=torch.nn.DataParallel(Res34_Unet_Double().cuda()).cuda(),
            version='0',
            seed=seed,
        ),
    )

    trainer = Resnet34UnetDoubleTrainer(training_config, use_cce_loss=False)

    trainer.train()
