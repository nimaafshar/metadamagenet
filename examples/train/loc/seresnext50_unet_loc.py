import os
import sys
import numpy as np

import random
import timeit
import cv2
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer,AdamW
from torch.optim.lr_scheduler import MultiStepLR
from apex import amp

from src.zoo.models import SeResNext50_Unet_Loc

from src.train.dataset import LocalizationDataset
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
from src.configs import GeneralConfig
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
        np.random.seed(self._config.model_config.seed + 123)
        random.seed(self._config.model_config.seed + 123)

    def _get_dataloaders(self) -> (DataLoader, DataLoader):
        return (DataLoader(self._config.train_dataset,
                           batch_size=self._config.batch_size,
                           num_workers=5,
                           shuffle=True,
                           pin_memory=False,
                           drop_last=True),
                DataLoader(self._config.validation_dataset,
                           batch_size=self._config.val_batch_size,
                           num_workers=5,
                           shuffle=False,
                           pin_memory=False))

    def _get_requirements(self) -> LocalizationRequirements:
        model = self._get_model().cuda()
        optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.00015,
                                     weight_decay=1e-6)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        lr_scheduler = MultiStepLR(optimizer,
                                   milestones=[15, 29, 43, 53, 65, 80, 90, 100, 110, 130, 150, 170, 180, 190],
                                   gamma=0.5)
        seg_loss = ComboLoss({'dice': 1.0, 'focal': 10.0}, per_image=False).cuda()
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
        torch.nn.utils.clip_grad_norm_(amp.master_params(self._optimizer), 1.1)
        self._optimizer.step()


if __name__ == '__main__':
    t0 = timeit.default_timer()

    GeneralConfig.load()
    config = GeneralConfig.get_instance()


    seed = int(sys.argv[1])

    input_shape = (512, 512)

    train_image_dataset = ImageDataset(config.test_dirs)
    train_image_dataset.discover()

    valid_image_data = ImageDataset(config.test_dirs)
    valid_image_data.discover()

    train_data = LocalizationDataset(
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

    vali_data = LocalizationDataset(
        image_dataset=valid_image_data,
        augmentations=None,
        post_version_prob=1
    )

    model_config = ModelConfig(
        name='res50_loc',
        model_type=SeResNext50_Unet_Loc,
        version='0',
        seed=seed
    )

    config = TrainingConfig(
        model_config=model_config,
        input_shape=input_shape,
        epochs=150,
        batch_size=15,
        val_batch_size=4,
        train_dataset=train_data,
        validation_dataset=vali_data,
        evaluation_interval=2
    )

    trainer = SEResNext50UnetLocTrainer(config)

    trainer.train()

    elapsed = timeit.default_timer() - t0
    log(':hourglass: : {:.3f} min'.format(elapsed / 60))
