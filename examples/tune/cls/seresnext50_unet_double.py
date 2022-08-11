import sys
import timeit
import random
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from apex import amp

from src.train.dataset import ClassificationDataset, ClassificationValidationDataset
from src.file_structure import Dataset as ImageDataset
from src import configs
from src.optim import AdamW
from src.train.cls import ClassificationTrainer, ClassificationRequirements
from src.train.trainer import TrainingConfig
from src.model_config import ModelConfig
from src.zoo.models import SeResNext50_Unet_Double, SeResNext50_Unet_Loc
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


class SEResnext50UnetDoubleTrainer(ClassificationTrainer):

    def _setup(self):
        super(SEResnext50UnetDoubleTrainer, self)._setup()
        np.random.seed(self._config.model_config.seed + 131313)
        random.seed(self._config.model_config.seed + 131313)

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

    def _get_requirements(self) -> ClassificationRequirements:
        model: nn.Module = self._get_model()

        optimizer: Optimizer = AdamW(model.parameters(),
                                     lr=0.00001,
                                     weight_decay=1e-6)

        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        lr_scheduler: MultiStepLR = MultiStepLR(optimizer,
                                                milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90,
                                                            110, 130, 150, 170, 180, 190],
                                                gamma=0.5)
        model: nn.Module = nn.DataParallel(model).cuda()

        seg_loss: ComboLoss = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).cuda()
        ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss().cuda()
        return ClassificationRequirements(
            model,
            optimizer,
            lr_scheduler,
            seg_loss,
            ce_loss,
            label_loss_weights=np.array([0.1, 0.1, 0.3, 0.3, 0.2, 11]),
            dice_metric_calculator=F1ScoreCalculator()
        )

    def _apply_activation(self, model_out: torch.Tensor) -> torch.Tensor:
        return torch.softmax(model_out, dim=1)

    def _update_weights(self, loss: torch.Tensor) -> None:
        self._optimizer.zero_grad()
        with amp.scale_loss(loss, self._optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(self._optimizer), 0.999)
        self._optimizer.step()


if __name__ == '__main__':
    t0 = timeit.default_timer()

    seed = int(sys.argv[1])

    input_shape = (512, 512)

    # TODO: count images with 2,3 damage level 2 times
    # train with all train data
    train_image_dataset = ImageDataset((configs.TRAIN_SPLIT,))
    train_image_dataset.discover()

    valid_image_data = ImageDataset((configs.VALIDATION_SPLIT,))
    valid_image_data.discover()

    train_dataset = ClassificationDataset(
        image_dataset=train_image_dataset,
        inverse_msk0=True,
        augmentations=Pipeline((
            TopDownFlip(
                probability=0.7
            ),
            Rotation90Degree(
                probability=0.3
            ),
            Shift(probability=0.99,
                  y_range=(-320, 320),
                  x_range=(-320, 320)),
            RotateAndScale(
                probability=0.5,
                center_x_range=(-320, 320),
                center_y_range=(-320, 320),
                scale_range=(0.9, 1.1),
                angle_range=(-10, 10)
            ),
            RandomCrop(
                default_crop_size=input_shape[0],
                size_change_probability=0.5,
                crop_size_range=(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85)),
                try_range=(1, 10),
                scoring_weights={'msk2': 5, 'msk3': 5, 'msk4': 2, 'msk1': 1}
            ),
            Resize(
                height=input_shape[0],
                width=input_shape[1],
            ),
            OneOf((
                ShiftRGB(
                    probability=0.99,
                    r_range=(-5, 5),
                    g_range=(-5, 5),
                    b_range=(-5, 5),
                    apply_to=('img_pre',)
                ),
                ShiftRGB(
                    probability=0.99,
                    r_range=(-5, 5),
                    g_range=(-5, 5),
                    b_range=(-5, 5),
                    apply_to=('img_post',)
                ),
            ), probability=0
            ),
            OneOf((
                ShiftHSV(
                    probability=0.99,
                    h_range=(-5, 5),
                    s_range=(-5, 5),
                    v_range=(-5, 5),
                    apply_to=('img_pre',)
                ),
                ShiftHSV(
                    probability=0.99,
                    h_range=(-5, 5),
                    s_range=(-5, 5),
                    v_range=(-5, 5),
                    apply_to=('img_post',)
                ),
            ), probability=0
            ),
            OneOf((
                OneOf((
                    Clahe(
                        probability=0.96,
                        apply_to=('img_pre',)
                    ),
                    GaussianNoise(
                        probability=0.96,
                        apply_to=('img_pre',)
                    ),
                    Blur(
                        probability=0.96,
                        apply_to=('img_post',)
                    )
                ), probability=0.99
                ),
                OneOf((
                    Saturation(
                        probability=0.99,
                        alpha_range=(0.9, 1.1),
                        apply_to=('img_pre',)
                    ),
                    Brightness(
                        probability=0.99,
                        alpha_range=(0.9, 1.1),
                        apply_to=('img_pre',)
                    ),
                    Contrast(
                        probability=0.99,
                        alpha_range=(0.9, 1.1),
                        apply_to=('img_pre',)
                    )
                ), probability=0.99
                )
            ), probability=0
            ),
            OneOf((
                OneOf((
                    Clahe(
                        probability=0.99,
                        apply_to=('img_post',)
                    ),
                    GaussianNoise(
                        probability=0.99,
                        apply_to=('img_post',)
                    ),
                    Blur(
                        probability=0.99,
                        apply_to=('img_post',)
                    )
                ),
                    probability=0.99
                ),
                OneOf((
                    Saturation(
                        probability=0.99,
                        alpha_range=(0.9, 1.1),
                        apply_to=('img_post',)
                    ),
                    Brightness(
                        probability=0.99,
                        alpha_range=(0.9, 1.1),
                        apply_to=('img_post',)
                    ),
                    Contrast(
                        probability=0.99,
                        alpha_range=(0.9, 1.1),
                        apply_to=('img_post',)
                    )
                ),
                    probability=0.99
                )
            )
                , probability=0),
            ElasticTransformation(
                probability=0.99,
                apply_to=('img_pre',)
            ),
            ElasticTransformation(
                probability=0.99,
                apply_to=('img_post',)
            )
        ))
        , do_dilation=True)

    validation_dataset = ClassificationValidationDataset(
        image_dataset=valid_image_data,
    )

    model_config: ModelConfig = ModelConfig(
        name='res50_cls_cce',
        model_type=SeResNext50_Unet_Double,
        version='tuned',
        seed=seed
    )

    training_config: TrainingConfig = TrainingConfig(
        model_config=model_config,
        input_shape=input_shape,
        epochs=2,
        batch_size=16,
        val_batch_size=4,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        evaluation_interval=2,
        start_checkpoint=ModelConfig(
            name='res50_cls_cce',
            model_type=SeResNext50_Unet_Double,
            version='0',
            seed=seed
        ),
    )

    trainer = SEResnext50UnetDoubleTrainer(
        training_config,
        use_cce_loss=True
    )
    # use_cce_loss means inverse the msk0 because in order to use cross-entropy loss
    # for 0-5 masks we have to flag pixels that are not in any damage boundary with
    # value 1 in mask 0. so that the cross entropy loss forces other values (mask 1-4) to be 0
