import argparse
import timeit

import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from src.validate import LocalizationValidator, ValidationConfig
from src.configs import GeneralConfig
from src.file_structure import Dataset as ImageDataset
from src.train.dataset import LocalizationDataset
from src.model_config import ModelConfig
from src.zoo.models import Res34_Unet_Loc
from src.logs import log
from src.augment import FourFlips


class Resnet34LocValidator(LocalizationValidator):
    def _setup(self):
        super(Resnet34LocValidator, self)._setup()
        np.random.seed(self._config.model_config.seed + 545)
        random.seed(self._config.model_config.seed + 454)


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("seed", help="model configuration seed", type=int)
    parser.add_argument("--test-time-augment", help="use test-time augmentation", action="store_true")
    args = parser.parse_args()

    t0 = timeit.default_timer()
    GeneralConfig.load()
    config = GeneralConfig.get_instance()
    image_dataset: ImageDataset = ImageDataset(config.test_dirs)
    image_dataset.discover()
    dataset: LocalizationDataset = LocalizationDataset(image_dataset, post_version_prob=1)
    dataloader: DataLoader = DataLoader(dataset,
                                        batch_size=4,
                                        num_workers=6,
                                        shuffle=False,
                                        pin_memory=True)
    model_config = ModelConfig(
        name='res34_loc',
        empty_model=torch.nn.DataParallel(Res34_Unet_Loc().cuda()).cuda(),
        version='1',
        seed=args.seed
    )
    validation_config = ValidationConfig(
        model_config=model_config,
        dataloader=dataloader,
        test_time_augmentor=FourFlips() if args.test_time_augment else None
    )
    validator: LocalizationValidator = Resnet34LocValidator(validation_config)
    validator.validate()
    elapsed = timeit.default_timer() - t0
    log(':hourglass: : {:.3f} min'.format(elapsed / 60))
