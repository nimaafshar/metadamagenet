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
from src.zoo.models import SeResNext50_Unet_Loc
from src.logs import log
from src.augment import FourFlips


class SEResnext50LocValidator(LocalizationValidator):
    def _setup(self):
        super(SEResnext50LocValidator, self)._setup()
        np.random.seed(self._config.model_config.seed + 123)
        random.seed(self._config.model_config.seed + 123)


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
        name='res50_loc',
        empty_model=SeResNext50_Unet_Loc().cuda(),
        version='tuned',
        seed=args.seed
    )
    validation_config = ValidationConfig(
        model_config=model_config,
        dataloader=dataloader,
        test_time_augmentor=FourFlips() if args.test_time_augment else None
    )
    validator: LocalizationValidator = SEResnext50LocValidator(validation_config)
    validator.validate()
    elapsed = timeit.default_timer() - t0
    log(':hourglass: : {:.3f} min'.format(elapsed / 60))
