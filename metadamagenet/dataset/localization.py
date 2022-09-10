from typing import Optional, Tuple
import random

import torch
import numpy.typing as npt
import numpy as np
import cv2

from ..utils import normalize_colors
from ..augment import Pipeline
from .data_time import DataTime
from .image_dataset import ImageDataset
from .image_data import ImageData
from .base import Dataset


class LocalizationDataset(Dataset):
    def __init__(self,
                 image_dataset: ImageDataset,
                 augmentations: Optional[Pipeline] = None,
                 post_version_prob: float = 0.985):
        """
        Train Dataset
        :param image_dataset: dataset of images
        :param augmentations: pipeline of augmentations
        :param post_version_prob: 1 - probability of replacing the image with its post version
        """
        super(LocalizationDataset, self).__init__(image_dataset, augmentations)
        if not 0 <= post_version_prob <= 1:
            raise TypeError("post_version_prob should be in [0,1]")
        self._post_version_prob: float = post_version_prob

    def __len__(self) -> int:
        return len(self._image_dataset)

    def __getitem__(self, identifier: int) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        image_data: ImageData = self._image_dataset[identifier]

        img: npt.NDArray
        msk: npt.NDArray

        # read pre_disaster image and msk
        img: npt.NDArray = cv2.imread(str(image_data.image(DataTime.PRE)), cv2.IMREAD_COLOR)
        msk: npt.NDArray = cv2.imread(str(image_data.mask(DataTime.PRE)), cv2.IMREAD_UNCHANGED)

        if random.random() > self._post_version_prob:
            # replace with post_disaster version
            img: npt.NDArray = cv2.imread(str(image_data.image(DataTime.POST)), cv2.IMREAD_COLOR)
            msk: npt.NDArray = (cv2.imread(str(image_data.mask(DataTime.POST)), cv2.IMREAD_UNCHANGED) > 0) \
                .astype(np.uint8)
            # tell the owner: previously pre-disaster masks were used for post-disaster images too

        if self._augments is not None:
            img, msk, _ = self._augments.apply_tuple(img, msk)

        msk = msk[..., np.newaxis]

        img = normalize_colors(img)

        img: torch.FloatTensor = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        msk: torch.BoolTensor = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).bool()

        return img, msk
