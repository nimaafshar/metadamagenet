from typing import Optional, Tuple, List
import random

import torch
from torch.utils.data import Dataset
import numpy.typing as npt
import numpy as np
import cv2

from ..utils import normalize_colors
from ..augment import Pipeline
from .data_time import DataTime
from .image_data import ImageData


class LocalizationDataset(Dataset):
    def __init__(self,
                 image_dataset: List[ImageData],
                 augmentations: Optional[Pipeline] = None,
                 post_version_prob: float = 0.985,
                 use_post_disaster_images: bool = False):
        """
        Train Dataset
        :param image_dataset: dataset of images
        :param augmentations: pipeline of augmentations
        :param post_version_prob: 1 - probability of replacing the image with its post version
        """
        self._image_dataset: List[ImageData] = image_dataset
        self._augments: Optional[Pipeline] = augmentations
        if not 0 <= post_version_prob <= 1:
            raise TypeError("post_version_prob should be in [0,1]")
        self._post_version_prob: float = post_version_prob
        self._use_post_disaster_images: bool = use_post_disaster_images

    def __len__(self) -> int:
        if self._use_post_disaster_images:
            return 2 * len(self._image_dataset)
        return len(self._image_dataset)

    def __getitem__(self, identifier: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        :param identifier: # of image data
        :return: (inputs, targets)
            inputs: FloatTensor of shape (3,1024,1024)
            targets: LongTensor of shape (1,1024,1024)
        """
        image_data: ImageData
        data_time: DataTime
        if self._use_post_disaster_images:
            image_data = self._image_dataset[identifier // 2]
            data_time = DataTime.PRE if identifier % 2 == 0 else DataTime.POST
        else:
            image_data = self._image_dataset[identifier]
            data_time = DataTime.PRE

        img: npt.NDArray
        msk: npt.NDArray

        # read image and msk
        img: npt.NDArray = cv2.imread(str(image_data.image(data_time)), cv2.IMREAD_COLOR).astype("uint8")
        msk: npt.NDArray = (cv2.imread(str(image_data.mask(data_time)), cv2.IMREAD_UNCHANGED) > 0).astype("uint8")

        if not self._use_post_disaster_images and random.random() > self._post_version_prob:
            # replace with post_disaster version
            img: npt.NDArray = cv2.imread(str(image_data.image(DataTime.POST)), cv2.IMREAD_COLOR).astype("uint8")
            msk: npt.NDArray = (cv2.imread(str(image_data.mask(DataTime.POST)), cv2.IMREAD_UNCHANGED) > 0) \
                .astype("uint8")
            # tell the owner: previously pre-disaster masks were used for post-disaster images too

        if self._augments is not None:
            img, msk, _ = self._augments.apply_tuple(img, msk)

        msk = msk[..., np.newaxis]

        img = normalize_colors(img)

        img: torch.FloatTensor = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        msk: torch.IntTensor = torch.from_numpy(msk.transpose((2, 0, 1)).copy())

        return img, msk
