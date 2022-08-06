import random
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import cv2
import torch
from torch.utils.data import Dataset as TorchDataset

from src.file_structure import Dataset as ImageDataset
from src.file_structure import ImageData, DataTime
from src.augment import Pipeline
from src.util.utils import normalize_colors


class Dataset(TorchDataset):
    def __init__(self,
                 image_dataset: ImageDataset,
                 augmentations: Union[Pipeline, None] = None,
                 post_version_prob: float = 0.985):
        """
        Train Dataset
        :param image_dataset: dataset of images
        :param augmentations: pipeline of augmentations
        :param post_version_prob: 1 - probability of replacing the image with its post version
        """
        super().__init__()
        self._image_dataset: ImageDataset = image_dataset
        self._augments: Union[Pipeline, None] = augmentations
        if not 0 <= post_version_prob <= 1:
            raise TypeError("post_version_prob should be in [0,1]")
        self._post_version_prob: float = post_version_prob

    def __len__(self) -> int:
        return len(self._image_dataset)

    def __getitem__(self, identifier: int) -> Tuple[npt.NDArray, npt.NDArray]:
        image_data: ImageData = self._image_dataset[identifier]

        img: npt.NDArray
        msk: npt.NDArray

        # read pre_disaster image and msk
        img: npt.NDArray = cv2.imread(str(image_data.image(DataTime.PRE)), cv2.IMREAD_COLOR)
        msk: npt.NDArray = cv2.imread(str(image_data.mask(DataTime.PRE)), cv2.IMREAD_UNCHANGED)

        if random.random() > self._post_version_prob:
            # replace with post_disaster version
            img: npt.NDArray = cv2.imread(str(image_data.image(DataTime.POST)), cv2.IMREAD_COLOR)
            msk: npt.NDArray = (cv2.imread(str(image_data.mask(DataTime.POST)), cv2.IMREAD_UNCHANGED) > 0) * 255
            # FIXME: FIXED. tell the owner: previously pre-disaster masks were used for post-disaster images too

        if self._augments is not None:
            img, msk, _ = self._augments.apply(img, msk)

        msk = msk[..., np.newaxis]
        msk = (msk > 127) * 1

        img = normalize_colors(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        return img, msk
