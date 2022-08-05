import random
from typing import Tuple

import numpy as np
import numpy.typing as npt
import cv2
import torch
from torch.utils.data import Dataset as TorchDataset

from src.file_structure import Dataset as ImageDataset
from src.file_structure import ImageData, DataTime
from src.augment.augmentation import OneOf
from src.util.utils import normalize_colors
from src.augment.train import (
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


class TrainData(TorchDataset):
    def __init__(self, image_dataset: ImageDataset, model_input_shape: Tuple[int, int]):
        super().__init__()
        self._image_dataset: ImageDataset = image_dataset
        self._model_input_shape: Tuple[int, int] = model_input_shape

    def __len__(self) -> int:
        return len(self._image_dataset)

    def __getitem__(self, identifier: str) -> Tuple[npt.NDArray, npt.NDArray, ImageData]:
        image_data: ImageData = self._image_dataset[identifier]

        img: npt.NDArray
        msk: npt.NDArray

        # read pre_disaster image and msk
        img: npt.NDArray = cv2.imread(str(image_data.image(DataTime.PRE)), cv2.IMREAD_COLOR)
        msk: npt.NDArray = cv2.imread(str(image_data.mask(DataTime.PRE)), cv2.IMREAD_UNCHANGED)

        if random.random() > 0.985:
            # replace with post_disaster version
            img: npt.NDArray = cv2.imread(str(image_data.image(DataTime.POST)), cv2.IMREAD_COLOR)
            msk: npt.NDArray = (cv2.imread(str(image_data.mask(DataTime.POST)), cv2.IMREAD_UNCHANGED) > 0) * 255
            # FIXME: FIXED. tell the owner: previously pre-disaster masks were used for post-disaster images too

        TopDownFlip(probability=0.5)

        Rotation90Degree(probability=0.05)

        Shift(probability=0.8,
              y_range=(-320, 320),
              x_range=(-320, 320))

        RotateAndScale(
            probability=0.2,
            center_y_range=(-320, 320),
            center_x_range=(-320, 320),
            angle_range=(-10, 10),
            scale_range=(0.9, 1.1)
        )

        RandomCrop(
            default_crop_size=self._model_input_shape[0],
            size_change_probability=0.3,
            crop_size_range=(int(self._model_input_shape[0] / 1.2), int(self._model_input_shape[0] / 0.8)),
            try_range=(1, 5)
        )

        Resize(*self._model_input_shape)

        OneOf((
            ShiftRGB(probability=0.97,
                     r_range=(-5, 5),
                     g_range=(-5, 5),
                     b_range=(-5, 5)),

            ShiftHSV(probability=0.97,
                     h_range=(-5, 5),
                     s_range=(-5, 5),
                     v_range=(-5, 5))), probability=0)

        OneOf((
            OneOf((
                Clahe(0.97),
                GaussianNoise(0.97),
                Blur(0.98)),
                probability=0.93),
            OneOf((
                Saturation(0.97, (0.9, 1.1)),
                Brightness(0.97, (0.9, 1.1)),
                Contrast(0.97, (0.9, 1.1))),
                probability=0.93)), probability=0)

        ElasticTransformation(0.97)

        msk = msk[..., np.newaxis]

        msk = (msk > 127) * 1

        img = normalize_colors(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        return img, msk, image_data
