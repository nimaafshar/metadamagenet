from typing import Optional, List, Tuple

import numpy as np
import numpy.typing as npt
import cv2
import torch
from torch.utils.data import Dataset
from skimage.morphology import square, dilation

from ..augment import Pipeline
from ..utils import normalize_colors
from .data_time import DataTime
from .image_data import ImageData


class ClassificationDataset(Dataset):
    def __init__(self,
                 image_dataset: List[ImageData],
                 do_dilation: bool = False,
                 augmentations: Optional[Pipeline] = None):
        """
        Train Dataset
        :param do_dilation: do morphological dilation to image or not
        :param image_dataset: list of image datas
        :param augmentations: pipeline of augmentations
        """
        self._image_dataset: List[ImageData] = image_dataset
        self._do_dilation: bool = do_dilation
        self._augments: Optional[Pipeline] = augmentations

    def __getitem__(self, identifier: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        :param identifier: # of image data
        :return: (inputs, targets)
            inputs: FloatTensor of shape (6,1024,1024)
            targets: LongTensor of shape (5,1024,1024)
        """
        image_data: ImageData = self._image_dataset[identifier]

        # read pre_disaster image and msk
        img_pre: npt.NDArray = cv2.imread(str(image_data.image(DataTime.PRE)), cv2.IMREAD_COLOR)
        # msk_pre: npt.NDArray = cv2.imread(str(image_data.mask(DataTime.PRE)), cv2.IMREAD_UNCHANGED)

        # read post_disaster image and mask
        img_post: npt.NDArray = cv2.imread(str(image_data.image(DataTime.POST)), cv2.IMREAD_COLOR)
        msk_post: npt.NDArray = cv2.imread(str(image_data.mask(DataTime.POST)), cv2.IMREAD_UNCHANGED)  # * 255
        # values 0-4

        # convert label_mask into four black and white masks
        msk0: npt.NDArray = np.zeros_like(msk_post)
        msk1: npt.NDArray = np.zeros_like(msk_post)
        msk2: npt.NDArray = np.zeros_like(msk_post)
        msk3: npt.NDArray = np.zeros_like(msk_post)
        msk4: npt.NDArray = np.zeros_like(msk_post)

        msk0[msk_post == 0] = 1
        msk1[msk_post == 1] = 1
        msk2[msk_post == 2] = 1
        msk3[msk_post == 3] = 1
        msk4[msk_post == 4] = 1

        if self._augments is not None:
            img_batch = {
                'img_pre': img_pre,  # pre-disaster image
                'img_post': img_post,  # post-disaster image
                'msk0': msk0,  # mask for pre-disaster building localization
                'msk1': msk1,  # damage level 1
                'msk2': msk2,  # damage level 2
                'msk3': msk3,  # damage level 3
                'msk4': msk4  # damage level 4
            }
            img_batch, _ = self._augments.apply_batch(img_batch)
            img_pre = img_batch['img_pre']
            img_post = img_batch['img_post']
            msk0 = img_batch['msk0']
            msk1 = img_batch['msk1']
            msk2 = img_batch['msk2']
            msk3 = img_batch['msk3']
            msk4 = img_batch['msk4']

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)

        if self._do_dilation:
            msk[..., 0] = True
            msk[..., 1] = dilation(msk[..., 1], square(5))
            msk[..., 2] = dilation(msk[..., 2], square(5))
            msk[..., 3] = dilation(msk[..., 3], square(5))
            msk[..., 4] = dilation(msk[..., 4], square(5))
            msk[..., 1][msk[..., 2:].max(axis=2)] = False
            msk[..., 3][msk[..., 2]] = False
            msk[..., 4][msk[..., 2]] = False
            msk[..., 4][msk[..., 3]] = False
            msk[..., 0][msk[..., 1:].max(axis=2)] = False

        msk = msk * 1

        img = np.concatenate([img_pre, img_post], axis=2)
        img = normalize_colors(img)

        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).long()

        return img, msk
