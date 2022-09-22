from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import cv2
import torch
from torch.utils.data import Dataset

from ..utils import normalize_colors
from .data_time import DataTime
from .image_data import ImageData


# class ClassificationDataset(Dataset):
#     def __init__(self,
#                  image_dataset: List[ImageData]):
#         """
#         Train Dataset
#         :param image_dataset: list of image datas
#         """
#         self._image_dataset: List[ImageData] = image_dataset
#
#     def __len__(self):
#         return len(self._image_dataset)
#
#     def __getitem__(self, identifier: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
#         """
#         :param identifier: # of image data
#         :return: (inputs, targets)
#             inputs: FloatTensor of shape (6,1024,1024)
#             targets: LongTensor of shape (5,1024,1024)
#         """
#         image_data: ImageData = self._image_dataset[identifier]
#
#         # read pre_disaster image and msk
#         img_pre: npt.NDArray = cv2.imread(str(image_data.image(DataTime.PRE)), cv2.IMREAD_COLOR).astype("uint8")
#         # msk_pre: npt.NDArray = cv2.imread(str(image_data.mask(DataTime.PRE)), cv2.IMREAD_UNCHANGED)
#
#         # read post_disaster image and mask
#         img_post: npt.NDArray = cv2.imread(str(image_data.image(DataTime.POST)), cv2.IMREAD_COLOR).astype("uint8")
#         msk_post: npt.NDArray = cv2.imread(str(image_data.mask(DataTime.POST)), cv2.IMREAD_UNCHANGED).astype("uint8")
#         # values 0-4
#
#         # convert label_mask into four black and white masks
#         msk0: npt.NDArray = np.zeros_like(msk_post)
#         msk1: npt.NDArray = np.zeros_like(msk_post)
#         msk2: npt.NDArray = np.zeros_like(msk_post)
#         msk3: npt.NDArray = np.zeros_like(msk_post)
#         msk4: npt.NDArray = np.zeros_like(msk_post)
#
#         msk0[msk_post == 0] = 1
#         msk1[msk_post == 1] = 1
#         msk2[msk_post == 2] = 1
#         msk3[msk_post == 3] = 1
#         msk4[msk_post == 4] = 1
#
#         if self._augments is not None:
#             img_batch = {
#                 'img_pre': img_pre,  # pre-disaster image
#                 'img_post': img_post,  # post-disaster image
#                 'msk0': msk0,  # mask for pre-disaster building localization
#                 'msk1': msk1,  # damage level 1
#                 'msk2': msk2,  # damage level 2
#                 'msk3': msk3,  # damage level 3
#                 'msk4': msk4  # damage level 4
#             }
#             img_batch, _ = self._augments.apply_batch(img_batch)
#             img_pre = img_batch['img_pre']
#             img_post = img_batch['img_post']
#             msk0 = img_batch['msk0']
#             msk1 = img_batch['msk1']
#             msk2 = img_batch['msk2']
#             msk3 = img_batch['msk3']
#             msk4 = img_batch['msk4']
#
#         msk0 = msk0[..., np.newaxis]
#         msk1 = msk1[..., np.newaxis]
#         msk2 = msk2[..., np.newaxis]
#         msk3 = msk3[..., np.newaxis]
#         msk4 = msk4[..., np.newaxis]
#
#         msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
#
#         msk = msk * 1
#
#         img = np.concatenate([img_pre, img_post], axis=2)
#         img = normalize_colors(img)
#
#         img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
#         msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).long()
#
#         return img, msk
