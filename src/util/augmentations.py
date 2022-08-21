from typing import List

import numpy as np
import numpy.typing as npt


def test_time_augment(img: npt.NDArray) -> npt.NDArray:
    """
    :param img: input which is a single image
    :return: a batch of 4 images stacked together
    """
    return np.asarray(
        (img,  # original
         img[::-1, ...],  # flip up-down
         img[:, ::-1, ...],  # flip left-right
         img[::-1, ::-1, ...]  # flip along both x and y-axis (180 rotation)
         ), dtype='float') \
        .transpose((0, 3, 1, 2))


def revert_augmentation(img_batch: npt.NDArray) -> List[npt.NDArray]:
    """
    :param img_batch: predictions input which is a batch of 4 images
    :return: mean of the batch with augmentations reverted
    """
    return [img_batch[0, ...],  # original
            img_batch[1, :, ::-1, :],  # flip left-right
            img_batch[2, :, :, ::-1],  # flip from RGB to BRG
            img_batch[3, :, ::-1, ::-1]]  # left-right and RGB to BRG flip
