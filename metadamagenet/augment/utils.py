from typing import Tuple

import numpy as np
import numpy.typing as npt
import cv2



def change_hsv(img: npt.NDArray, h_shift: int, s_shift: int, v_shift: int) -> npt.NDArray:
    """
    :param img: input image
    :param h_shift:
    :param s_shift:
    :param v_shift:
    :return: image which is shifted in hsv
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(int)
    hsv[:, :, 0] += h_shift
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 255)
    hsv[:, :, 1] += s_shift
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] += v_shift
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype('uint8')
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def _blend(img1: npt.NDArray, img2: npt.NDArray, alpha: float) -> npt.NDArray:
    """
    belnd two images
    :param img1: image 1
    :param img2: image 2
    :param alpha: share of image one in blend, between 0 and 1
    :return: blending result
    """
    return np.clip(img1 * alpha + (1 - alpha) * img2, 0, 255).astype('uint8')


def _grayscale(img: npt.NDArray) -> npt.NDArray:
    """
    :param img: input image
    :return: grayscale image
    """
    _alpha = np.asarray([0.114, 0.587, 0.299]).reshape((1, 1, 3))
    return np.sum(_alpha * img, axis=2, keepdims=True)


def invert(img: npt.NDArray) -> npt.NDArray:
    """
    :param img: input image
    :return: inverted image
    """
    return 255 - img


def channel_shuffle(img):
    ch_arr = [0, 1, 2]
    np.random.shuffle(ch_arr)
    img = img[..., ch_arr]
    return img
