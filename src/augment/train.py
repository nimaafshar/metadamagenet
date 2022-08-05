import random
from typing import Tuple

import numpy as np
from numpy import typing as npt
import cv2
from imgaug import augmenters as iaa

from .augmentation import Augmentation, AugmentationInterface
from .utils import (
    shift_image,
    rotate_image,
    shift_channels,
    change_hsv,
    clahe,
    gauss_noise,
    blur,
    saturation,
    brightness,
    contrast
)


class TopDownFlip(Augmentation):
    """image top-down flip"""

    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        return img[::-1, ...], msk[::-1, ...]


class Rotation90Degree(Augmentation):
    """rotates image 90 degrees randomly between 0-3 times """

    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        # rotate 0 or 90 or 180 or 270 degrees
        rot = random.randrange(4)
        if rot > 0:
            img = np.rot90(img, k=rot)
            msk = np.rot90(msk, k=rot)
        return img, msk


class Shift(Augmentation):
    """shifts image. moving the shift point to (0,0). replaces empty pixels with reflection"""

    def __init__(self, probability: float, y_range: Tuple[int, int], x_range: Tuple[int, int]):
        """
        :param probability: 1 - probability of this augmentation being applied
        :param y_range: shift range in y-axis
        :param x_range: shift range in x-axis
        """
        super().__init__(probability)
        self._y_range: Tuple[int, int] = y_range
        self._x_range: Tuple[int, int] = x_range

    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        shift_pnt = (random.randint(*self._y_range),
                     random.randint(*self._x_range))
        img = shift_image(img, shift_pnt)
        msk = shift_image(msk, shift_pnt)
        return img, msk


class RotateAndScale(Augmentation):
    """ rotate image around a center and scale"""

    def __init__(self, probability: float,
                 center_y_range: Tuple[int, int],
                 center_x_range: Tuple[int, int],
                 angle_range: Tuple[int, int],
                 scale_range: Tuple[float, float]):
        """
        :param probability: 1 - probability of this augmentation being applied
        :param center_y_range: y-range of center of rotation with respect to image center,
            should be less than half of image height
        :param center_x_range: x-range of center of rotation with respect to image center,
            should be less than half of image width
        :param angle_range: rotation angle range in degrees
        :param scale_range: scale range
        """
        super().__init__(probability)
        self._center_y_range: Tuple[int, int] = center_y_range
        self._center_x_range: Tuple[int, int] = center_x_range
        self._angle_range: Tuple[int, int] = angle_range
        self._scale_range: Tuple[float, float] = scale_range

    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        rot_pnt = (img.shape[0] // 2 + random.randint(*self._center_y_range),
                   img.shape[1] // 2 + random.randint(*self._center_x_range))
        scale = random.uniform(*self._scale_range)
        angle = random.randint(*self._angle_range)
        if (angle != 0) or (scale != 1):
            img = rotate_image(img, angle, scale, rot_pnt)
            msk = rotate_image(msk, angle, scale, rot_pnt)
        return img, msk


class Resize(AugmentationInterface):
    def __init__(self, height: int, width: int):
        """
        :param height: target image height
        :param width: target image width
        """
        self._height: int = height
        self._width: int = width

    def apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, bool]:
        if img.shape[0] != self._height or img.shape[1] != self._width:
            img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
        return img, msk, True


class ShiftRGB(Augmentation):
    """shift RGB channels"""

    def __init__(self, probability: float, r_range: Tuple[int, int], g_range: Tuple[int, int],
                 b_range: Tuple[int, int]):
        """
        :param probability: 1 - probability of this augmentation being applied
        :param r_range: range of R channel shift
        :param g_range: range of G channel shift
        :param b_range: range of B channel shift
        """
        super().__init__(probability)
        self._r_range: Tuple[int, int] = r_range
        self._g_range: Tuple[int, int] = g_range
        self._b_range: Tuple[int, int] = b_range

    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        return shift_channels(img,
                              random.randint(*self._b_range),
                              random.randint(*self._g_range),
                              random.randint(*self._r_range)), msk


class ShiftHSV(Augmentation):
    """shift HSV channels"""

    def __init__(self, probability: float, h_range: Tuple[int, int], s_range: Tuple[int, int],
                 v_range: Tuple[int, int]):
        """
        :param probability: 1 - probability of this augmentation being applied
        :param h_range: range of H channel shift
        :param s_range: range of S channel shift
        :param v_range: range of V channel shift
        """
        super().__init__(probability)
        self._h_range: Tuple[int, int] = h_range
        self._s_range: Tuple[int, int] = s_range
        self._v_range: Tuple[int, int] = v_range

    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        return change_hsv(img,
                          random.randint(*self._h_range),
                          random.randint(*self._s_range),
                          random.randint(*self._v_range)), msk


class RandomCrop(AugmentationInterface):
    """
    crop image with a random square.
    does multiple tries, selects the one who is covering more of the white pixels in the mask
    """

    def __init__(self,
                 default_crop_size: int,
                 size_change_probability: float,
                 crop_size_range: Tuple[int, int],
                 try_range: Tuple[int, int]):
        """
        :param default_crop_size: default crop size
        :param size_change_probability: 1 - probability of crop size being changed randomly
        :param crop_size_range: range of crop size
        :param try_range: range for the number of tries to select the best crop randomly
        """
        self._default_crop_size: int = default_crop_size
        if not 0 <= size_change_probability < 1:
            raise TypeError("probability of augmentation should be in [0,1)")
        self._size_change_probability: float = size_change_probability
        self._crop_size_range: Tuple[int, int] = crop_size_range
        self._try_range: Tuple[int, int] = try_range

    @property
    def _crop_size(self):
        if random.random() > self._size_change_probability:
            return random.randint(*self._crop_size_range)
        else:
            return self._default_crop_size

    @staticmethod
    def score(msk: npt.NDArray, crop_size: int, x0: int, y0: int) -> int:
        return msk[y0:y0 + crop_size, x0:x0 + crop_size].sum()

    def apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, bool]:
        crop_size = self._crop_size
        best_x0 = random.randint(0, img.shape[1] - crop_size)
        best_y0 = random.randint(0, img.shape[0] - crop_size)
        best_score = -1
        try_count = random.randint(*self._try_range)
        for _ in range(try_count):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            score = RandomCrop.score(msk, crop_size, x0, y0)
            if score > best_score:
                best_score = score
                best_x0 = x0
                best_y0 = y0
        x0 = best_x0
        y0 = best_y0
        img = img[y0:y0 + crop_size, x0:x0 + crop_size, :]
        msk = msk[y0:y0 + crop_size, x0:x0 + crop_size]
        return img, msk, True


class Clahe(Augmentation):
    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        return clahe(img), msk


class GaussianNoise(Augmentation):
    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        return gauss_noise(img), msk


class Blur(Augmentation):
    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        return blur(img), msk


class Saturation(Augmentation):
    def __init__(self, probability: float, alpha_range: Tuple[float, float]):
        super().__init__(probability)
        self._alpha_range: Tuple[float, float] = alpha_range

    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        return saturation(img, random.uniform(*self._alpha_range)), msk


class Brightness(Augmentation):
    def __init__(self, probability: float, alpha_range: Tuple[float, float]):
        super().__init__(probability)
        self._alpha_range: Tuple[float, float] = alpha_range

    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        return brightness(img, random.uniform(*self._alpha_range)), msk


class Contrast(Augmentation):
    def __init__(self, probability: float, alpha_range: Tuple[float, float]):
        super().__init__(probability)
        self._alpha_range: Tuple[float, float] = alpha_range

    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        return contrast(img, random.uniform(*self._alpha_range)), msk


class ElasticTransformation(Augmentation):
    def __init__(self,
                 probability: float,
                 alpha: Tuple[float, float] = (0.25, 1.2),
                 sigma: float = 0.2):
        super().__init__(probability)
        self._transformation: iaa.Augmenter = iaa.ElasticTransformation(alpha, sigma).to_deterministic()

    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        return self._transformation.augment_image(img), msk
