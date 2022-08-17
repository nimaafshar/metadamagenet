import dataclasses
import random
from typing import Tuple, Type, Any, Optional, Sequence, Dict

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
    ParamType = Any

    def __init__(self, probability: float, apply_to: Optional[Sequence[str]] = None):
        super().__init__(probability, apply_to)

    def _determine_params(self) -> ParamType:
        return None

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        return img[::-1, ...]

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        return img[::-1, ...], msk[::-1, ...]


class Rotation90Degree(Augmentation):
    """rotates image 90 degrees randomly between 0-3 times """
    ParamType: Type = int

    def __init__(self, probability: float, apply_to: Optional[Sequence[str]] = None):
        super().__init__(probability, apply_to)

    def _determine_params(self) -> ParamType:
        return random.randrange(4)

    def _transform(self, img: npt.NDArray, rot: ParamType) -> npt.NDArray:
        if rot > 0:
            return np.rot90(img, k=rot)

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, rot: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        # rotate 0 or 90 or 180 or 270 degrees
        if rot > 0:
            img = np.rot90(img, k=rot)
            msk = np.rot90(msk, k=rot)
        return img, msk


class Shift(Augmentation):
    """shifts image. moving the shift point to (0,0). replaces empty pixels with reflection"""

    ParamType = Tuple[int, int]

    def __init__(self, probability: float, y_range: Tuple[int, int], x_range: Tuple[int, int],
                 apply_to: Optional[Sequence[str]] = None):
        """
        :param probability: 1 - probability of this augmentation being applied
        :param y_range: shift range in y-axis
        :param x_range: shift range in x-axis
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(probability, apply_to)
        self._y_range: Tuple[int, int] = y_range
        self._x_range: Tuple[int, int] = x_range

    def _determine_params(self) -> ParamType:
        return (
            random.randint(*self._y_range),
            random.randint(*self._x_range)
        )

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        return shift_image(img, params)

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        img = shift_image(img, params)
        msk = shift_image(msk, params)
        return img, msk


class RotateAndScale(Augmentation):
    """ rotate image around a center and scale"""

    @dataclasses.dataclass
    class RotateAndScaleParams:
        center_y: int
        center_x: int
        angle: int
        scale: float

    ParamType = RotateAndScaleParams

    def __init__(self, probability: float,
                 center_y_range: Tuple[int, int],
                 center_x_range: Tuple[int, int],
                 angle_range: Tuple[int, int],
                 scale_range: Tuple[float, float],
                 apply_to: Optional[Sequence[str]] = None
                 ):
        """
        :param probability: 1 - probability of this augmentation being applied
        :param center_y_range: y-range of center of rotation with respect to image center,
            should be less than half of image height
        :param center_x_range: x-range of center of rotation with respect to image center,
            should be less than half of image width
        :param angle_range: rotation angle range in degrees
        :param scale_range: scale range
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(probability, apply_to)
        self._center_y_range: Tuple[int, int] = center_y_range
        self._center_x_range: Tuple[int, int] = center_x_range
        self._angle_range: Tuple[int, int] = angle_range
        self._scale_range: Tuple[float, float] = scale_range

    def _determine_params(self) -> ParamType:
        return RotateAndScale.RotateAndScaleParams(
            center_y=random.randint(*self._center_y_range),
            center_x=random.randint(*self._center_x_range),
            angle=random.randint(*self._angle_range),
            scale=random.uniform(*self._scale_range)
        )

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        rot_pnt = (img.shape[0] // 2 + params.center_y,
                   img.shape[1] // 2 + params.center_x)
        if (params.angle != 0) or (params.scale != 1):
            img = rotate_image(img, params.angle, params.scale, rot_pnt)
        return img

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        rot_pnt = (img.shape[0] // 2 + params.center_y,
                   img.shape[1] // 2 + params.center_x)
        if (params.angle != 0) or (params.scale != 1):
            img = rotate_image(img, params.angle, params.scale, rot_pnt)
            msk = rotate_image(msk, params.angle, params.scale, rot_pnt)
        return img, msk


class Resize(AugmentationInterface):
    ParamType = Any

    def __init__(self, height: int, width: int, apply_to: Optional[Sequence[str]] = None):
        """
        :param height: target image height
        :param width: target image width
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(apply_to)
        self._height: int = height
        self._width: int = width

    def _determine_params(self) -> ParamType:
        return None

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        if img.shape[0] != self._height or img.shape[1] != self._width:
            print(img.dtype)
            img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
        return img

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        if img.shape[0] != self._height or img.shape[1] != self._width:
            img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
        return img, msk


class ShiftRGB(Augmentation):
    """shift RGB channels"""
    ParamType = Tuple[int, int, int]

    def __init__(self, probability: float, r_range: Tuple[int, int], g_range: Tuple[int, int],
                 b_range: Tuple[int, int], apply_to: Optional[Sequence[str]] = None):
        """
        :param probability: 1 - probability of this augmentation being applied
        :param r_range: range of R channel shift
        :param g_range: range of G channel shift
        :param b_range: range of B channel shift
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(probability, apply_to)
        self._r_range: Tuple[int, int] = r_range
        self._g_range: Tuple[int, int] = g_range
        self._b_range: Tuple[int, int] = b_range

    def _determine_params(self) -> ParamType:
        return (random.randint(*self._b_range),
                random.randint(*self._g_range),
                random.randint(*self._r_range))

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        return shift_channels(img, *params)

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        return shift_channels(img, *params), msk


class ShiftHSV(Augmentation):
    """shift HSV channels"""
    ParamType = Tuple[int, int, int]

    def __init__(self, probability: float, h_range: Tuple[int, int], s_range: Tuple[int, int],
                 v_range: Tuple[int, int], apply_to: Optional[Sequence[str]] = None):
        """
        :param probability: 1 - probability of this augmentation being applied
        :param h_range: range of H channel shift
        :param s_range: range of S channel shift
        :param v_range: range of V channel shift
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(probability, apply_to)
        self._h_range: Tuple[int, int] = h_range
        self._s_range: Tuple[int, int] = s_range
        self._v_range: Tuple[int, int] = v_range

    def _determine_params(self) -> ParamType:
        return (random.randint(*self._h_range),
                random.randint(*self._s_range),
                random.randint(*self._v_range))

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        return change_hsv(img, *params)

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        return change_hsv(img, *params), msk


class RandomCrop(AugmentationInterface):
    """
    crop image with a random square.
    does multiple tries, selects the one who is covering more of the white pixels in the mask
    """

    @dataclasses.dataclass
    class RandomCropParams:
        crop_size: int
        try_count: int
        x0: Optional[int] = None
        y0: Optional[int] = None
        batch: bool = False

    ParamType = RandomCropParams

    def __init__(self, default_crop_size: int, size_change_probability: float, crop_size_range: Tuple[int, int],
                 try_range: Tuple[int, int], apply_to: Optional[Sequence[str]] = None,
                 scoring_weights: Optional[Dict[str, float]] = None):
        """
        :param default_crop_size: default crop size
        :param size_change_probability: 1 - probability of crop size being changed randomly
        :param crop_size_range: range of crop size
        :param try_range: range for the number of tries to select the best crop randomly
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(apply_to)
        self._default_crop_size: int = default_crop_size
        if not 0 <= size_change_probability < 1:
            raise TypeError("probability of augmentation should be in [0,1)")
        self._size_change_probability: float = size_change_probability
        self._crop_size_range: Tuple[int, int] = crop_size_range
        self._try_range: Tuple[int, int] = try_range
        self._scoring_weights: Optional[Dict[str, float]] = scoring_weights

    @property
    def _crop_size(self) -> int:
        if random.random() > self._size_change_probability:
            return random.randint(*self._crop_size_range)
        else:
            return self._default_crop_size

    @staticmethod
    def score(msk: npt.NDArray, crop_size: int, x0: int, y0: int) -> int:
        return msk[y0:y0 + crop_size, x0:x0 + crop_size].sum()

    @staticmethod
    def _score_on_batch(msk_batch: Dict[str, npt.NDArray], crop_size: int, x0: int, y0: int,
                        weights: Dict[str, float]) -> int:
        score: int = 0
        for key, weight in weights:
            msk = msk_batch.get(key)
            if msk is not None:
                score += RandomCrop.score(msk, crop_size, x0, y0)
        return score

    def _determine_params(self) -> ParamType:
        return RandomCrop.RandomCropParams(
            crop_size=self._crop_size,
            try_count=random.randint(*self._try_range)
        )

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        return img[params.y0: params.y0 + params.crop_size, params.x0: params.x0 + params.crop_size, ...]

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        params.batch = False

        params = self._best_crop(msk, params)

        img = self._transform(img, params)
        msk = self._transform(msk, params)
        return img, msk

    def _best_crop_batch(self, img_batch: Dict[str, npt.NDArray], params: ParamType) -> ParamType:
        img = next(iter(img_batch.values()))
        best_x0 = random.randint(0, img.shape[1] - params.crop_size)
        best_y0 = random.randint(0, img.shape[0] - params.crop_size)
        best_score = -1

        weights = self._scoring_weights if self._scoring_weights is not None else {key: 1 for key in img_batch}

        for _ in range(params.try_count):
            x0 = random.randint(0, img.shape[1] - params.crop_size)
            y0 = random.randint(0, img.shape[0] - params.crop_size)
            score = RandomCrop._score_on_batch(img_batch, params.crop_size, x0, y0, weights)
            if score > best_score:
                best_score = score
                best_x0 = x0
                best_y0 = y0
        params.x0 = best_x0
        params.y0 = best_y0
        return params

    @staticmethod
    def _best_crop(msk: npt.NDArray, params: ParamType) -> ParamType:
        best_x0 = random.randint(0, msk.shape[1] - params.crop_size)
        best_y0 = random.randint(0, msk.shape[0] - params.crop_size)
        best_score = -1

        for _ in range(params.try_count):
            x0 = random.randint(0, msk.shape[1] - params.crop_size)
            y0 = random.randint(0, msk.shape[0] - params.crop_size)
            score = RandomCrop.score(msk, params.crop_size, x0, y0)
            if score > best_score:
                best_score = score
                best_x0 = x0
                best_y0 = y0
        params.x0 = best_x0
        params.y0 = best_y0
        return params

    def apply_batch(self, img_batch: Dict[str, npt.NDArray]) -> Tuple[Dict[str, npt.NDArray], bool]:
        params = self._determine_params()
        params.batch = True

        params = self._best_crop_batch(img_batch, params)

        keys = self._apply_to if self._apply_to is not None else img_batch.keys()

        for key in keys:
            img_batch[key] = self._transform(img_batch[key], params)

        return img_batch, True


class Clahe(Augmentation):
    ParamType = Any

    def __init__(self, probability: float, apply_to: Optional[Sequence[str]] = None):
        super().__init__(probability, apply_to)

    def _determine_params(self) -> ParamType:
        return None

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        return clahe(img)

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        return clahe(img), msk


class GaussianNoise(Augmentation):
    ParamType = Any

    def __init__(self, probability: float, apply_to: Optional[Sequence[str]] = None):
        super().__init__(probability, apply_to)

    def _determine_params(self) -> ParamType:
        return None

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        return gauss_noise(img)

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        return gauss_noise(img), msk


class Blur(Augmentation):
    ParamType = Any


    def __init__(self, probability: float, apply_to: Optional[Sequence[str]] = None):
        super().__init__(probability, apply_to)

    def _determine_params(self) -> ParamType:
        return None

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        return blur(img)

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        return blur(img), msk


class Saturation(Augmentation):
    ParamType = float

    def __init__(self, probability: float, alpha_range: Tuple[float, float], apply_to: Optional[Sequence[str]] = None):
        """
        :param probability: 1 - probability of this augmentation being applied
        :param alpha_range: range of alpha for saturation
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(probability, apply_to)
        self._alpha_range: Tuple[float, float] = alpha_range

    def _determine_params(self) -> ParamType:
        return random.uniform(*self._alpha_range)

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        return saturation(img, params)

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        return saturation(img, params), msk


class Brightness(Augmentation):
    ParamType = float

    def __init__(self, probability: float, alpha_range: Tuple[float, float], apply_to: Optional[Sequence[str]] = None):
        """
        :param probability: 1 - probability of this augmentation being applied
        :param alpha_range: range of alpha for brightness
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(probability, apply_to)
        self._alpha_range: Tuple[float, float] = alpha_range

    def _determine_params(self) -> ParamType:
        return random.uniform(*self._alpha_range)

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        return brightness(img, params)

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        return brightness(img, params), msk


class Contrast(Augmentation):
    ParamType = float

    def __init__(self, probability: float, alpha_range: Tuple[float, float], apply_to: Optional[Sequence[str]] = None):
        """
        :param probability: 1 - probability of this augmentation being applied
        :param alpha_range: range of alpha for contrast
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(probability, apply_to)
        self._alpha_range: Tuple[float, float] = alpha_range

    def _determine_params(self) -> ParamType:
        return random.uniform(*self._alpha_range)

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        return contrast(img, params)

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        return contrast(img, params), msk


class ElasticTransformation(Augmentation):
    ParamType = Any

    def __init__(self,
                 probability: float,
                 alpha: Tuple[float, float] = (0.25, 1.2),
                 sigma: float = 0.2,
                 apply_to: Optional[Sequence[str]] = None):
        super().__init__(probability, apply_to)
        self._transformation: iaa.Augmenter = iaa.ElasticTransformation(alpha, sigma).to_deterministic()

    def _determine_params(self) -> ParamType:
        return None

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        self._transformation.augment_image(img)

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        return self._transformation.augment_image(img), msk
