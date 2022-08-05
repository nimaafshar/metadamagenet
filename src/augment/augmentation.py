import abc
import random
from typing import Sequence

import numpy.typing as npt


class AugmentationInterface(abc.ABC):
    """augmentation which can be applied to an image, and it's corresponding mask"""

    @abc.abstractmethod
    def apply(self, img: npt.NDArray, msk: npt.NDArray) -> (npt.NDArray, npt.NDArray, bool):
        """
        :param img: image h*w*3
        :param msk: mask h*w*1
        :return: (transformed image h*w*3, transformed mask h*w*1, if augmentation is applied)
        """
        pass


class Augmentation(AugmentationInterface, abc.ABC):
    """augmentation which can be applied to an image, and it's corresponding mask with a certain change"""

    def __init__(self, probability: float):
        """
        :param probability: 1 - probability of the augmentation being applied
        """
        if not 0 <= probability < 1:
            raise TypeError("probability of augmentation should be in [0,1)")

        self._probability: float = probability

    @abc.abstractmethod
    def _apply(self, img: npt.NDArray, msk: npt.NDArray) -> (npt.NDArray, npt.NDArray):
        """
        :param img: image h*w*3
        :param msk: mask h*w*1
        :return: (transformed image h*w*3, transformed mask h*w*1)
        """
        pass

    def apply(self, img: npt.NDArray, msk: npt.NDArray) -> (npt.NDArray, npt.NDArray, bool):
        if random.random() > self._probability:
            img, msk = self._apply(img, msk)
            return img, msk, True
        return img, msk, False


class OneOf(AugmentationInterface):
    """applying one of given augmentations but with certain probability"""

    def __init__(self, augmentations: Sequence[AugmentationInterface], probability: float):
        """
        :param augmentations: sequence of Augmentation Interface to be applied
        :param probability: 1 - probability of this set of augmentations being applied
        """
        self._augmentations: Sequence[AugmentationInterface] = augmentations
        if not 0 <= probability < 1:
            raise TypeError("probability of augmentation should be in [0,1)")
        self._probability = probability

    def apply(self, img: npt.NDArray, msk: npt.NDArray) -> (npt.NDArray, npt.NDArray, bool):
        if random.random() > self._probability:
            for augmentation in self._augmentations:
                img, msk, applied = augmentation.apply(img, msk)
                if applied:
                    break
            return img, msk, True
        return img, msk, False
