import abc
import random
from typing import Sequence, Tuple, Dict, Optional, Any, Type

import numpy
import numpy.typing as npt


class AugmentationInterface(abc.ABC):
    """augmentation which can be applied to an image, and it's corresponding mask, or a batch of images"""
    ParamType: Type = Any

    def __init__(self, apply_to: Optional[Sequence[str]] = None):
        """
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        self._apply_to: Optional[Sequence[str]] = apply_to

    @abc.abstractmethod
    def _determine_params(self) -> ParamType:
        """
        determine augmentation parameters.
        parameter types can be different based on augmentation
        :return: augmentation parameters
        """
        pass

    @abc.abstractmethod
    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        """
            :param img: image h*w*c
            :return: (transformed image h'*w'*c')
        """
        pass

    @abc.abstractmethod
    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        """
            :param img: image h*w*3
            :param msk: mask h*w*1
            :return: (transformed image h*w*3, transformed mask h*w*1)
        """
        pass

    def apply_tuple(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, bool]:
        """
        :param img: image h*w*3
        :param msk: mask h*w*1
        :return: (transformed image h*w*3, transformed mask h*w*1, if augmentation is applied)
        """
        params = self._determine_params()
        try:
            img, msk = self._apply_tuple(img, msk, params)
        except Exception:
            print(type(img), type(msk))
            print(img)
            print(msk)
        return img, msk, True

    def apply_batch(self, img_batch: Dict[str, npt.NDArray]) -> Tuple[Dict[str, npt.NDArray], bool]:
        params = self._determine_params()

        keys = self._apply_to if self._apply_to is not None else img_batch.keys()

        for key in keys:
            if type(img_batch[key]) != numpy.ndarray:
                print(type(img_batch[key]))

            img_batch[key] = self._transform(img_batch[key], params)

        return img_batch, True


class Augmentation(AugmentationInterface, abc.ABC):
    """augmentation which can be applied to an image, and it's corresponding mask with a certain change"""

    def __init__(self, probability: float, apply_to: Optional[Sequence[str]]):
        """
        :param probability: 1 - probability of the augmentation being applied
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(apply_to)
        if not 0 <= probability < 1:
            raise TypeError("probability of augmentation should be in [0,1)")

        self._probability: float = probability

    def apply_tuple(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, bool]:
        if random.random() > self._probability:
            return super(Augmentation, self).apply_tuple(img, msk)
        return img, msk, False

    def apply_batch(self, img_batch: Dict[str, npt.NDArray]) -> Tuple[Dict[str, npt.NDArray], bool]:
        if random.random() > self._probability:
            return super(Augmentation, self).apply_batch(img_batch)
        return img_batch, False


class OneOf(AugmentationInterface):
    """applying one of given augmentations but with certain probability"""
    ParamType: Type = Any

    def _determine_params(self) -> ParamType:
        return None

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        pass

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        pass

    def __init__(self, augmentations: Sequence[AugmentationInterface], probability: float):
        """
        :param augmentations: sequence of Augmentation Interface to be applied
        :param probability: 1 - probability of this set of augmentations being applied
        """
        super().__init__(apply_to=None)
        self._augmentations: Sequence[AugmentationInterface] = augmentations
        if not 0 <= probability < 1:
            raise TypeError("probability of augmentation should be in [0,1)")
        self._probability = probability

    def apply_tuple(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, bool]:
        if random.random() > self._probability:
            for augmentation in self._augmentations:
                img, msk, applied = augmentation.apply_tuple(img, msk)
                if applied:
                    break
            return img, msk, True
        return img, msk, False

    def apply_batch(self, img_batch: Dict[str, npt.NDArray]) -> Tuple[Dict[str, npt.NDArray], bool]:
        if random.random() > self._probability:
            for augmentation in self._augmentations:
                img_batch, applied = augmentation.apply_batch(img_batch)
                if applied:
                    break
            return img_batch, True
        return img_batch, False


class Pipeline(AugmentationInterface):
    ParamType: Type = Any

    def __init__(self, augmentations: Sequence[AugmentationInterface]):
        """
        :param augmentations: sequence of Augmentation Interface to be applied in order
        """
        super().__init__(apply_to=None)
        self._augmentations: Sequence[AugmentationInterface] = augmentations

    def _determine_params(self) -> ParamType:
        return None

    def _transform(self, img: npt.NDArray, params: ParamType) -> npt.NDArray:
        pass

    def _apply_tuple(self, img: npt.NDArray, msk: npt.NDArray, params: ParamType) -> Tuple[npt.NDArray, npt.NDArray]:
        pass

    def apply_tuple(self, img: npt.NDArray, msk: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, bool]:
        for augmentation in self._augmentations:
            img, msk, _ = augmentation.apply_tuple(img, msk)
        return img, msk, True

    def apply_batch(self, img_batch: Dict[str, npt.NDArray]) -> Tuple[Dict[str, npt.NDArray], bool]:
        for augmentation in self._augmentations:
            img_batch, _ = augmentation.apply_batch(img_batch)
        return img_batch, True
