import abc
from typing import List

import numpy as np
import numpy.typing as npt

from src.util.utils import dice


class MetricCalculator(abc.ABC):
    @abc.abstractmethod
    def update(self, msk: npt.NDArray, msk_pred: npt.NDArray) -> None:
        """
        :param msk: mask predicted by model
        :param msk_pred: correct mask
        :return: None
        """
        pass

    @abc.abstractmethod
    def aggregate(self) -> float:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass


class DiceCalculator(MetricCalculator):

    def __init__(self, threshold=0.3):
        self._dices: List[float] = []
        self._threshold = threshold

    def update(self, msk: npt.NDArray, msk_pred: npt.NDArray) -> None:
        self._dices.append(dice(msk, msk_pred > self._threshold))

    def aggregate(self) -> float:
        return float(np.mean(self._dices))

    def reset(self) -> None:
        self._dices.clear()


class F1ScoreCalculator(MetricCalculator):
    def __init__(self):
        self._tp: float = 0.0
        self._fn: float = 0.0
        self._fp: float = 0.0

    def update(self, msk: npt.NDArray, msk_pred: npt.NDArray) -> None:
        self._tp += np.logical_and(msk > 0, msk_pred > 0).sum()
        self._fn += np.logical_and(msk < 1, msk_pred > 0).sum()
        self._fp += np.logical_and(msk > 0, msk_pred < 1).sum()

    def aggregate(self) -> float:
        return 2 * self._tp / (2 * self._tp + self._fp + self._fn)

    def reset(self):
        self._tp: float = 0.0
        self._fn: float = 0.0
        self._fp: float = 0.0
