import abc

import torch


class FloatMetric:
    """
    Compute a float metric in multiple steps
    """

    def __init__(self):
        self._val: float = 0
        self._avg: float = 0
        self._sum: float = 0
        self._count: float = 0

    def reset(self) -> None:
        """
        reset metric for another round of calculation
        """
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    @abc.abstractmethod
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        update metric based on a batch of outputs and a batch of targets
        :param outputs: batch of outputs
        :param targets: batch of targets
        :return: None
        """
        pass

    def _update(self, value: float, n: int = 1) -> None:
        """
        submit some value n times
        :param value:
        :param n:
        :return:
        """
        self._val = value
        self._sum += value * n
        self._count += n
        self._avg = self._sum / self._count

    @property
    def last(self) -> float:
        """
        :return: last value
        """
        return self._val

    @property
    def sum(self) -> float:
        """
        :return: sum of values
        """
        return self._sum

    @property
    def count(self) -> float:
        """
        :return: count of values
        """
        return self._count

    @property
    def avg(self) -> float:
        """
        :return: average of values
        """
        return self._avg

    @property
    def status(self) -> str:
        """
        :return: last status of metric formatted as string
        """
        return f"{self._val:.4f} ({self._avg:.4f})"

    @property
    def avg_status(self) -> str:
        """
        :return: average status of metric formatted as string
        """
        return f"{self._avg:.4f}"


class AverageMeter(FloatMetric):
    def update(self, value: int, n: int = 1) -> None:
        self._update(value, n)
