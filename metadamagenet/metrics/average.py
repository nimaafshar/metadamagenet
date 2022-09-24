import math

from .base import Metric


class AverageMetric(Metric):
    def __init__(self):
        self._avg: float = 0
        self._sum: float = 0
        self._count: float = 0

    def update(self, value: float, count: int = 1) -> None:
        """
        consider value n times
        """
        if math.isnan(value):
            return
        self._sum += value * count
        self._count += count
        self._avg = self._sum / self._count

    def reset(self) -> None:
        """
        reset metric for another round of calculation
        """
        self._avg = 0
        self._sum = 0
        self._count = 0

    def till_here(self) -> float:
        return self._avg

    def status_till_here(self) -> str:
        return f"{self._avg:.4f}"
