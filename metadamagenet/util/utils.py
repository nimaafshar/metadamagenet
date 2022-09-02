import numpy as np
import numpy.typing as npt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self._val: float = 0
        self._avg: float = 0
        self._sum: float = 0
        self._count: float = 0

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, value: float, n: int = 1) -> None:
        self._val = value
        self._sum += value * n
        self._count += n
        self._avg = self._sum / self._count

    @property
    def val(self) -> float:
        return self._val

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def count(self) -> float:
        return self._count

    @property
    def avg(self) -> float:
        return self._avg


def normalize_colors(x: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """normalizes array values range from [0,255] to [-1,1]"""

    x: npt.NDArray[np.float32] = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x
