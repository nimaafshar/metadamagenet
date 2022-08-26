import numpy as np
import numpy.typing as npt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def normalize_colors(x: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """normalizes array values range from [0,255] to [-1,1]"""

    x: npt.NDArray[np.float32] = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x


