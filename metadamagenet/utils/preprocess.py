import numpy as np
import numpy.typing as npt


def normalize_colors(x: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """normalizes array values range from [0,255] to [-1,1]"""

    x: npt.NDArray[np.float32] = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x
