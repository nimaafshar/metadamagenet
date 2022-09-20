import dataclasses
from typing import Tuple

import torch
import kornia.enhance as ke

from .base import Transform

__all__ = ('Clahe', 'Brightness', 'Contrast', 'Saturation', 'RGBShift', 'RGBShiftState', 'HSVShift', 'HSVShiftState')


class Clahe(Transform[None]):
    """
    input values are expected to be in [0,1]
    """

    def __init__(self, clip_limit: float = 40, grid_size: Tuple[int, int] = (8, 8)):
        super().__init__()
        self._clip_limit: float = clip_limit
        self._grid_size: Tuple[int, int] = grid_size

    def generate_state(self, input_shape: torch.Size) -> None:
        return None

    def forward(self, images: torch.FloatTensor, _) -> torch.FloatTensor:
        return ke.equalize_clahe(images, self._clip_limit, self._grid_size)


class Brightness(Transform[torch.FloatTensor]):

    def __init__(self, factor_range: Tuple[float, float] = (0.2, 0.8)):
        super().__init__()
        self._factor_range: Tuple[float, float] = factor_range

    def generate_state(self, input_shape: torch.Size) -> torch.FloatTensor:
        return torch.rand((input_shape[0],)) * (self._factor_range[1] - self._factor_range[0]) + self._factor_range[0]

    def forward(self, images: torch.FloatTensor, state: torch.FloatTensor) -> torch.FloatTensor:
        return ke.adjust_brightness(images, state)


class Contrast(Transform[torch.FloatTensor]):

    def __init__(self, factor_range: Tuple[float, float] = (0.2, 0.8)):
        super(Contrast, self).__init__()
        self._factor_range: Tuple[float, float] = factor_range

    def generate_state(self, input_shape: torch.Size) -> torch.FloatTensor:
        return torch.rand((input_shape[0],)) * (self._factor_range[1] - self._factor_range[0]) + self._factor_range[0]

    def forward(self, images: torch.FloatTensor, state: torch.FloatTensor) -> torch.FloatTensor:
        return ke.adjust_contrast(images, state)


class Saturation(Transform[torch.FloatTensor]):
    def __init__(self, factor_range: Tuple[float, float] = (0.9, 1.1)):
        super(Saturation, self).__init__()
        self._factor_range: Tuple[float, float] = factor_range

    def generate_state(self, input_shape: torch.Size) -> torch.FloatTensor:
        return torch.rand((input_shape[0],)) * (self._factor_range[1] - self._factor_range[0]) + self._factor_range[0]

    def forward(self, images: torch.Tensor, state: torch.FloatTensor) -> torch.FloatTensor:
        return ke.adjust_saturation(images, state)


@dataclasses.dataclass
class RGBShiftState:
    r_shift: torch.FloatTensor
    g_shift: torch.FloatTensor
    b_shift: torch.FloatTensor


class RGBShift(Transform[RGBShiftState]):
    def __init__(self,
                 r_range: Tuple[int, int] = (-.1, .1),
                 g_range: Tuple[int, int] = (-.1, .1),
                 b_range: Tuple[int, int] = (-.1, .1)):
        """
        :param r_range: range of R channel shift
        :param g_range: range of G channel shift
        :param b_range: range of B channel shift
        """
        super().__init__()
        self._r_range: Tuple[int, int] = r_range
        self._g_range: Tuple[int, int] = g_range
        self._b_range: Tuple[int, int] = b_range

    def generate_state(self, input_shape: torch.Size) -> RGBShiftState:
        batch_size = input_shape[0]
        return RGBShiftState(
            r_shift=torch.rand((batch_size,)) * (self._r_range[1] - self._r_range[0]) + self._r_range[0],
            g_shift=torch.rand((batch_size,)) * (self._g_range[1] - self._g_range[0]) + self._g_range[0],
            b_shift=torch.rand((batch_size,)) * (self._b_range[1] - self._b_range[0]) + self._b_range[0]
        )

    def forward(self, images: torch.FloatTensor, state: RGBShiftState) -> torch.FloatTensor:
        return ke.shift_rgb(images, state.r_shift, state.g_shift, state.b_shift)


@dataclasses.dataclass
class HSVShiftState:
    h_shift: torch.FloatTensor
    s_shift: torch.FloatTensor
    v_shift: torch.FloatTensor


class HSVShift(Transform[HSVShiftState]):

    def __init__(self,
                 h_range: Tuple[int, int] = (-.1, .1),
                 s_range: Tuple[int, int] = (-.1, .1),
                 v_range: Tuple[int, int] = (-.1, .1)):
        """
        :param h_range: range of hue channel shift
        :param s_range: range of saturation channel shift
        :param v_range: range of brightness/value channel shift
        """
        super().__init__()
        self._h_range: Tuple[int, int] = h_range
        self._s_range: Tuple[int, int] = s_range
        self._v_range: Tuple[int, int] = v_range

    def generate_state(self, input_shape: torch.Size) -> HSVShiftState:
        batch_size = input_shape[0]
        return HSVShiftState(
            h_shift=torch.rand((batch_size,)) * (self._h_range[1] - self._h_range[0]) + self._h_range[0],
            s_shift=torch.rand((batch_size,)) * (self._s_range[1] - self._s_range[0]) + self._s_range[0],
            v_shift=torch.rand((batch_size,)) * (self._v_range[1] - self._v_range[0]) + self._v_range[0]
        )

    def forward(self, images: torch.FloatTensor, state: HSVShiftState) -> torch.FloatTensor:
        images = ke.adjust_hue(images, state.h_shift)
        images = ke.adjust_saturation(images, state.s_shift)
        return ke.adjust_brightness(images, state.v_shift)
