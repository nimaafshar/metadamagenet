from typing import Tuple

import kornia.color
import torch
import kornia.enhance as ke

from .base import Transform
from .utils import random_float_tensor

__all__ = ('Clahe', 'Brightness', 'Contrast', 'Saturation', 'RGBShift', 'HSVShift')


class Clahe(Transform[None]):
    """
    input values are expected to be in [0,1]
    """

    def __init__(self, clip_limit: float = 2., grid_size: Tuple[int, int] = (5, 5)):
        super().__init__()
        self._clip_limit: float = clip_limit
        self._grid_size: Tuple[int, int] = grid_size

    def generate_state(self, input_shape: torch.Size) -> None:
        return None

    def forward(self, images: torch.FloatTensor, _) -> torch.FloatTensor:
        return ke.equalize_clahe(images, self._clip_limit, self._grid_size)


class Brightness(Transform[torch.FloatTensor]):

    def __init__(self, factor: Tuple[float, float] = (-0.1, 0.1)):
        super().__init__()
        self._factor: Tuple[float, float] = factor

    def generate_state(self, input_shape: torch.Size) -> torch.FloatTensor:
        return random_float_tensor((input_shape[0],), self._factor, device=self.device)

    def forward(self, images: torch.FloatTensor, state: torch.FloatTensor) -> torch.FloatTensor:
        return ke.adjust_brightness(images, state)


class Contrast(Transform[torch.FloatTensor]):

    def __init__(self, factor: Tuple[float, float] = (0.9, 1.1)):
        super(Contrast, self).__init__()
        self._factor: Tuple[float, float] = factor

    def generate_state(self, input_shape: torch.Size) -> torch.FloatTensor:
        return random_float_tensor((input_shape[0],), self._factor, device=self.device)

    def forward(self, images: torch.FloatTensor, state: torch.FloatTensor) -> torch.FloatTensor:
        return ke.adjust_contrast(images, state)


class Saturation(Transform[torch.FloatTensor]):
    def __init__(self, factor: Tuple[float, float] = (0.5, 1.5)):
        super(Saturation, self).__init__()
        self._factor: Tuple[float, float] = factor

    def generate_state(self, input_shape: torch.Size) -> torch.FloatTensor:
        return random_float_tensor((input_shape[0],), self._factor, device=self.device)

    def forward(self, images: torch.Tensor, state: torch.FloatTensor) -> torch.FloatTensor:
        return ke.adjust_saturation(images, state)


class RGBShift(Transform[torch.FloatTensor]):
    def __init__(self,
                 r: Tuple[float, float] = (-.1, .1),
                 g: Tuple[float, float] = (-.1, .1),
                 b: Tuple[float, float] = (-.1, .1)):
        """
        :param r: range of R channel shift
        :param g: range of G channel shift
        :param b: range of B channel shift
        """
        super().__init__()
        self._r_range: Tuple[int, int] = r
        self._g_range: Tuple[int, int] = g
        self._b_range: Tuple[int, int] = b

    def generate_state(self, input_shape: torch.Size) -> torch.FloatTensor:
        batch_size = input_shape[0]
        return torch.stack(
            (random_float_tensor((batch_size,), self._r_range, device=self.device),
             random_float_tensor((batch_size,), self._g_range, device=self.device),
             random_float_tensor((batch_size,), self._b_range, device=self.device)), dim=1) \
            .to(self.device)

    def forward(self, images: torch.FloatTensor, state: torch.FloatTensor) -> torch.FloatTensor:
        return (images + state).clamp_(min=0, max=1)


class HSVShift(Transform[torch.FloatTensor]):

    def __init__(self,
                 h: Tuple[float, float] = (-.1, .1),
                 s: Tuple[float, float] = (-.1, .1),
                 v: Tuple[float, float] = (-.1, .1)):
        """
        :param h: range of hue channel shift (will be multiplied by 2*pi)
        :param s: range of saturation channel shift
        :param v: range of brightness/value channel shift
        """
        super().__init__()
        self._h_range: Tuple[int, int] = h
        self._s_range: Tuple[int, int] = s
        self._v_range: Tuple[int, int] = v

    def generate_state(self, input_shape: torch.Size) -> torch.FloatTensor:
        batch_size = input_shape[0]
        return torch.stack(
            (random_float_tensor((batch_size,), self._h_range, self.device) * (2 * torch.pi),
             random_float_tensor((batch_size,), self._s_range, self.device),
             random_float_tensor((batch_size,), self._v_range, self.device)), dim=1) \
            .to(self.device)

    def forward(self, images: torch.FloatTensor, state: torch.FloatTensor) -> torch.FloatTensor:
        return kornia.color.hsv_to_rgb(kornia.color.rgb_to_hsv(images) + state).clamp_(min=0, max=1)
