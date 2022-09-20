from typing import Tuple

import torch
import kornia.filters as kf

from .base import Transform


class Blur(Transform[None]):
    def __init__(self, kernel_size: Tuple[int, int] = (3, 3)):
        super().__init__()
        self._kernel_size: Tuple[int, int] = kernel_size

    def generate_state(self, input_shape: torch.Size) -> None:
        return None

    def forward(self, images: torch.FloatTensor, _) -> torch.FloatTensor:
        return kf.box_blur(images, self._kernel_size)
