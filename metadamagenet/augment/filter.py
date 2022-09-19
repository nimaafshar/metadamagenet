from typing import Tuple, Optional, Sequence

import torch
import kornia.filters as kf

from .base import RandomTransform, StateType


class Blur(RandomTransform[None]):
    def __init__(self, apply_to: Optional[Sequence[str]] = None, p: float = 0, kernel_size: Tuple[int, int] = (3, 3)):
        super().__init__(apply_to, p)
        self._kernel_size: Tuple[int, int] = kernel_size

    def generate_random_state(self, input_shape: torch.Size) -> None:
        return None

    def transform(self, images: torch.Tensor, _) -> torch.Tensor:
        return kf.box_blur(images, self._kernel_size)
