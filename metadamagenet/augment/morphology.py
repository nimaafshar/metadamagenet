from typing import Tuple

import torch

from .base import Transform
import kornia.morphology as km


class Dilation(Transform[torch.IntTensor]):
    def __init__(self, kernel_size: Tuple[int, int] = (5, 5)):
        super().__init__()
        self._kernel_size: Tuple[int, int] = kernel_size

    def generate_state(self, input_shape: torch.Size) -> torch.IntTensor:
        return torch.ones(self._kernel_size, device=self.device)

    def forward(self, images: torch.FloatTensor, state: torch.IntTensor) -> torch.FloatTensor:
        return km.dilation(images, state)
