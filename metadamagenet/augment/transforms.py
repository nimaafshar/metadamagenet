from typing import Optional, Sequence

import torch

from .base import Transform
import kornia.geometry as kg
import kornia.enhance.shift_rgb


class Resize(Transform[None]):
    def __init__(self, height: int, width: int, apply_to: Optional[Sequence[str]] = None):
        """
        :param height: target image height
        :param width: target image width
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(apply_to)
        self._height: int = height
        self._width: int = width

    def transform(self, images: torch.Tensor, _) -> torch.Tensor:
        return kg.resize(images, size=(self._height, self._width))
