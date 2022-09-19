from typing import Tuple, Optional, Sequence
import dataclasses

import torch
import kornia.geometry as kg

from .base import RandomTransform, StateType


class GaussianNoise(RandomTransform[None]):
    """
    Gaussian Noise
    should be applied on images
    """

    def __init__(self, apply_to: Optional[Sequence[str]] = None, p: float = 0, mean: float = 0,
                 std: float = 1, clip_min: float = 0, clip_max: float = 255,
                 output_type: torch.dtype = torch.uint8):
        super().__init__(apply_to, p)
        self._std: float = std ** 0.5
        self._mean: float = mean
        self._clip_min: float = clip_min
        self._clip_max: float = clip_max
        self._output_type: torch.dtype = output_type

    def generate_random_state(self, batch_size: int) -> None:
        return None

    def transform(self, images: torch.Tensor, _) -> torch.Tensor:
        noise: torch.FloatTensor = torch.randn(images.size()) * self._std + self._mean
        return torch.clip(images + noise, self._, 255).to(dtype=self._output_type)
