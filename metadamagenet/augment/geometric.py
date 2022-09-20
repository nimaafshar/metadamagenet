from typing import Tuple
import dataclasses

import torch
import kornia.geometry as kg

from .base import Transform
from .utils import random_float_tensor

__all__ = ('VFlip', 'Rotate90', 'Shift', 'RotateAndScaleState', 'RotateAndScale', 'ElasticTransform')


class VFlip(Transform[None]):
    """image vertical flip"""

    def forward(self, images: torch.FloatTensor, _) -> torch.FloatTensor:
        return kg.vflip(images)

    def generate_state(self, _) -> None:
        return None


class Rotate90(Transform[torch.IntTensor]):
    """rotates image 90 degrees randomly between 0-3 times """

    def generate_state(self, input_shape: torch.Size) -> torch.IntTensor:
        return torch.randint(low=0, high=4, size=(input_shape[0],), device=self.device).float() * 90

    def forward(self, images: torch.FloatTensor, state: torch.IntTensor) -> torch.FloatTensor:
        return kg.rotate(images, state)


class Shift(Transform[torch.FloatTensor]):
    """shifts image. moving the shift point to (0,0). replaces empty pixels with reflection"""

    def __init__(self, y: Tuple[float, float], x: Tuple[float, float]):
        """
        :param y: shift range in y-axis (relative to image height)
        :param x: shift range in x-axis (relative to image width)
        the augmentation will be applied to all the batch.
        """
        super().__init__()
        self._y: Tuple[float, float] = y
        self._x: Tuple[float, float] = x

    def generate_state(self, input_shape: torch.Size) -> torch.FloatTensor:
        return torch.stack((random_float_tensor((input_shape[0],), self._y, device=self.device),
                            random_float_tensor((input_shape[0],), self._x, device=self.device)), dim=1).to(self.device)

    def forward(self, images: torch.Tensor, state: torch.FloatTensor) -> torch.Tensor:
        _, _, h, w = images.size()
        return kg.translate(images, state * torch.FloatTensor([h, w]), padding_mode='reflect')


class ElasticTransform(Transform[torch.FloatTensor]):
    def __init__(self, kernel_size: Tuple[int, int] = (63, 63), sigma: Tuple[float, float] = (32., 32.),
                 alpha: Tuple[float, float] = (1., 1.)):
        super().__init__()
        self._kernel_size: Tuple[int, int] = kernel_size
        self._sigma: Tuple[float, float] = sigma
        self._alpha: Tuple[float, float] = alpha

    def generate_state(self, input_shape: torch.Size) -> torch.FloatTensor:
        b, _, h, w = input_shape
        return random_float_tensor((b, 2, h, w), (-1., 1.), device=self.device)

    def forward(self, images: torch.FloatTensor, state: torch.FloatTensor) -> torch.FloatTensor:
        return kg.elastic_transform2d(images, state, self._kernel_size, self._sigma, self._alpha)


@dataclasses.dataclass
class RotateAndScaleState:
    center: torch.FloatTensor  # B*2
    angle: torch.FloatTensor  # B
    scale: torch.FloatTensor  # B


class RotateAndScale(Transform[RotateAndScaleState]):
    """ rotate image around a center and scale"""

    def __init__(self,
                 center_y: Tuple[float, float],
                 center_x: Tuple[float, float],
                 angle: Tuple[float, float],
                 scale: Tuple[float, float]
                 ):
        """
        :param center_y: y-range of center of rotation relative to image height
        :param center_x: x-range of center of rotation relative to image width
        :param angle: rotation angle range in degrees
        :param scale: scale range
        """
        super().__init__()
        assert center_y[0] >= 0. and center_y[1] <= 1., "center_y should be in (0.,1.)"
        self._center_y: Tuple[float, float] = center_y
        assert center_x[0] >= 0. and center_x[1] <= 1., "center_x should be in (0.,1.)"
        self._center_x: Tuple[float, float] = center_x
        self._angle: Tuple[float, float] = angle
        self._scale: Tuple[float, float] = scale

    def generate_state(self, input_shape: torch.Size) -> RotateAndScaleState:
        return RotateAndScaleState(
            center=torch.stack((random_float_tensor((input_shape[0],), self._center_y, device=self.device),
                                random_float_tensor((input_shape[0],), self._center_x, device=self.device)),
                               dim=1).to(self.device),
            angle=random_float_tensor((input_shape[0],), self._angle, device=self.device),
            scale=random_float_tensor((input_shape[0],), self._scale, device=self.device)
        )

    def forward(self, images: torch.FloatTensor, state: RotateAndScaleState) -> torch.FloatTensor:
        transform_mat: torch.Tensor = kg.get_affine_matrix2d(torch.zeros_like(state.center),
                                                             state.center, state.scale, state.angle)
        return kg.warp_affine(images,
                              transform_mat,
                              dsize=(images.size(-2), images.size(-1)),
                              mode='bilinear',
                              padding_mode='reflect')
