from typing import Tuple
import dataclasses

import torch
import kornia.geometry as kg

from .base import Transform

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
        return torch.randint(low=0, high=4, size=(input_shape[0],)) * 90

    def forward(self, images: torch.FloatTensor, state: torch.IntTensor) -> torch.FloatTensor:
        return kg.rotate(images, state)


class Shift(Transform[torch.IntTensor]):
    """shifts image. moving the shift point to (0,0). replaces empty pixels with reflection"""

    def __init__(self, y_range: Tuple[int, int], x_range: Tuple[int, int]):
        """
        :param y_range: shift range in y-axis
        :param x_range: shift range in x-axis
        the augmentation will be applied to all the batch.
        """
        super().__init__()
        self._y_range: Tuple[int, int] = y_range
        self._x_range: Tuple[int, int] = x_range

    def generate_state(self, input_shape: torch.Size) -> torch.IntTensor:
        y_shift: torch.IntTensor = torch.randint(low=self._y_range[0], high=self._y_range[1], size=(input_shape[0],))
        x_shift: torch.IntTensor = torch.randint(low=self._x_range[0], high=self._y_range[1], size=(input_shape[0],))
        return torch.stack((y_shift, x_shift), dim=1)

    def forward(self, images: torch.Tensor, state: torch.IntTensor) -> torch.Tensor:
        return kg.translate(images, state, padding_mode='reflect')


@dataclasses.dataclass
class RotateAndScaleState:
    center_y: torch.IntTensor
    center_x: torch.IntTensor
    angle: torch.IntTensor
    scale: torch.FloatTensor


class RotateAndScale(Transform[RotateAndScaleState]):
    """ rotate image around a center and scale"""

    def __init__(self,
                 center_y: Tuple[int, int],
                 center_x: Tuple[int, int],
                 angle: Tuple[int, int],
                 scale: Tuple[float, float]
                 ):
        """
        :param center_y: y-range of center of rotation with respect to image center,
            should be less than half of image height
        :param center_x: x-range of center of rotation with respect to image center,
            should be less than half of image width
        :param angle: rotation angle range in degrees
        :param scale: scale range
        """
        super().__init__()
        self._center_y: Tuple[int, int] = center_y
        self._center_x: Tuple[int, int] = center_x
        self._angle: Tuple[int, int] = angle
        self._scale: Tuple[float, float] = scale

    def generate_state(self, input_shape: torch.Size) -> RotateAndScaleState:
        return RotateAndScaleState(
            center_y=torch.randint(low=self._center_y[0], high=self._center_y[1], size=(input_shape[0],)),
            center_x=torch.randint(low=self._center_x[0], high=self._center_x[1], size=(input_shape[0],)),
            angle=torch.randint(low=self._angle[0], high=self._angle[1], size=(input_shape[0],)),
            scale=torch.rand(size=(input_shape[0],)) * (self._scale[1] - self._scale[0]) + self._scale[0]
        )

    def forward(self, images: torch.FloatTensor, state: RotateAndScaleState) -> torch.FloatTensor:
        center: torch.IntTensor = torch.stack((state.center_y + (images.size(-2) // 2),
                                               state.center_x + (images.size(-1) // 2)), dim=1)
        transform_mat: torch.Tensor = kg.get_affine_matrix2d(torch.zeros_like(center), center, state.scale, state.angle)
        return kg.warp_affine(images,
                              transform_mat,
                              dsize=(images.size(-2), images.size(-1)),
                              mode='bilinear',
                              padding_mode='reflect')


class ElasticTransform(Transform[torch.FloatTensor]):
    def __init__(self, kernel_size: Tuple[int, int] = (63, 63), sigma: Tuple[float, float] = (32., 32.),
                 alpha: Tuple[float, float] = (1., 1.)):
        super().__init__()
        self._kernel_size: Tuple[int, int] = kernel_size
        self._sigma: Tuple[float, float] = sigma
        self._alpha: Tuple[float, float] = alpha

    def generate_state(self, input_shape: torch.Size) -> torch.FloatTensor:
        b, _, h, w = input_shape
        return torch.rand(b, 2, h, w) * 2 - 1

    def forward(self, images: torch.FloatTensor, state: torch.FloatTensor) -> torch.FloatTensor:
        return kg.elastic_transform2d(images, state, self._kernel_size, self._sigma, self._alpha)
