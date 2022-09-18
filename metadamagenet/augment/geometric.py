from typing import Tuple, Optional, Sequence
import dataclasses

import torch
import kornia.geometry as kg

from .base import RandomTransform, StateType


class RandomVFlip(RandomTransform[None]):
    """image vertical flip"""

    def transform(self, images: torch.Tensor, _) -> torch.Tensor:
        return kg.vflip(images)

    def generate_random_state(self, _) -> None:
        return None


class Random90Rot(RandomTransform[torch.IntTensor]):
    """rotates image 90 degrees randomly between 0-3 times """

    def generate_random_state(self, batch_size: int) -> StateType:
        return torch.randint(low=0, high=4, size=(batch_size,)) * 90

    def transform(self, images: torch.Tensor, state: torch.IntTensor) -> torch.Tensor:
        return kg.rotate(images, state)


class Shift(RandomTransform[torch.IntTensor]):
    """shifts image. moving the shift point to (0,0). replaces empty pixels with reflection"""

    def __init__(self, probability: float,
                 y_range: Tuple[int, int],
                 x_range: Tuple[int, int],
                 apply_to: Optional[Sequence[str]] = None):
        """
        :param probability: 1 - probability of this augmentation being applied
        :param y_range: shift range in y-axis
        :param x_range: shift range in x-axis
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(apply_to, probability)
        self._y_range: Tuple[int, int] = y_range
        self._x_range: Tuple[int, int] = x_range

    def generate_random_state(self, batch_size: int) -> torch.IntTensor:
        y_shift: torch.IntTensor = torch.randint(low=self._y_range[0], high=self._y_range[1], size=(batch_size,))
        x_shift: torch.IntTensor = torch.randint(low=self._x_range[0], high=self._y_range[1], size=(batch_size,))
        return torch.stack((y_shift, x_shift), dim=1)

    def transform(self, images: torch.Tensor, state: torch.IntTensor) -> torch.Tensor:
        return kg.translate(images, state, padding_mode='reflect')


@dataclasses.dataclass
class RotateAndScaleState:
    center_y: torch.IntTensor
    center_x: torch.IntTensor
    angle: torch.IntTensor
    scale: torch.FloatTensor


class RotateAndScale(RandomTransform[RotateAndScaleState]):
    """ rotate image around a center and scale"""

    def __init__(self,
                 probability: float,
                 center_y_range: Tuple[int, int],
                 center_x_range: Tuple[int, int],
                 angle_range: Tuple[int, int],
                 scale_range: Tuple[float, float],
                 apply_to: Optional[Sequence[str]] = None
                 ):
        """
        :param probability: 1 - probability of this augmentation being applied
        :param center_y_range: y-range of center of rotation with respect to image center,
            should be less than half of image height
        :param center_x_range: x-range of center of rotation with respect to image center,
            should be less than half of image width
        :param angle_range: rotation angle range in degrees
        :param scale_range: scale range
        :param apply_to: list of dict keys to apply the augmentation to, if none,
        the augmentation will be applied to all the batch.
        """
        super().__init__(probability, apply_to)
        self._center_y_range: Tuple[int, int] = center_y_range
        self._center_x_range: Tuple[int, int] = center_x_range
        self._angle_range: Tuple[int, int] = angle_range
        self._scale_range: Tuple[float, float] = scale_range

    def generate_random_state(self, batch_size: int) -> RotateAndScaleState:
        return RotateAndScaleState(
            center_y=torch.randint(low=self._center_y_range[0], high=self._center_y_range[1], size=(batch_size,)),
            center_x=torch.randint(low=self._center_x_range[0], high=self._center_x_range[1], size=(batch_size,)),
            angle=torch.randint(low=self._angle_range[0], high=self._angle_range[1], size=(batch_size,)),
            scale=torch.rand(size=(batch_size,)) * (self._scale_range[1] - self._scale_range[0]) + self._scale_range[0]
        )

    def transform(self, images: torch.Tensor, state: RotateAndScaleState) -> torch.Tensor:
        center: torch.IntTensor = torch.stack((state.center_y + (images.size(-2) // 2),
                                               state.center_x + (images.size(-1) // 2)), dim=1)
        transform_mat: torch.Tensor = kg.get_affine_matrix2d(torch.zeros_like(center), center, state.scale, state.angle)
        return kg.warp_affine(images,
                              transform_mat,
                              dsize=(images.size(-2), images.size(-1)),
                              mode='bilinear',
                              padding_mode='reflect')
