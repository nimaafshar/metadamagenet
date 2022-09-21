from typing import Tuple, Optional, Sequence
import dataclasses

import torch
import kornia.geometry as kg

from .base import Transform, CollectionTransform, ImageCollection
from .utils import random_float_tensor

__all__ = ('VFlip', 'Rotate90', 'Shift', 'RotateAndScaleState', 'RotateAndScale', 'ElasticTransform', 'BestCrop')


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

    def __init__(self, y: Tuple[float, float] = (.2, .8), x: Tuple[float, float] = (.2, .8)):
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
        return kg.translate(images, state * torch.tensor([h, w], device=self.device, dtype=torch.float32),
                            padding_mode='reflection')


class ElasticTransform(Transform[torch.FloatTensor]):
    def __init__(self, kernel_size: Tuple[int, int] = (63, 63), sigma: Tuple[float, float] = (10., 10.),
                 alpha: Tuple[float, float] = (0.0, 0.5)):
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
                 center_y: Tuple[float, float] = (0.3, 0.7),
                 center_x: Tuple[float, float] = (0.3, 0.7),
                 angle: Tuple[float, float] = (-10., 10.),
                 scale: Tuple[float, float] = (0.9, 1.1)
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
            scale=random_float_tensor((input_shape[0], 1), self._scale, device=self.device).repeat(1, 2)
        )

    def forward(self, images: torch.FloatTensor, state: RotateAndScaleState) -> torch.FloatTensor:
        _, _, h, w = images.size()
        transform_mat: torch.Tensor = kg.get_rotation_matrix2d(
            state.center * torch.tensor([h, w], device=self.device, dtype=torch.float32),
            state.angle,
            state.scale)
        return kg.transform.warp_affine(images,
                                        transform_mat,
                                        dsize=(images.size(-2), images.size(-1)),
                                        mode='bilinear',
                                        padding_mode='reflection')


class BestCrop(CollectionTransform):
    def __init__(self,
                 samples: int = 5,
                 size_range: Tuple[float, float] = (0.9, 1),
                 dsize: Tuple[int, int] = (512, 512),
                 msk_key: str = 'msk',
                 only_on: Optional[Sequence[str]] = None):
        super().__init__()
        self._samples: int = samples
        self._size_range: Tuple[float, float] = size_range
        self._dsize: Tuple[int, int] = dsize
        self._msk_key: str = msk_key
        self._only_on: Optional[Sequence[str]] = only_on

    def _generate_boxes(self, batch_size: int, count: int) -> torch.FloatTensor:
        sizes_rel = random_float_tensor((batch_size * count,), self._size_range, device=self.device) \
            .unsqueeze(1).repeat(1, 2)
        bottom_rel = sizes_rel.clone()
        bottom_rel[:, 0] = 0
        right_rel = sizes_rel.clone()
        right_rel[:, 1] = 0

        top_left_rel = torch.rand_like(sizes_rel) * (1 - sizes_rel)
        top_right_rel = top_left_rel + right_rel
        bottom_left_rel = top_left_rel + bottom_rel
        bottom_right_rel = top_left_rel + sizes_rel
        return torch.stack((top_left_rel, top_right_rel, bottom_right_rel, bottom_left_rel), dim=1)

    def forward(self, collection: ImageCollection) -> ImageCollection:
        b, c, h, w = collection[self._msk_key].size()
        img_shape = torch.tensor([h, w], device=self.device)
        repeated_msk = collection[self._msk_key].unsqueeze(1).repeat(1, self._samples, 1, 1, 1) \
            .reshape(b * self._samples, c, h, w)

        candidate_boxes: torch.FloatTensor = self._generate_boxes(b, self._samples) * img_shape
        candidates: torch.FloatTensor = kg.transform.crop_and_resize(repeated_msk, candidate_boxes, size=self._dsize,
                                                                     padding_mode='reflection') \
            .reshape(b, self._samples, -1)
        indices = candidates.sum(dim=2).argmax(dim=1, keepdim=True).unsqueeze(-1).repeat(1, 1, 4 * 2)
        boxes = candidate_boxes.reshape(b, self._samples, 4 * 2).gather(dim=1, index=indices).reshape(b, 4, 2)

        result: ImageCollection = {}
        keys = self._only_on if self._only_on is not None else collection.keys()
        for key, val in collection.items():
            if key in keys:
                result[key] = kg.transform.crop_and_resize(val, boxes, size=self._dsize, padding_mode='reflection')
            else:
                result[key] = val
        return result
