import torch

from . import ImageCollection
from .base import CollectionTransform
import kornia.geometry as kg


class Resize(CollectionTransform):
    def forward(self, img_group: ImageCollection) -> ImageCollection:
        return {k: kg.resize(v, size=(self._height, self._width)) for k, v in img_group}

    def __init__(self, height: int, width: int):
        """
        :param height: target image height
        :param width: target image width
        """
        super().__init__()
        self._height: int = height
        self._width: int = width
