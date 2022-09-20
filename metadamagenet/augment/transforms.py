import torch

from .base import Transform
import kornia.geometry as kg


class Resize(Transform[None]):
    def __init__(self, height: int, width: int):
        """
        :param height: target image height
        :param width: target image width
        """
        super().__init__()
        self._height: int = height
        self._width: int = width

    def generate_state(self, input_shape: torch.Size) -> None:
        return None

    def forward(self, images: torch.FloatTensor, _) -> torch.FloatTensor:
        return kg.resize(images, size=(self._height, self._width))
