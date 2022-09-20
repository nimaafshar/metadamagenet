import torch

from .base import Transform

__all__ = ('GaussianNoise',)


class GaussianNoise(Transform[None]):
    """
    Gaussian Noise
    should be applied on images
    """

    def __init__(self, mean: float = 0, std: float = 1):
        super().__init__()
        self._std: float = std ** 0.5
        self._mean: float = mean

    def generate_state(self, batch_size: int) -> None:
        return None

    def forward(self, images: torch.FloatTensor, _) -> torch.FloatTensor:
        noise: torch.FloatTensor = torch.randn(images.size()) * self._std + self._mean
        return torch.clamp(images + noise, 0, 1)
