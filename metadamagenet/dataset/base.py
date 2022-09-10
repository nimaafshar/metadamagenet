import abc
from typing import Optional
from torch.utils.data import Dataset as TorchDataset

from .image_dataset import ImageDataset
from ..augment import Pipeline


class Dataset(TorchDataset, abc.ABC):
    def __init__(self,
                 image_dataset: ImageDataset,
                 augmentations: Optional[Pipeline] = None):
        """
        Train Dataset
        :param image_dataset: dataset of images
        :param augmentations: pipeline of augmentations
        """
        super().__init__()
        self._image_dataset: ImageDataset = image_dataset
        self._augments: Optional[Pipeline] = augmentations

    def __len__(self) -> int:
        return len(self._image_dataset)
