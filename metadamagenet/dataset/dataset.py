from abc import ABC
from typing import Dict, Union, Sequence, Iterable
from pathlib import Path

import torch
from torch.utils.data import Dataset
import kornia.io as kio
from kornia.io import ImageLoadType

from .data_time import DataTime
from .image_data import ImageData, discover_directories, discover_directory


class Xview2Dataset(Dataset, ABC):
    def __init__(self, source: Union[Sequence[ImageData], Iterable[Path], Path], check: bool = False):
        super(Xview2Dataset, self).__init__()
        self._image_dataset: Sequence[ImageData]
        if isinstance(source, Sequence) and isinstance(source[0], ImageData):
            self._image_dataset = source
        elif isinstance(source, Path):
            self._image_dataset = discover_directory(source, check)
        elif isinstance(source, Iterable) and isinstance(source[0], Path):
            self._image_dataset = discover_directories(source, check)


class LocalizationDataset(Xview2Dataset):
    def __init__(self, source: Union[Sequence[ImageData], Iterable[Path], Path],
                 check: bool = False,
                 use_post_disaster_images: Union[bool, float] = 0.015):
        """
        Train Dataset
        :param source: source of images, could be a folder
        :param use_post_disaster_images: if true, the dataset will include all post-disaster images,
        if false, this dataset won't include any post-disaster image. if float value passed,
        this value should be a probability in [0,1) and with this probability pre-disaster images will be replaced
        with their corresponding post-disaster image
        """
        super().__init__(source, check)
        self._post_version_prob: float
        self._use_post_disaster_images: bool
        if isinstance(use_post_disaster_images, float):
            if not 0 <= use_post_disaster_images < 1:
                raise ValueError("post_version_prob should be in [0,1]")
            self._post_version_prob = use_post_disaster_images
            self._use_post_disaster_images = False
        elif isinstance(use_post_disaster_images, bool):
            self._use_post_disaster_images = use_post_disaster_images
            self._post_version_prob = 0.
        else:
            raise TypeError(f"unsupported type for 'use_post_disaster_images'"
                            f" expected Union[bool,float] got {type(use_post_disaster_images)}")

    def __len__(self) -> int:
        if self._use_post_disaster_images:
            return 2 * len(self._image_dataset)
        return len(self._image_dataset)

    def __getitem__(self, identifier: int) -> Dict[str, torch.FloatTensor]:
        """
        :param identifier: # of image data
        """
        image_data: ImageData
        data_time: DataTime
        if self._use_post_disaster_images:
            image_data = self._image_dataset[identifier // 2]
            data_time = DataTime.PRE if identifier % 2 == 0 else DataTime.POST
        else:
            image_data = self._image_dataset[identifier]
            data_time = DataTime.PRE

        img: torch.FloatTensor
        msk: torch.FloatTensor
        img = kio.load_image(str(image_data.image(DataTime.PRE)), ImageLoadType.RGB32)
        msk = kio.load_image(str(image_data.mask(DataTime.PRE)), ImageLoadType.UNCHANGED) \
            .float().mean(dim=0, keepdim=True)

        if not self._use_post_disaster_images and torch.rand(1).item() > self._post_version_prob:
            img = kio.load_image(str(image_data.image(DataTime.POST)), ImageLoadType.RGB32)
            # localization mask is the same as pre-disaster version

        return {"img": img, "msk": msk}


class ClassificationDataset(Xview2Dataset):
    def __len__(self):
        return len(self._image_dataset)

    def __getitem__(self, identifier: int) -> Dict[str, torch.FloatTensor]:
        """
        :param identifier: # of image data
        """
        image_data: ImageData = self._image_dataset[identifier]

        pre_image: torch.IntTensor = kio.load_image(str(image_data.image(DataTime.PRE)), ImageLoadType.RGB32)
        post_image: torch.IntTensor = kio.load_image(str(image_data.image(DataTime.POST)), ImageLoadType.RGB32)
        post_msk: torch.FloatTensor = kio.load_image(str(image_data.mask(DataTime.POST)),
                                                     ImageLoadType.UNCHANGED).float().mean(dim=0, keepdim=True) / 4

        # TODO: normalize colors, one-hot labels and concat pre-and-post disaster images
        return {
            "img_pre": pre_image,
            "img_post": post_image,
            "msk": post_msk
        }
