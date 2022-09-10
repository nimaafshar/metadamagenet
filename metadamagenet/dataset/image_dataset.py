from typing import Iterable, List, Iterator
import pathlib

from ..logging import log
from .exceptions import DatasetNotDiscovered
from .image_data import ImageData


class ImageDataset:
    """
    Dataset of ImageDatas
    """

    def __init__(self, base_directories: Iterable[pathlib.Path]):
        self._base_directories: Iterable[pathlib.Path] = base_directories
        self._data: List[ImageData] = []
        self._is_discovered: bool = False

    def discover(self) -> None:
        """
        discover directories
        """
        for base_directory in self._base_directories:
            log(f":mag: discovering {base_directory.absolute()}...")
            for file_path in (base_directory / 'images').glob('*_pre_disaster.png'):
                disaster, identifier, time, _ = file_path.name.split('_')
                self._data.append(ImageData(base_directory, identifier, disaster))

        self._is_discovered = True
        log(f":file_folder: {len(self)} files found.")

    def _assert_discovered(self) -> None:
        if not self._is_discovered:
            raise DatasetNotDiscovered()

    @property
    def images(self) -> List[ImageData]:
        self._assert_discovered()
        return self._data

    def __getitem__(self, item) -> ImageData:
        self._assert_discovered()
        return self._data[item]

    def __iter__(self) -> Iterator[ImageData]:
        self._assert_discovered()
        return iter(self._data)

    def __len__(self) -> int:
        self._assert_discovered()
        return len(self._data)
