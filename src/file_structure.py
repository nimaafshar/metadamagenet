import dataclasses
from typing import List, Dict, Iterator
import pathlib
import enum


class DataTime(enum.Enum):
    """
    refers to pre- or post-disaster
    """
    PRE = 'pre'
    POST = 'post'


@dataclasses.dataclass
class ImageData:
    """
    represents a data in our dataset
    """
    base: pathlib.Path
    identifier: str
    disaster: str

    def image(self, time: DataTime = DataTime.PRE) -> pathlib.Path:
        """
        path to pre/post disaster image
        """
        return self.base / 'images' / f'{self.disaster}_{self.identifier}_{time.value}_disaster.png'

    def label(self, time: DataTime = DataTime.PRE) -> pathlib.Path:
        return self.base / 'labels' / f'{self.disaster}_{self.identifier}_{time.value}_disaster.json'

    def target(self, time: DataTime = DataTime.PRE) -> pathlib.Path:
        return self.base / 'target' / f'{self.disaster}_{self.identifier}_{time.value}_disaster.png'


class Dataset:
    """
    Dataset of ImageDatas
    """

    def __init__(self, train_directories: List[pathlib.Path]):
        self._base_directories: List[pathlib.Path] = train_directories
        self._data: Dict[str, ImageData] = {}  # a mapping from identifier to ImageData instance

    def discover(self):
        """
        discover directories
        """
        for train_directory in self._base_directories:
            for file_path in train_directory.glob('*_pre_disaster.png'):
                disaster, identifier, time, _ = file_path.name.split('_')
                self._data[identifier] = ImageData(train_directory, identifier, disaster)

    def __getitem__(self, item) -> ImageData:
        return self._data[item]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)
