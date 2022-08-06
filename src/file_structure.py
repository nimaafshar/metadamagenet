import dataclasses
import json
import typing
from typing import List, Dict, Iterator, Tuple, Iterable
import pathlib
import enum
from shapely.geometry import Polygon
from shapely import wkt

from src.logs import log

from src.configs import (
    DamageType,
    damage_to_damage_type,
    IMAGES_DIRECTORY,
    LABELS_DIRECTORY,
    MASKS_DIRECTORY
)


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

    def name(self, time: DataTime = DataTime.PRE) -> str:
        """
        returns filename (without extension)
        """
        return f'{self.disaster}_{self.identifier}_{time.value}_disaster'

    def image(self, time: DataTime = DataTime.PRE) -> pathlib.Path:
        """
        :param time: pre/post-disaster version
        :return: path to image file
        """
        return self.base / IMAGES_DIRECTORY / f'{self.name(time)}.png'

    def label(self, time: DataTime = DataTime.PRE) -> pathlib.Path:
        """
        :param time: pre/post-disaster version
        :return: path to json label file
        """
        return self.base / LABELS_DIRECTORY / f'{self.name(time)}.json'

    def mask(self, time: DataTime = DataTime.PRE) -> pathlib.Path:
        """
        pre-disaster masks contain only black and white pixels
        post-disaster masks contain pixels with 0-4 values indicating damage level
        :param time: pre/post-disaster version
        :return: path to mask image file
        """
        return self.base / MASKS_DIRECTORY / f'{self.name(time)}.png'

    def polygons(self, time: DataTime = DataTime.PRE) -> List[Tuple[Polygon, DamageType]]:
        """
        list of image polygons and their subtypes
        for per disaster images it returns
        """
        with open(self.label(time)) as json_file:
            json_data = json.load(json_file)

        polygons: List[Tuple[Polygon, DamageType]] = []
        for feat in json_data['features']['xy']:
            polygon: Polygon = wkt.loads(feat['wkt'])
            subtype: DamageType = feat.get('properties', {}).get('subtype', DamageType.UN_CLASSIFIED)
            if isinstance(subtype, str):
                subtype = damage_to_damage_type[subtype]
            polygons.append((polygon, subtype))

        return polygons


class DatasetNotDiscovered(Exception):
    """when a dataset is not discovered this error is raised"""


class Dataset:
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
