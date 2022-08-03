import dataclasses
import json
import typing
from typing import List, Dict, Iterator, Tuple, Iterable
import pathlib
import enum
from shapely.geometry import Polygon
from shapely import wkt

from configs import DamageType, damage_to_damage_type


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

    def name(self, time: DataTime = DataTime.PRE) -> str:
        """
        returns filename (without extension)
        """
        return f'{self.disaster}_{self.identifier}_{time.value}_disaster'


class Dataset:
    """
    Dataset of ImageDatas
    """

    def __init__(self, base_directories: Iterable[pathlib.Path]):
        self._base_directories: Iterable[pathlib.Path] = base_directories
        self._data: Dict[str, ImageData] = {}  # a mapping from identifier to ImageData instance

    def discover(self) -> None:
        """
        discover directories
        """
        for train_directory in self._base_directories:
            for file_path in (train_directory / 'images').glob('*_pre_disaster.png'):
                disaster, identifier, time, _ = file_path.name.split('_')
                self._data[identifier] = ImageData(train_directory, identifier, disaster)

    @property
    def images(self) -> typing.Dict.values:
        return self._data.values()

    def __getitem__(self, item) -> ImageData:
        return self._data[item]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)
