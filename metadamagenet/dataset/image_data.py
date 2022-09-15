from dataclasses import dataclass
import pathlib
import json
from typing import List, Tuple, Iterable

from shapely.geometry import Polygon
from shapely import wkt

from ..configs import DamageType, damage_to_damage_type, GeneralConfig
from .data_time import DataTime
from ..logging import log


@dataclass
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
        return self.base / GeneralConfig.get_instance().images_dirname / f'{self.name(time)}.png'

    def label(self, time: DataTime = DataTime.PRE) -> pathlib.Path:
        """
        :param time: pre/post-disaster version
        :return: path to json label file
        """
        return self.base / GeneralConfig.get_instance().labels_dirname / f'{self.name(time)}.json'

    def mask(self, time: DataTime = DataTime.PRE) -> pathlib.Path:
        """
        pre-disaster masks contain only black and white pixels
        post-disaster masks contain pixels with 0-4 values indicating damage level
        :param time: pre/post-disaster version
        :return: path to mask image file
        """
        return self.base / GeneralConfig.get_instance().masks_dirname / f'{self.name(time)}_target.png'

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


def discover_directories(directories: Iterable[pathlib.Path], check: bool = True) -> List[ImageData]:
    results: List[ImageData] = []
    directory: pathlib.Path
    for directory in directories:
        results.extend(discover_directory(directory))
    return results


def discover_directory(base_directory: pathlib.Path, check: bool = True) -> List[ImageData]:
    """
    :param base_directory: directory to discover
    :param check: check if other files of image data exist
    :return: list of image datas
    """
    results: List[ImageData] = []
    if not base_directory.is_dir():
        raise ValueError(f"{base_directory.absolute()} is not a directory")

    log(f":mag: discovering {base_directory.absolute()}...")
    for file_path in (base_directory / 'images').glob('*_pre_disaster.png'):
        disaster, identifier, time, _ = file_path.name.split('_')
        image_data: ImageData = ImageData(base_directory, identifier, disaster)
        if check:
            path: pathlib.Path
            for path in [
                image_data.image(DataTime.PRE),
                image_data.image(DataTime.POST),
                image_data.label(DataTime.PRE),
                image_data.label(DataTime.POST),
                image_data.mask(DataTime.PRE),
                image_data.mask(DataTime.POST)
            ]:
                assert path.exists(), f"{path} does not exist"
                assert path.is_file(), f"{path} is not a file"
        results.append(image_data)

    if len(results) == 0:
        log(f"warning: directory {base_directory} is empty")

    return results
