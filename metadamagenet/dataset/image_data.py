from dataclasses import dataclass
import pathlib
import json
from typing import List, Tuple

from shapely.geometry import Polygon
from shapely import wkt

from ..configs import DamageType, damage_to_damage_type, GeneralConfig
from .data_time import DataTime


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

    @property
    def localization_mask(self) -> pathlib.Path:
        """
        :return: predicted localization msk for this image
        """
        return self.base / GeneralConfig.get_instance().localization_dirname / f'{self.name(DataTime.PRE)}_part1.png'

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
