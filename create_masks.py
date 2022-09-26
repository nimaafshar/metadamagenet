import argparse
import pathlib

import numpy as np
import numpy.typing as npt
import cv2
import shapely.geometry
import torch
from tqdm.autonotebook import tqdm

from metadamagenet.configs import DamageType
from metadamagenet.dataset import ImageData, DataTime, discover_directory


class MaskCreator:
    damage_type_color = {
        DamageType.UN_CLASSIFIED: 1,
        DamageType.NO_DAMAGE: 1,
        DamageType.MINOR_DAMAGE: 2,
        DamageType.MAJOR_DAMAGE: 3,
        DamageType.DESTROYED: 4
    }

    @staticmethod
    def mask_for_polygon(poly: shapely.geometry.Polygon, im_size=(1024, 1024)):
        """
        creates a binary mask from a polygon
        """

        def int_coords(x):
            return np.array(x).round().astype(np.int32)

        img_mask = np.zeros(im_size, np.uint8)
        exteriors = [int_coords(poly.exterior.coords)]
        interiors = [int_coords(p.coords) for p in poly.interiors]
        cv2.fillPoly(img_mask, exteriors, 1)
        cv2.fillPoly(img_mask, interiors, 0)
        return img_mask

    @classmethod
    def create_loc_mask(cls, image_data: ImageData) -> npt.NDArray:
        """
        creates localization mask for image data
        """
        localization_mask = np.zeros((1024, 1024), dtype='uint8')  # a mask-image including polygons

        for polygon in image_data.polygons(DataTime.PRE):
            _msk = cls.mask_for_polygon(polygon)
            localization_mask[_msk > 0] = 1

        return localization_mask

    @classmethod
    def create_cls_mask(cls, image_data: ImageData) -> npt.NDArray:
        classification_mask = np.zeros((1024, 1024), dtype='uint8')  # a mask-image with damage levels
        damage_type: DamageType
        for polygon, damage_type in image_data.polygons(DataTime.POST):
            _msk = cls.mask_for_polygon(polygon)
            classification_mask[_msk > 0] = cls.damage_type_color[damage_type]
        return

    @classmethod
    def save_masks(cls, image_data: ImageData, localization_msk: torch.Tensor,
                   classification_msk: torch.Tensor) -> None:

        cv2.imwrite(str(image_data.mask(DataTime.PRE)),
                    localization_msk,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

        cv2.imwrite(str(image_data.mask(DataTime.POST)),
                    classification_msk,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def __init__(self, source: pathlib.Path):
        assert source.exists() and source.is_dir(), \
            f"source {source.absolute()} does not exist or its not a directory"
        self._source: pathlib.Path = source

    def run(self):
        image_dataset = discover_directory(self._source, check=False)
        for image_data in tqdm(image_dataset):
            localization_msk: torch.Tensor = self.create_loc_mask(image_data)
            classification_msk: torch.Tensor = self.create_cls_mask(image_data)
            self.save_masks(image_data, localization_msk, classification_msk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    mask_creator = MaskCreator(pathlib.Path(args.source))
    mask_creator.run()


if __name__ == '__main__':
    main()
