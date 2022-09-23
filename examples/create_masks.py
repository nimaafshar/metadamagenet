from typing import Tuple
import argparse
import sys
from multiprocessing import Pool
import pathlib

import numpy as np
import numpy.typing as npt
import cv2
import shapely.geometry
from tqdm.autonotebook import tqdm

from metadamagenet.configs import DamageType, GeneralConfig
from metadamagenet.utils import single_thread_numpy, set_random_seeds
from metadamagenet.dataset import ImageData, DataTime, discover_directory
from metadamagenet.logging import log

single_thread_numpy()
set_random_seeds()
sys.setrecursionlimit(10000)

damage_type_color = {
    DamageType.UN_CLASSIFIED: 1,
    DamageType.NO_DAMAGE: 1,
    DamageType.MINOR_DAMAGE: 2,
    DamageType.MAJOR_DAMAGE: 3,
    DamageType.DESTROYED: 4
}


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


def create_loc_mask(image_data: ImageData) -> npt.NDArray:
    """
    creates localization mask for image data
    """
    localization_mask = np.zeros((1024, 1024), dtype='uint8')  # a mask-image including polygons

    for polygon in image_data.polygons(DataTime.PRE):
        _msk = mask_for_polygon(polygon)
        localization_mask[_msk > 0] = 1

    return localization_mask,


def create_cls_mask(image_data: ImageData) -> npt.NDArray:
    classification_mask = np.zeros((1024, 1024), dtype='uint8')  # a mask-image with damage levels
    damage_type: DamageType
    for polygon, damage_type in image_data.polygons(DataTime.POST):
        _msk = mask_for_polygon(polygon)
        classification_mask[_msk > 0] = damage_type_color[damage_type]
    return classification_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=pathlib.Path, required=True)
    parser.add_argument('--dest', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert args.source.exists() and args.source.is_dir(), \
        f"source {args.path.absolute()} does not exist or its not a directory"

    assert args.dest.exists() and args.dest.is_dir(), \
        f"source {args.path.absolute()} does not exist or its not a directory"

    image_dataset = discover_directory(args.source)
    source: pathlib.Path = args.source
    dest: pathlib.Path = args.dest

    def create_and_save_targets(image_data: ImageData) -> None:
        localization_msk = create_loc_mask(image_data)
        classification_msk = create_cls_mask(image_data)

        cv2.imwrite(str(dest / f'{image_data.name(DataTime.PRE)}_target.png'),
                    localization_msk,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

        cv2.imwrite(str(source / f'{image_data.name(DataTime.POST)}_target.png'),
                    classification_msk,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

    with Pool() as pool:
        tqdm(pool.imap(create_and_save_targets, image_dataset), total=len(image_dataset))

    log("masks created")
