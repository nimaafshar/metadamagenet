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


def create_masks(image_data: ImageData) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    creates localization and classification mask for image data
    """
    localization_mask = np.zeros((1024, 1024), dtype='uint8')  # a mask-image including polygons
    classification_mask = np.zeros((1024, 1024), dtype='uint8')  # a mask-image with damage levels

    for polygon in image_data.polygons(DataTime.PRE):
        _msk = mask_for_polygon(polygon)
        localization_mask[_msk > 0] = 1

    damage_type: DamageType
    for polygon, damage_type in image_data.polygons(DataTime.POST):
        _msk = mask_for_polygon(polygon)
        classification_mask[_msk > 0] = damage_type_color[damage_type]

    return localization_mask, classification_mask


def create_and_save_targets(image_data: ImageData) -> None:
    localization_msk, classification_msk = create_masks(image_data)

    cv2.imwrite(str(image_data.mask(DataTime.PRE)),
                localization_msk,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])

    cv2.imwrite(str(image_data.mask(DataTime.POST)),
                classification_msk,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert args.path.exists() and args.path.is_dir(), \
        f"path {args.path.absolute()} does not exist or its not a directory"

    GeneralConfig.load()
    config = GeneralConfig.get_instance()
    # create mask directories
    (args.dir / config.masks_dirname).mkdir(exist_ok=True)
    image_dataset = discover_directory(args.path)

    with Pool() as pool:
        tqdm(pool.imap(create_and_save_targets, image_dataset), total=len(image_dataset))

    log("masks created")
