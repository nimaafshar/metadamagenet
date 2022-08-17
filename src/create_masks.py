import emoji
import numpy as np

import cv2
import timeit
import sys
from multiprocessing import Pool

import shapely.geometry

import configs
from setup import single_thread_numpy, set_random_seeds
from file_structure import Dataset, ImageData, DataTime
from src.logs import log

single_thread_numpy()
set_random_seeds()
sys.setrecursionlimit(10000)


def mask_for_polygon(poly: shapely.geometry.Polygon, im_size=(1024, 1024)):
    """
    creates a binary mask from a polygon
    """

    def int_coords(x):
        return np.array(x).round().astype(np.int32)

    img_mask = np.zeros(im_size, np.uint8)
    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def create_masks(image_data: ImageData):
    """
    creates a polygon and a damage mask from image labels and saves them to `masks` folder
    """
    polygons_mask = np.zeros((1024, 1024), dtype='uint8')  # a mask-image including polygons
    damages_mask = np.zeros((1024, 1024), dtype='uint8')  # a mask-image with damage levels

    for polygon, _ in image_data.polygons(DataTime.PRE):
        _msk = mask_for_polygon(polygon)
        polygons_mask[_msk > 0] = 255

    for polygon, damage_type in image_data.polygons(DataTime.POST):
        _msk = mask_for_polygon(polygon)
        damages_mask[_msk > 0] = damage_type

    cv2.imwrite(str(image_data.mask(DataTime.PRE)),
                polygons_mask,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])

    cv2.imwrite(str(image_data.mask(DataTime.POST)),
                damages_mask,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == '__main__':
    t0 = timeit.default_timer()

    configs.GeneralConfig.load()

    config: configs.GeneralConfig = configs.GeneralConfig.get_instance()

    train_dataset = Dataset(config.train_dirs)
    train_dataset.discover()

    # create mask directories
    for train_dir in config.train_dirs:
        (train_dir / 'masks').mkdir(exist_ok=True)

    with Pool() as pool:
        _ = pool.map(create_masks, train_dataset.images)

    elapsed = timeit.default_timer() - t0
    log(':hourglass: : {:.3f} min'.format(elapsed / 60))
