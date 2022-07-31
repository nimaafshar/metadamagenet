import numpy as np

import cv2
import timeit
from os import path, makedirs, listdir
import sys
from multiprocessing import Pool
from shapely import wkt

import json

from configs import damage_dict, TRAIN_DIRS, MASKS_DIR
from setup import single_thread_numpy, set_random_seeds

single_thread_numpy()
set_random_seeds()
sys.setrecursionlimit(10000)


def mask_for_polygon(poly, im_size=(1024, 1024)):
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


def process_image(json_file):
    """
    creates a polygon and a damage mask from image labels and saves them to `masks` folder
    """
    js1 = json.load(open(json_file))
    js2 = json.load(open(json_file.replace('_pre_disaster', '_post_disaster')))

    msk = np.zeros((1024, 1024), dtype='uint8')
    msk_damage = np.zeros((1024, 1024), dtype='uint8')

    for feat in js1['features']['xy']:
        poly = wkt.loads(feat['wkt'])
        _msk = mask_for_polygon(poly)
        msk[_msk > 0] = 255

    for feat in js2['features']['xy']:
        poly = wkt.loads(feat['wkt'])
        subtype = feat['properties']['subtype']
        _msk = mask_for_polygon(poly)
        msk_damage[_msk > 0] = damage_dict[subtype]

    cv2.imwrite(json_file.replace('/labels/', '/masks/').replace('_pre_disaster.json', '_pre_disaster.png'), msk,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(json_file.replace('/labels/', '/masks/').replace('_pre_disaster.json', '_post_disaster.png'),
                msk_damage, [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == '__main__':
    t0 = timeit.default_timer()

    all_files = []
    for d in TRAIN_DIRS:
        makedirs(d / MASKS_DIR, exist_ok=True)
        for f in sorted(listdir(path.join(d, 'images'))):
            if '_pre_disaster.png' in f:
                all_files.append(path.join(d, 'labels', f.replace('_pre_disaster.png', '_pre_disaster.json')))

    with Pool() as pool:
        _ = pool.map(process_image, all_files)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
