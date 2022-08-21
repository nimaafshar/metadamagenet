import os

import torch
from numpy import typing as npt
import numpy as np

from .predictor import MultipleModelPredictor
from src.file_structure import ImageData, DataTime
from src.util.utils import normalize_colors
from src.util.augmentations import test_time_augment, revert_augmentation
import cv2


class LocalizationPredictor(MultipleModelPredictor):

    def setup(self):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        # os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
        # cudnn.benchmark = True

    def _process_input(self, image_data: ImageData) -> torch.Tensor:
        img: npt.NDArray = cv2.imread(str(image_data.image(DataTime.PRE)), cv2.IMREAD_COLOR)

        img = normalize_colors(img)
        inp = test_time_augment(img)

        return torch.from_numpy(inp).float().cuda()

    def _make_prediction(self, inp: torch.Tensor) -> torch.Tensor:
        pred = []
        for model in self._models:
            msk_batch = model(inp)
            msk_batch = torch.sigmoid(msk_batch).cpu().numpy()

            pred.extend(revert_augmentation(msk_batch))

        # aggregating model results by calculating the mean
        return np.asarray(pred).mean(axis=0)

    def _process_output(self, model_output: torch.Tensor) -> npt.NDArray:
        return (model_output * 255).astype('uint8').transpose(1, 2, 0)

    def _save_output(self, output_mask: npt.NDArray, image_data: ImageData) -> None:
        cv2.imwrite(str(self._pred_directory / f"{image_data.name(DataTime.PRE)}_part1.png"),
                    output_mask[..., 0],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
