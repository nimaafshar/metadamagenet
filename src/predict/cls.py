from abc import ABC

import numpy as np
import numpy.typing as npt
import torch
import cv2

from .predictor import Predictor
from src.file_structure import ImageData, DataTime
from src.util.utils import normalize_colors
from src.logs import log


class ClassificationPredictor(Predictor, ABC):
    """
    predictor for classification models
    """

    def _load_model(self):
        log('=> loading best model...')
        self._model: torch.nn.Module = self._model_config.load_best_model()
        self._model.eval()

    def _make_prediction(self, inp: torch.Tensor) -> torch.Tensor:
        return self._model(inp)

    def _process_input(self, image_data: ImageData) -> torch.Tensor:
        pre_image: npt.NDArray = cv2.imread(image_data.image(DataTime.PRE), cv2.IMREAD_COLOR)
        post_image: npt.NDArray = cv2.imread(image_data.image(DataTime.POST), cv2.IMREAD_COLOR)

        img: npt.NDArray = np.concatenate((pre_image, post_image), axis=2)
        img = normalize_colors(img)

        # test-time augmentations
        inp: npt.NDArray = np.asarray((img,  # original
                                       img[::-1, ...],  # flip up-down
                                       img[:, ::-1, ...],  # flip left-right
                                       img[::-1, ::-1, ...]),
                                      dtype='float')  # flip along both x and y-axis (180 rotation)
        return torch.from_numpy(inp.transpose((0, 3, 1, 2))).float().cuda()


class SigmoidClassificationPredictor(ClassificationPredictor):
    def _process_output(self, model_output: torch.Tensor) -> npt.NDArray:
        msk: npt.NDArray = torch.sigmoid(model_output).cpu().numpy()

        pred = np.asarray((msk[0, ...],
                           msk[1, :, ::-1, :],  # flip left-right
                           msk[2, :, :, ::-1],  # flip from RGB to BRG
                           msk[3, :, ::-1, ::-1])).mean(axis=0)  # left-right and RGB to BRG flip

        msk = pred * 255
        return msk.astype('uint8').transpose(1, 2, 0)


class SoftmaxClassificationPredictor(ClassificationPredictor):
    def _process_output(self, model_output: torch.Tensor) -> npt.NDArray:
        msk: npt.NDArray = torch.softmax(model_output[:, :, ...], dim=1).cpu().numpy()

        # FIXME: what is this line for
        msk[:, 0, ...] = 1 - msk[:, 0, ...]

        pred_full = np.asarray((msk[0, ...],
                                msk[1, :, ::-1, :],
                                msk[2, :, :, ::-1],
                                msk[3, :, ::-1, ::-1])).mean(axis=0)

        msk = pred_full * 255
        return msk.astype('uint8').transpose(1, 2, 0)
