from abc import ABC

import numpy as np
import numpy.typing as npt
import torch
import cv2

from .predictor import SingleModelPredictor
from src.file_structure import ImageData, DataTime
from src.util.utils import normalize_colors
from src.util.augmentations import test_time_augment
from src.logs import log


class ClassificationPredictor(SingleModelPredictor, ABC):
    """
    predictor for classification models
    """

    def setup(self):
        # vis_dev = sys.argv[2]

        # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev
        # cudnn.benchmark = True
        pass

    def _process_input(self, image_data: ImageData) -> torch.Tensor:
        pre_image: npt.NDArray = cv2.imread(image_data.image(DataTime.PRE), cv2.IMREAD_COLOR)
        post_image: npt.NDArray = cv2.imread(image_data.image(DataTime.POST), cv2.IMREAD_COLOR)

        img: npt.NDArray = np.concatenate((pre_image, post_image), axis=2)
        img = normalize_colors(img)
        inp = test_time_augment(img)

        return torch.from_numpy(inp).float().cuda()


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
