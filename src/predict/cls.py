from abc import ABC

import numpy as np
import numpy.typing as npt
import torch
import cv2

from .predictor import SingleModelPredictor
from src.file_structure import ImageData, DataTime
from src.util.utils import normalize_colors
from src.util.augmentations import test_time_augment, revert_augmentation
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

    def _save_output(self, output_mask: npt.NDArray, image_data: ImageData) -> None:
        # write predictions to file
        # FIXME: what is part1 and part2?
        cv2.imwrite(str(self._pred_directory / f'{image_data.name(DataTime.PRE)}_part1.png'),
                    output_mask[..., :3],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

        cv2.imwrite(str(self._pred_directory / f'{image_data.name(DataTime.PRE)}_part2.png'),
                    output_mask[..., 2:],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])


class SigmoidClassificationPredictor(ClassificationPredictor):
    def _process_output(self, model_output: torch.Tensor) -> npt.NDArray:
        img_batch: npt.NDArray = torch.sigmoid(model_output).cpu().numpy()

        msk = revert_augmentation(img_batch) * 255

        return msk.astype('uint8').transpose(1, 2, 0)


class SoftmaxClassificationPredictor(ClassificationPredictor):
    def _process_output(self, model_output: torch.Tensor) -> npt.NDArray:
        msk: npt.NDArray = torch.softmax(model_output[:, :, ...], dim=1).cpu().numpy()

        # FIXME: what is this line for
        msk[:, 0, ...] = 1 - msk[:, 0, ...]

        msk = revert_augmentation(msk) * 255

        return msk.astype('uint8').transpose(1, 2, 0)
