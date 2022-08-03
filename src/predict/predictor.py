import abc

from torch import nn
import torch
import cv2
from tqdm import tqdm
import numpy.typing as npt

from src.model_config import ModelConfig
from src.file_structure import Dataset, ImageData, DataTime
from src.logs import log


class Predictor(abc.ABC):
    def __init__(self, model_config: ModelConfig, dataset: Dataset):
        self._model_config: ModelConfig = model_config
        self._dataset: Dataset = dataset

    @abc.abstractmethod
    def setup(self):
        pass

    def _make_predictions_dir(self):
        self._model_config.pred_directory.mkdir(parents=False, exist_ok=True)
        log(f':file_folder: directory {self._model_config.pred_directory} created to save predictions.')

    @abc.abstractmethod
    def _process_input(self, image_data: ImageData) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _load_model(self):
        """load needed model or models"""
        pass

    @abc.abstractmethod
    def _make_prediction(self, inp: torch.Tensor) -> torch.Tensor:
        """make prediction for an instance"""
        pass

    @abc.abstractmethod
    def _process_output(self, model_output: torch.Tensor) -> npt.NDArray:
        pass

    def _save_output(self, output_mask: npt.NDArray, image_data: ImageData) -> None:
        # write predictions to file
        # FIXME: what is part1 and part2?
        cv2.imwrite(str(self._model_config.pred_directory / f'{image_data.name(DataTime.PRE)}_part1.png'),
                    output_mask[..., :3],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

        cv2.imwrite(str(self._model_config.pred_directory / f'{image_data.name(DataTime.PRE)}_part2.png'),
                    output_mask[..., 2:],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def predict(self) -> None:
        self._make_predictions_dir()

        self._load_model()

        log('=> discovering dataset...')
        self._dataset.discover()

        log('=> making predictions...')
        with torch.no_grad():
            image_data: ImageData
            for image_data in tqdm(self._dataset.images):
                inp: torch.Tensor = self._process_input(image_data)

                output: torch.Tensor = self._make_prediction(inp)

                msk: npt.NDArray = self._process_output(output)
                self._save_output(msk, image_data)

        log('=> predicting job done.')
