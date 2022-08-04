import abc
import pathlib
from abc import ABC

import torch
import cv2
from tqdm import tqdm
import numpy.typing as npt
from typing import List, Sequence

from src.model_config import ModelConfig
from src.file_structure import Dataset, ImageData, DataTime
from src.logs import log


class Predictor(abc.ABC):
    def __init__(self, pred_directory: pathlib.Path, dataset: Dataset):
        self._pred_directory: pathlib.Path = pred_directory
        self._dataset: Dataset = dataset

    @abc.abstractmethod
    def setup(self):
        pass

    def _make_predictions_dir(self):
        self._pred_directory.mkdir(parents=False, exist_ok=True)
        log(f':file_folder: directory {self._pred_directory} created to save predictions.')

    @abc.abstractmethod
    def _process_input(self, image_data: ImageData) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _load(self):
        """load needed model or models"""
        pass

    @abc.abstractmethod
    def _make_prediction(self, inp: torch.Tensor) -> torch.Tensor:
        """
        make prediction for an instance
        when using a single model just returning the model output is fine
        but when using multiple models, their results should be aggregated
        """
        pass

    @abc.abstractmethod
    def _process_output(self, model_output: torch.Tensor) -> npt.NDArray:
        pass

    def _save_output(self, output_mask: npt.NDArray, image_data: ImageData) -> None:
        # write predictions to file
        # FIXME: what is part1 and part2?
        cv2.imwrite(str(self._pred_directory / f'{image_data.name(DataTime.PRE)}_part1.png'),
                    output_mask[..., :3],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

        cv2.imwrite(str(self._pred_directory / f'{image_data.name(DataTime.PRE)}_part2.png'),
                    output_mask[..., 2:],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def predict(self) -> None:
        self.setup()

        self._make_predictions_dir()

        self._load()

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


class SingleModelPredictor(Predictor, ABC):
    """A Predictor accepting only one model"""

    def __init__(self, model_config: ModelConfig, dataset: Dataset):
        super().__init__(model_config.pred_directory, dataset)
        self._model_config: ModelConfig = model_config
        self._model: torch.nn.Module

    def _load(self):
        self._load_model()

    def _load_model(self):
        log('=> loading best model...')
        self._model: torch.nn.Module = self._model_config.load_best_model()
        self._model.eval()

    def _make_prediction(self, inp: torch.Tensor) -> torch.Tensor:
        return self._model(inp)


class MultipleModelPredictor(Predictor, ABC):
    """A Predictor accepting multiple models and aggregate their prediction"""

    def __init__(self, model_configs: Sequence[ModelConfig], pred_directory: pathlib.Path, dataset: Dataset):
        super(MultipleModelPredictor, self).__init__(pred_directory, dataset)
        self._model_configs: Sequence[ModelConfig] = model_configs
        self._models: List[torch.nn.Module] = []

    def _load_models(self):
        for model_config in self._model_configs:
            log(f"==> loading model {model_config.name} [seed={model_config.seed},tuned={model_config.tuned}]")
            self._models.append(model_config.load_best_model())

    def _load(self):
        self._load_models()
