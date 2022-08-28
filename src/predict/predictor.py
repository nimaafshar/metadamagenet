import abc
import pathlib
from abc import ABC
from typing import List, Sequence, Optional

import torch
from torch import nn
from tqdm import tqdm
import numpy.typing as npt

from src.model_config import ModelConfig
from src.file_structure import Dataset, ImageData
from src.logs import log


class Predictor(abc.ABC):
    def __init__(self, pred_directory: pathlib.Path, dataset: Dataset):
        self._pred_directory: pathlib.Path = pred_directory
        self._dataset: Dataset = dataset

    @abc.abstractmethod
    def setup(self):
        pass

    def _make_predictions_dir(self):
        self._pred_directory.mkdir(parents=True, exist_ok=True)
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

    @abc.abstractmethod
    def _save_output(self, output_mask: npt.NDArray, image_data: ImageData) -> None:
        pass

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
        log(f"=> loading best model... {self._model_config.name} "
            f"[seed={self._model_config.seed},version={self._model_config.version}]")
        model: nn.Module
        best_score: Optional[float]
        start_epoch: int
        self._model, best_score, start_epoch = self._model_config.load_best_model()
        self._model = self._model.cuda()
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
            log(f"==> loading model {model_config.name} [seed={model_config.seed},version={model_config.version}]")
            model: nn.Module
            best_score: Optional[float]
            start_epoch: int
            model, best_score, start_epoch = model_config.load_best_model()
            model = model.cuda()
            model.eval()
            self._models.append(model)

    def _load(self):
        self._load_models()
