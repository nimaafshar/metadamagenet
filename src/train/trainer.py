import dataclasses
from typing import Tuple, Union, Optional
import abc

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model_config import ModelConfig
from src.train.dataset import Dataset
from src import configs
from src.logs import log


@dataclasses.dataclass
class TrainingConfig:
    model_config: ModelConfig
    input_shape: Tuple[int, int]
    epochs: int
    batch_size: int
    val_batch_size: int
    train_dataset: Dataset
    validation_dataset: Dataset
    evaluation_interval: int = 1
    start_checkpoint: Optional[ModelConfig] = None


class Trainer(abc.ABC):
    def __init__(self, config: TrainingConfig):
        self._config: TrainingConfig = config
        self._steps_per_epoch: int = len(self._config.train_dataset) // self._config.batch_size
        self._validation_steps: int = len(self._config.validation_dataset) // self._config.val_batch_size
        self._train_data_loader: DataLoader
        self._val_data_loader: DataLoader
        self._train_data_loader, self._val_data_loader = self._get_dataloaders()

    def _setup(self):
        configs.MODELS_WEIGHTS_FOLDER.mkdir(parents=False, exist_ok=True)

    @abc.abstractmethod
    def _get_dataloaders(self) -> (DataLoader, DataLoader):
        """
        :return: (train_data_loader, valid_data_loader)
        """
        pass

    def _get_model(self) -> nn.Module:
        """
        loading model from model_config. start from best snap if available. otherwise, just return an instance of the model
        :return: model instance
        """
        if self._config.start_checkpoint is None:
            if self._config.model_config.best_snap_path.exists():
                log(":eyes: model snap for your original exists. loading from snap...")
                return self._config.model_config.load_best_model().cuda()
            else:
                log(":poop: no model snap found. starting from scratch")
                return self._config.model_config.model_type().cuda()
        else:
            log(":watch: model snap for your start checkpoint exists. loading from snap...")
            # TODO: load starting checkpoint model weights into model_config model
            return self._config.start_checkpoint.load_best_model().cuda()

    @abc.abstractmethod
    def _save_model(self, epoch: int, score: float, best_score: Union[float, None]) -> bool:
        pass

    @abc.abstractmethod
    def _train_epoch(self, epoch: int) -> float:
        pass

    @abc.abstractmethod
    def _evaluate(self, number: int) -> float:
        pass

    @abc.abstractmethod
    def _update_best_score(self, score: float, best_score: Union[float, None]) -> float:
        pass

    def train(self):
        self._setup()
        log(f':arrow_forward: starting to train model {self._config.model_config.full_name} ...')
        log(f'[steps_per_epoch: {self._steps_per_epoch}, validation_steps:{self._validation_steps}]')

        best_score: Union[float, None] = None
        evaluation_round: int = -1
        torch.cuda.empty_cache()
        for epoch in range(self._config.epochs):
            log(f"===> :repeat_one: epoch {epoch}")
            self._train_epoch(epoch)
            if epoch % self._config.evaluation_interval == 0:
                evaluation_round += 1
                torch.cuda.empty_cache()
                score: float = self._evaluate(evaluation_round)
                self._save_model(epoch, score, best_score)
                best_score = self._update_best_score(score, best_score)
