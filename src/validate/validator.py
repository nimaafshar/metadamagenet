import abc
import dataclasses

from torch import nn
from torch.utils.data import DataLoader

from src.logs import log
from src.model_config import ModelConfig


@dataclasses.dataclass
class ValidationConfig:
    model_config: ModelConfig
    dataloader: DataLoader
    dice_threshold: float = 0.5


class Validator(abc.ABC):
    def __init__(self, config: ValidationConfig):
        self._config: ValidationConfig = config
        self._model: nn.Module
        self._model, _, _ = config.model_config.load_best_model()
        self._dataloader: DataLoader = config.dataloader
        self._evaluation_dice_thr = config.dice_threshold

    def _setup(self):
        pass

    def validate(self):
        self._setup()

        log(f':arrow_forward: starting to train model {self._config.model_config.full_name} ...')
        log(f'steps: {len(self._dataloader)}')

        self._evaluate()

    @abc.abstractmethod
    def _evaluate(self) -> float:
        pass
