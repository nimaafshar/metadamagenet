import dataclasses
import pathlib
from typing import Type
from torch import nn

from configs import PREDICTIONS_DIRECTORY, MODELS_WEIGHTS_FOLDER


@dataclasses.dataclass
class ModelConfig:
    name: str
    model_type: Type[nn.Module]
    tuned: bool
    seed: int

    @property
    def pred_directory(self) -> pathlib.Path:
        """
        predictions directory path
        this method does not guarantee the directory to exist
        """
        return PREDICTIONS_DIRECTORY / f'{self.name}_{self.seed}_{"tuned" if self.tuned else ""}'

    @property
    def best_snap_path(self) -> pathlib.Path:
        return MODELS_WEIGHTS_FOLDER / f'{self.name}_{self.seed}_{"tuned" if self.tuned else ""}_best'
