import dataclasses
import pathlib
from typing import Type

from torch import nn
import torch

from src.configs import PREDICTIONS_DIRECTORY, MODELS_WEIGHTS_FOLDER
from src.logs import log


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

    def load_best_model(self) -> nn.Module:
        model: nn.Module = self.model_type().cuda()
        model = nn.DataParallel(model).cuda()

        log(f":arrow_up: loading checkpoint '{self.best_snap_path}'")
        checkpoint: dict = torch.load(self.best_snap_path, map_location='cpu')
        loaded_dict: dict = checkpoint['state_dict']
        sd: dict = model.state_dict()

        # loading parts of the state dict that are saved in the checkpoint
        for k in sd:
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]

        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        log(f":white_check_mark: loaded checkpoint '{self.best_snap_path}' "
            f"[epoch={checkpoint['epoch']}, best_score={checkpoint['best_score']}]")

        return model
