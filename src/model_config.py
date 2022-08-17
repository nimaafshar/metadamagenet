import dataclasses
import gc
import pathlib
from typing import Type

from torch import nn
import torch

from src.configs import GeneralConfig
from src.logs import log

config = GeneralConfig.get_instance()


@dataclasses.dataclass
class ModelConfig:
    name: str
    model_type: Type[nn.Module]
    version: str
    seed: int

    @property
    def full_name(self) -> str:
        return f'{self.name}_{self.seed}_{self.version}'

    @property
    def pred_directory(self) -> pathlib.Path:
        """
        predictions directory path
        this method does not guarantee the directory to exist
        """
        return config.predictions_dir / self.full_name

    @property
    def best_snap_path(self) -> pathlib.Path:
        return config.model_weights_dir / f'{self.full_name}_best'

    def load_best_model(self) -> nn.Module:
        """
        loading model from best pretrained version
        :return: model instance
        """
        log(f":arrow_up: loading checkpoint '{self.best_snap_path}'")
        return ModelConfig.load_from_checkpoint_into_type(self.best_snap_path,self.model_type)

    def init_weights_from(self, checkpoint: 'ModelConfig') -> nn.Module:
        """
        initializing this model data partly with another model config path
        :param checkpoint: a model config
        :return: model instance
        """
        log(f":arrow_up: loading checkpoint '{checkpoint.best_snap_path}'")
        return ModelConfig.load_from_checkpoint_into_type(checkpoint.best_snap_path, self.model_type)

    @staticmethod
    def load_from_checkpoint_into_type(checkpoint_path: pathlib.Path, model_type: Type) -> nn.Module:
        model = model_type().cuda()
        checkpoint: dict = torch.load(checkpoint_path, map_location='cpu')
        loaded_dict: dict = checkpoint['state_dict']
        sd: dict = model.state_dict()

        # loading parts of the state dict that are saved in the checkpoint
        for k in sd:
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]

        loaded_dict = sd
        model.load_state_dict(loaded_dict)

        log(f":white_check_mark: loaded checkpoint '{checkpoint_path}' "
            f"[epoch={checkpoint['epoch']}, best_score={checkpoint['best_score']}]")

        del loaded_dict
        del sd
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        return model
