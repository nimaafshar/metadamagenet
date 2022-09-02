import dataclasses
import gc
import pathlib
from typing import Optional, Tuple, Type

from torch import nn
import torch

from src.configs import GeneralConfig
from src.logs import log


@dataclasses.dataclass
class ModelConfig:
    name: str
    model_type: Type[nn.Module]
    version: str
    seed: int
    data_parallel: bool = False

    @property
    def full_name(self) -> str:
        return f'{self.name}_{self.seed}_{self.version}'

    @property
    def empty_model(self) -> nn.Module:
        model = self.model_type().cuda()
        if self.data_parallel:
            model = nn.DataParallel(model).cuda()
        return model

    @property
    def pred_directory(self) -> pathlib.Path:
        """
        predictions directory path
        this method does not guarantee the directory to exist
        """
        return GeneralConfig.get_instance().predictions_dir / self.full_name

    @property
    def best_snap_path(self) -> pathlib.Path:
        return GeneralConfig.get_instance().model_weights_dir / f'{self.full_name}_best'

    def load_best_model(self) -> Tuple[nn.Module, Optional[float], int]:
        """
        loading model from best pretrained version
        :return: model instance
        """
        log(f":arrow_up: loading checkpoint '{self.best_snap_path}'")
        model, best_score, epochs_trained = \
            ModelConfig.load_from_checkpoint_into_type(self.best_snap_path, self.empty_model)
        return model, best_score, epochs_trained + 1

    def init_weights_from(self, checkpoint: 'ModelConfig') -> Tuple[nn.Module, Optional[float], int]:
        """
        initializing this model data partly with another model config path
        :param checkpoint: a model config
        :return: (model instance, best score , start epoch)
        """
        log(f":arrow_up: loading checkpoint '{checkpoint.best_snap_path}'")
        model, best_score, epochs_trained = \
            ModelConfig.load_from_checkpoint_into_type(checkpoint.best_snap_path, self.empty_model)

        if isinstance(self.empty_model, type(checkpoint.empty_model)):
            return model, best_score, epochs_trained + 1
        else:
            return model, None, 0

    @staticmethod
    def load_from_checkpoint_into_type(checkpoint_path: pathlib.Path, empty_model: nn.Module) -> \
            Tuple[nn.Module, Optional[float], int]:
        checkpoint: dict = torch.load(checkpoint_path, map_location='cpu')
        loaded_dict: dict = checkpoint['state_dict']
        sd: dict = empty_model.state_dict()

        # loading parts of the state dict that are saved in the checkpoint
        for k in sd:
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]

        loaded_dict = sd
        empty_model.load_state_dict(loaded_dict)

        trained_epochs = checkpoint['epoch']
        best_score = checkpoint['best_score']
        log(f":white_check_mark: loaded checkpoint '{checkpoint_path}' "
            f"[epoch={trained_epochs}, best_score={best_score}]")

        del loaded_dict
        del sd
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        return empty_model, best_score, trained_epochs
