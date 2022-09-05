import json
from typing import Tuple

import torch

from .checkpoint import Checkpoint
from .metadata import Metadata


class Manager:
    _instance: 'Manager' = None

    @classmethod
    def get_instance(cls) -> 'Manager':
        if not cls._instance:
            cls._instance = Manager()
        return cls._instance

    def load_checkpoint(self, checkpoint: Checkpoint) -> Tuple[dict, Metadata]:
        """
        :param checkpoint:
        :return: (state_dict,metadata)
        """
        assert checkpoint.exists, "checkpoint does not exist"
        model_state_dict: dict = torch.load(checkpoint.model_path, map_location="cpu")
        with open(checkpoint.metadata_path) as metadata_file:
            metadata: Metadata = Metadata.from_dict(json.load(metadata_file))
        return model_state_dict, metadata

    def save_checkpoint(self, checkpoint: Checkpoint, model_state_dict: dict, metadata: Metadata) -> None:
        checkpoint.path.mkdir(exist_ok=True)
        torch.save(model_state_dict, checkpoint.model_path)
        with open(checkpoint.metadata_path) as metadata_file:
            json.dump(metadata.to_dict(), metadata_file)
