from dataclasses import dataclass
import json
from typing import Tuple
import pathlib

import torch
from typing_extensions import Self

from ..configs import GeneralConfig

config = GeneralConfig.get_instance()


@dataclass
class Checkpoint:
    model_name: str
    version: str
    seed: int

    @property
    def path(self) -> pathlib.Path:
        return config.models_root / pathlib.Path(self.name)

    @property
    def name(self) -> str:
        return f"{self.model_name.lower()}V{self.version.lower()}S{str(self.seed)}"

    @property
    def model_path(self) -> pathlib.Path:
        return self.path / "model.pth"

    @property
    def metadata_path(self) -> pathlib.Path:
        return self.path / "metadata.json"

    @property
    def exists(self) -> bool:
        return (self.path.exists() and
                self.path.is_dir() and
                self.model_path.exists() and
                self.metadata_path.exists())


@dataclass
class Metadata:
    best_score: float = 0
    trained_epochs: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return Metadata(
            best_score=d['best_score'],
            trained_epochs=d['trained_epochs']
        )

    def to_dict(self) -> dict:
        return {
            "best_score": self.best_score,
            "trained_epochs": self.trained_epochs
        }


class ModelManager:
    _instance: Self = None

    @classmethod
    def get_instance(cls) -> Self:
        if not cls._instance:
            cls._instance = ModelManager()
        return cls._instance

    def load_checkpoint(self, checkpoint: Checkpoint) -> Tuple[dict, Metadata]:
        """
        :param checkpoint:
        :return: (state_dict,metadata)
        """
        assert checkpoint.exists, "checkpoint does not exist"
        model_state_dict: dict = torch.load(checkpoint.model_path, map_location="cpu")
        with open(checkpoint.metadata_path, "r") as metadata_file:
            metadata: Metadata = Metadata.from_dict(json.load(metadata_file))
        return model_state_dict, metadata

    def save_checkpoint(self, checkpoint: Checkpoint, model_state_dict: dict, metadata: Metadata) -> None:
        checkpoint.path.mkdir(exist_ok=True)
        torch.save(model_state_dict, checkpoint.model_path)
        with open(checkpoint.metadata_path, "w") as metadata_file:
            json.dump(metadata.to_dict(), metadata_file)
