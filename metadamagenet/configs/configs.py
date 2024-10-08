import pathlib
import ssl
import enum
from typing import Optional, List

import torch.hub
import yaml


class DamageType(enum.Enum):
    UN_CLASSIFIED = 0
    NO_DAMAGE = 1
    MINOR_DAMAGE = 2
    MAJOR_DAMAGE = 3
    DESTROYED = 4


damage_to_damage_type = {
    "un-classified": DamageType.UN_CLASSIFIED,
    "no-damage": DamageType.NO_DAMAGE,
    "minor-damage": DamageType.MINOR_DAMAGE,
    "major-damage": DamageType.MAJOR_DAMAGE,
    "destroyed": DamageType.DESTROYED
}

damage_dict = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1
}


class GeneralConfig:
    _instance: Optional['GeneralConfig'] = None

    @classmethod
    def get_instance(cls) -> 'GeneralConfig':
        if cls._instance is None:
            cls.load()
        return cls._instance

    @classmethod
    def load(cls, path: pathlib.Path = pathlib.Path('./config.yaml')) -> None:
        cls._instance = GeneralConfig(path)

    def __init__(self, path: pathlib.Path):
        with open(path, "r") as stream:
            source: dict = yaml.safe_load(stream)

        self.train_dirs: List[pathlib.Path] = [pathlib.Path(value) for value in source['train-dirs']]
        self.test_dirs: List[pathlib.Path] = [pathlib.Path(value) for value in source['test-dirs']]

        self.images_dirname: str = source['dir-names']['images']
        self.labels_dirname: str = source['dir-names']['labels']
        self.masks_dirname: str = source['dir-names']['masks']

        self.predictions_dir = pathlib.Path(source['predictions-dir'])
        self.models_root = pathlib.Path(source['models-root'])
        self.submissions_dir = pathlib.Path(source['submissions-dir'])

        self.torch_hub_dir = pathlib.Path(source['torch-hub-dir'])
        self.torch_hub_dir.mkdir(exist_ok=True)
        torch.hub.set_dir(str(self.torch_hub_dir))


# enable ssl verification. ssl verification is disabled so downloading models from pretrained-models be possible
# their certificate has expired.
# noinspection PyProtectedMember
ssl._create_default_https_context = ssl._create_unverified_context
