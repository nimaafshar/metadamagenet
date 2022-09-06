import pathlib
from dataclasses import dataclass

from metadamagenet.configs import GeneralConfig

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
