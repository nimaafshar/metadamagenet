import abc
import dataclasses
from typing import Optional, Callable, Tuple, Type

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda import amp
from torchmetrics import Metric

from metadamagenet.wrapper import ModelWrapper
from metadamagenet.augment import TestTimeAugmentor
from metadamagenet.models import Metadata
from metadamagenet.dataset import ImagePreprocessor
from metadamagenet.validate import Validator
from metadamagenet.train import Trainer, ValidationInTrainingParams


def init_classifier(cls_wrapper_class: Type[ModelWrapper], loc_wrapper_class: Type[ModelWrapper], loc_version: str,
                    loc_seed: int) -> \
        Callable[[], Tuple[nn.Module, Metadata]]:
    def model_creator() -> Tuple[nn.Module, Metadata]:
        localizer, _ = loc_wrapper_class().from_checkpoint(loc_version, loc_seed)
        return cls_wrapper_class().from_unet(localizer.unet)

    return model_creator


class Command(abc.ABC):
    @abc.abstractmethod
    def run(self):
        pass


@dataclasses.dataclass
class Data:
    dataset: Dataset
    batch_size: int
    num_workers: int

    def dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)


@dataclasses.dataclass
class Validate(Command):
    model: Callable[[], Tuple[nn.Module, Metadata]]
    wrapper: ModelWrapper
    data: Data
    preprocessor: ImagePreprocessor
    score: Metric
    loss: Optional[nn.Module] = None
    device: Optional[str] = 'cuda'
    test_time_augmentor: Optional[TestTimeAugmentor] = None

    def run(self):
        model, metadata = self.model()
        print("metadata:", metadata)
        dataloader = self.data.dataloader()
        Validator(model, self.wrapper, dataloader, self.preprocessor, self.loss, self.score, self.device,
                  self.test_time_augmentor).validate()


@dataclasses.dataclass
class ValidateInTraining:
    data: Data
    preprocessor: ImagePreprocessor
    interval: int = 1
    score: Optional[Metric] = None
    test_time_augment: Optional[TestTimeAugmentor] = None

    def validation_params(self) -> ValidationInTrainingParams:
        return ValidationInTrainingParams(dataloader=self.data.dataloader(),
                                          preprocessor=self.preprocessor,
                                          interval=self.interval,
                                          score=self.score,
                                          test_time_augmentor=self.test_time_augment)


@dataclasses.dataclass
class Train(Command):
    model: Callable[[], Tuple[nn.Module, Metadata]]
    version: str
    seed: int
    wrapper: ModelWrapper
    data: Data
    preprocessor: ImagePreprocessor
    optimizer: Callable[[nn.Module], Optimizer]
    lr_scheduler: Callable[[Optimizer], MultiStepLR]
    loss: nn.Module
    score: Metric
    epochs: int
    random_seed: int
    device: Optional[str] = "cuda"
    grad_scaling: bool = True
    clip_grad_norm: Optional[float] = None
    validation: Optional[ValidateInTraining] = None

    def run(self):
        print(f"random seed: {self.random_seed + self.seed}")
        torch.manual_seed(self.random_seed + self.seed)

        model, metadata = self.model()
        print("metadata:", metadata)
        optimizer = self.optimizer(model)
        lr_scheduler = self.lr_scheduler(optimizer)
        grad_scaler = amp.grad_scaler.GradScaler() if self.grad_scaling else None
        Trainer(
            model,
            self.version,
            self.seed,
            self.wrapper,
            self.data.dataloader(),
            self.preprocessor,
            optimizer,
            lr_scheduler,
            self.loss,
            self.epochs,
            self.score,
            self.validation.validation_params() if self.validation is not None else None,
            metadata,
            self.device,
            grad_scaler,
            self.clip_grad_norm
        ).train()
