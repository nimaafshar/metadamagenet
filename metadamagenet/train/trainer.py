from dataclasses import dataclass
from typing import Optional
from contextlib import nullcontext
import gc

from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda import amp

from ..models import Checkpoint, Metadata
from ..models import Manager as ModelManager
from ..augment import TestTimeAugmentor
from ..metrics import Score, AverageMeter
from ..wrapper import ModelWrapper
from ..logging import log
from ..validate import Validator
from ..losses import MonitoredLoss


@dataclass
class ValidationInTrainingParams:
    dataloader: DataLoader
    score: Optional[Score]
    test_time_augmentor: Optional[TestTimeAugmentor]


class Trainer:

    def __init__(self,
                 model: nn.Module,
                 version: str,
                 seed: int,
                 model_wrapper: ModelWrapper,
                 dataloader: DataLoader,
                 optimizer: Optimizer,
                 lr_scheduler: MultiStepLR,
                 loss: nn.Module,
                 epochs: int,
                 score: Score,
                 validation_params: Optional[ValidationInTrainingParams] = None,
                 validation_interval: Optional[int] = None,
                 model_metadata: Metadata = Metadata(),
                 device: Optional[torch.device] = None,
                 grad_scaler: Optional[amp.GradScaler] = amp.GradScaler(),
                 clip_grad_norm: Optional[float] = None
                 ):
        self._model: nn.Module = model
        self._version: str = version
        self._seed: str = seed
        self._wrapper: ModelWrapper = model_wrapper
        self._dataloader: DataLoader = dataloader
        self._optimizer: Optimizer = optimizer
        self._lr_scheduler: MultiStepLR = lr_scheduler
        self._loss: nn.Module = loss
        self._epochs: int = epochs
        self._score: Score = score

        self._validation_params = validation_params
        self._validation_interval: int = validation_interval
        if validation_params is None and validation_interval is not None:
            raise ValueError("cannot validate without validation_params")
        if validation_params is not None and validation_interval is None:
            raise ValueError("validation_params passed but validation_interval is None")

        self._model_metadata: Metadata = model_metadata
        self._device: Optional[torch.device] = device
        self._grad_scaler: Optional[amp.GradScaler] = grad_scaler
        self._clip_grad_norm: Optional[float] = clip_grad_norm

    def train(self) -> None:
        log(f':arrow_forward: starting to train model {self._wrapper.model_name}'
            f' version {self._version} seed {self._seed}')
        log(f'steps_per_epoch: {len(self._dataloader)}')

        best_score: float = self._model_metadata.best_score
        for epoch in range(self._model_metadata.trained_epochs + 1, self._epochs):
            log(f"===> :repeat_one: epoch {epoch}")
            log(f"======>:relieved: training")
            torch.cuda.empty_cache()
            gc.collect()
            self._train_epoch()
            self._lr_scheduler.step()
            if self._validation_interval is not None and epoch % self._validation_interval == 0:
                torch.cuda.empty_cache()
                log(f"======>:fearful: validation")
                validator: Validator = self._make_validator()
                score: float = validator.validate()
                if self._score_improved(best_score, score):
                    best_score = score
                    self._save_model(epoch, best_score)

    def _score_improved(self, old_score: float, new_score: float) -> bool:
        if new_score > old_score:
            log(f":confetti_ball: score {old_score:.5f} --> {new_score:.5f}")
            return True
        elif new_score == old_score:
            log(f":neutral_face: score {old_score:.5f} --> {new_score:.5f}")
            return False
        else:
            log(f":disappointed: score {old_score:.5f} --> {new_score:.5f}")
            return False

    def _make_validator(self) -> Validator:
        return Validator(
            model=self._model,
            model_wrapper=self._wrapper,
            dataloader=self._validation_params.dataloader,
            loss=self._loss,
            score=self._validation_params.score if self._validation_params.score is not None else self._score,
            device=self._device,
            test_time_augmentor=self._validation_params.test_time_augmentor
        )

    def _train_epoch(self) -> None:
        self._model.train()
        loss_meter: AverageMeter = AverageMeter()
        iterator = tqdm(self._dataloader)

        i: int
        inputs: torch.Tensor  # (B,5,H,W) or (B,1,H,W)
        targets: torch.Tensor  # (B,5,H,W) or (B,1,H,W)
        for i, (inputs, targets) in enumerate(iterator):

            if self._device is not None:
                inputs = inputs.to(device=self._device, non_blocking=True)
                targets = targets.to(device=self._device, non_blocking=True)

            with amp.autocast() if self._grad_scaler is not None else nullcontext():
                outputs: torch.Tensor = self._model(inputs)
                loss: torch.Tensor = self._loss(outputs, targets)

            loss_status: str = "--"
            if isinstance(self._loss, MonitoredLoss) and self._loss.monitored:
                loss_status = self._loss.last_values()
            else:
                loss_meter.update(loss.item(), inputs.size(0))
                loss_status = loss_meter.status

            activated_outputs: torch.Tensor = self._wrapper.apply_activation(outputs)
            self._score.update(activated_outputs, targets)
            score_status: str = self._score.status()

            iterator.set_postfix({
                "loss": loss_status,
                "lr": f"{self._lr_scheduler.get_last_lr()[-1]:.7f}"
            })

            self._update_weights(loss)

        loss_status: str = "--"
        if isinstance(self._loss, MonitoredLoss) and self._loss.monitored:
            loss_status = self._loss.aggregate()
        else:
            loss_status = loss_meter.avg_status

        self._score.reset()

        score_status: str = self._score.avg_status()
        log(f"Training Results: loss: {loss_status} score:{score_status}")

    def _update_weights(self, loss: torch.Tensor):
        self._optimizer.zero_grad()
        if self._grad_scaler is not None:
            self._grad_scaler.scale(loss).backward()
            self._grad_scaler.unscale_(self._optimizer)
            if self._clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_grad_norm)
            self._grad_scaler.step(self._optimizer)
            self._grad_scaler.update()
        else:
            loss.backward()
            self._optimizer.step()
            if self._clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_grad_norm)

    def _save_model(self, epochs_trained: int, score: float) -> None:
        metadata: Metadata = Metadata(best_score=score, trained_epochs=epochs_trained)
        checkpoint: Checkpoint = Checkpoint(
            model_name=self._wrapper.model_name,
            version=self._version,
            seed=self._seed
        )
        manager: ModelManager = ModelManager.get_instance()
        manager.save_checkpoint(checkpoint, self._model.state_dict(), metadata)
