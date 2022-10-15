from dataclasses import dataclass
from typing import Optional, Dict
from contextlib import nullcontext
import gc
import logging

import emoji
from tqdm.autonotebook import tqdm
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda import amp
from torch.backends import cudnn
from torchmetrics import MeanMetric

from .base import Runner
from ..models import Checkpoint, Metadata, ModelManager, BaseModel
from ..augment import TestTimeAugmentor
from torchmetrics import Metric
from ..logging import EmojiAdapter
from .validator import Validator

logger = EmojiAdapter(logging.getLogger())


@dataclass
class ValidationInTrainingParams:
    dataloader: DataLoader
    interval: int = 1
    transform: Optional[nn.Module] = None
    score: Optional[Metric] = None
    test_time_augmentor: Optional[TestTimeAugmentor] = None


class Trainer(Runner):
    def __init__(self,
                 model: BaseModel,
                 version: str,
                 seed: int,
                 dataloader: DataLoader,
                 transform: nn.Module,
                 optimizer: Optimizer,
                 lr_scheduler: MultiStepLR,
                 loss: nn.Module,
                 epochs: int,
                 score: Metric,
                 validation_params: Optional[ValidationInTrainingParams] = None,
                 model_metadata: Metadata = Metadata(),
                 device: Optional[torch.device] = None,
                 grad_scaler: Optional[amp.GradScaler] = None,
                 clip_grad_norm: Optional[float] = None
                 ):

        self._device: Optional[torch.device] = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self._model: BaseModel = model.to(self._device) if self._device is not None else model
        self._version: str = version
        self._seed: str = seed
        self._dataloader: DataLoader = dataloader
        self._dataloader.pin_memory = True
        self._transform: nn.Module = transform.to(self._device)
        self._optimizer: Optimizer = optimizer
        self._lr_scheduler: MultiStepLR = lr_scheduler
        self._loss: nn.Module = loss.to(self._device)
        self._epochs: int = epochs
        self._score: Metric = score.to(self._device)
        self._validation_params = validation_params
        self._grad_scaler: Optional[amp.GradScaler] = grad_scaler
        self._clip_grad_norm: Optional[float] = clip_grad_norm

    def run(self) -> None:
        cudnn.benchmark = True
        logger.info(f':arrow_forward: starting to train model {self._model.name()}'
                    f" version='{self._version}' seed='{self._seed}'")
        logger.info(f'steps_per_epoch: {len(self._dataloader)}')

        best_score: float = self._model.metadata.best_score
        for epoch in range(1, self._epochs + 1):
            torch.cuda.empty_cache()
            gc.collect()
            self._train_epoch(epoch)
            self._lr_scheduler.step()
            if self._validation_params is not None and \
                    (epoch % self._validation_params.interval == 0 or epoch == self._epochs):
                torch.cuda.empty_cache()
                gc.collect()
                validator: Validator = self._make_validator()
                score: float = validator.run()
                if self._score_improved(best_score, score):
                    best_score = score
                    self._save_model(epoch, best_score)

    def _score_improved(self, old_score: float, new_score: float) -> bool:
        if new_score > old_score:
            logger.info(f":confetti_ball: score {old_score:.5f} --> {new_score:.5f}")
            return True
        elif new_score == old_score:
            logger.info(f":neutral_face: score {old_score:.5f} --> {new_score:.5f}")
            return False
        else:
            logger.info(f":disappointed: score {old_score:.5f} --> {new_score:.5f}")
            return False

    def _make_validator(self) -> Validator:
        return Validator(
            model=self._model,
            dataloader=self._validation_params.dataloader,
            transform=self._validation_params.transform,
            loss=self._loss,
            score=self._validation_params.score if self._validation_params.score is not None else self._score,
            device=self._device,
            test_time_augmentor=self._validation_params.test_time_augmentor
        )

    def _train_epoch(self, epoch: int) -> None:
        self._model.train()
        self._score.reset()
        loss_mean: MeanMetric = MeanMetric().to(self._device)
        iterator = tqdm(self._dataloader,
                        leave=False,
                        desc=emoji.emojize(f":repeat_one: Training {epoch}/{self._epochs}", language='alias'))
        i: int
        data_batch: Dict[str, torch.FloatTensor]
        for i, data_batch in enumerate(iterator):
            data_batch = {k: v.to(device=self._device, non_blocking=True) for k, v in data_batch.items()}

            inputs: torch.Tensor  # (B,5,H,W) or (B,1,H,W) with float values
            targets: torch.Tensor  # (B,H,W) with long values (0-4) or (0-1)
            with torch.no_grad():
                if self._transform is not None:
                    data_batch = self._transform(data_batch)
                inputs, targets = self._model.preprocess(data_batch)

            with amp.autocast() if self._grad_scaler is not None else nullcontext():
                outputs: torch.Tensor = self._model(inputs)
                loss: torch.Tensor = self._loss(outputs, targets)

            with torch.no_grad():
                activated_outputs: torch.Tensor = self._model.activate(outputs)
                current_score: torch.Tensor = self._score(activated_outputs, targets)
                loss_mean.update(loss)

            iterator.set_postfix({
                "loss": loss.item(),
                "score": current_score.item(),
                "lr": f"{self._lr_scheduler.get_last_lr()[-1]:.7f}"
            })
            self._update_weights(loss)

        logger.info(f"Training Results: loss: {loss_mean.compute().item()} score:{self._score.compute().item()}")

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
        self._model.metadata.best_score = score
        self._model.metadata.trained_epochs = epochs_trained
        checkpoint: Checkpoint = Checkpoint(
            model_name=self._model.name(),
            version=self._version,
            seed=self._seed
        )
        manager: ModelManager = ModelManager.get_instance()
        manager.save_checkpoint(checkpoint, self._model.state_dict(), self._model.metadata)
        logger.info(f"======> model saved at {checkpoint}")
