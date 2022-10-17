import dataclasses
import logging
from typing import Optional, Dict, Callable

import emoji
import higher
import torch
import gc
from torch import nn
from torch.backends import cudnn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import Metric, MeanMetric
from tqdm.autonotebook import tqdm

from ..logging import EmojiAdapter
from ..models import BaseModel, Metadata, Checkpoint, ModelManager
from ..dataset import MetaDataLoader, TaskSet, Task
from .base import Runner

logger = EmojiAdapter(logging.getLogger())


class MetaValidator(Runner):
    def __init__(self,
                 model: BaseModel,
                 meta_dataloader: MetaDataLoader,
                 score: Metric,
                 inner_opt: Callable[[nn.Module, ], torch.optim.Optimizer],
                 n_inner_iter: int,
                 transform: Optional[nn.Module] = None,
                 loss: nn.Module = None,
                 device: Optional[torch.device] = None):
        self._device: torch.device
        if device is not None:
            self._device = device
        else:
            self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._model: BaseModel = model.to(self._device)
        self._transform: nn.Module = transform
        if self._transform is not None:
            self._transform = self._transform.to(self._device)
        self._meta_dataloader: MetaDataLoader = meta_dataloader
        self._loss: nn.Module = loss.to(self._device)

        self._score: Metric = score.to(self._device)
        self._n_inner_iter: int = n_inner_iter
        self._inner_optim: Callable[[nn.Module, ], torch.optim.Optimizer] = inner_opt

    def _move_batch_to_device(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._device is not None:
            data_batch = {k: v.to(device=self._device, non_blocking=True) for k, v in
                          data_batch.items()}
        return data_batch

    def _transform(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            if self._transform is not None:
                data_batch = self._transform(data_batch)
        return data_batch

    def run(self) -> float:
        # Crucially in our testing procedure here, we do *not* fine-tune
        # the model during testing for simplicity.
        # Most research papers using MAML for this task do an extra
        # stage of fine-tuning here that should be added if you are
        # adapting this code for research.
        logger.info(f':arrow_forward: starting to validate model {self._model.name()} in meta-learning setup')
        logger.info(f'meta-steps: {len(self._meta_dataloader)}')

        self._model.train()
        self._score.reset()
        query_losses: MeanMetric = MeanMetric().to(self._device)

        iterator = tqdm(self._meta_dataloader,
                        leave=False,
                        desc=emoji.emojize(":fearful: Task Set Meta-Validation", language='alias'))

        task_set: TaskSet
        for task_set in iterator:
            inner_optim = self._inner_optim(self._model)

            task: Task
            for task in task_set:
                with higher.innerloop_ctx(self._model, self._inner_optim, track_higher_grads=False) \
                        as (f_model, diff_optim):
                    # Optimize the likelihood of the support set by taking
                    # gradient steps w.r.t. the model's parameters.
                    # This adapts the model's meta-parameters to the task.
                    support_batch: Dict[str, torch.Tensor] = next(iter(task.support))  # get one batch from dataloader
                    support_batch = self._move_batch_to_device(support_batch)
                    support_batch = self._transform(support_batch)
                    inputs: torch.Tensor
                    targets: torch.Tensor
                    inputs, targets = self._model.preprocess(support_batch)
                    for _ in range(self._n_inner_iter):
                        outputs: torch.Tensor = f_model(inputs)
                        loss_value = self._loss(outputs, targets)
                        diff_optim.step(loss_value)

                    # The query loss and score induced by these parameters.
                    query_batch: Dict[str, torch.Tensor] = next(iter(task.query))  # get one batch from dataloader
                    query_batch = self._move_batch_to_device(query_batch)
                    query_batch = self._transform(query_batch)
                    query_inputs: torch.Tensor
                    query_targets: torch.Tensor
                    query_inputs, query_targets = self._model.preprocess(query_batch)

                    query_outputs: torch.Tensor = f_model(query_inputs).detach()
                    query_loss_value = self._loss(query_outputs, query_targets).detach()
                    query_losses.update(query_loss_value)

                    with torch.no_grad():
                        activated_query_outputs: torch.Tensor = self._model.activate(query_outputs)
                        query_score: torch.Tensor = self._score(activated_query_outputs, query_targets).detach()

            iterator.set_postfix({
                "loss": query_losses.compute().item(),
                "score": self._score.compute().item()
            })
        logger.info("validation results:" + str({
            "loss": query_losses.compute().item(),
            "score": self._score.compute().item()
        }))
        return self._score.compute().item()


@dataclasses.dataclass
class MetaValidationInTrainingParams:
    meta_dataloader: MetaDataLoader
    interval: int = 1
    transform: Optional[nn.Module] = None
    score: Optional[Metric] = None


class MetaTrainer(Runner):
    def __init__(self,
                 model: BaseModel,
                 version: str,
                 seed: int,
                 meta_dataloader: MetaDataLoader,
                 transform: nn.Module,
                 meta_opt: Optimizer,
                 inner_opt: Callable[[nn.Module], Optimizer],
                 lr_scheduler: MultiStepLR,
                 loss: nn.Module,
                 epochs: int,
                 n_inner_iter: int,
                 score: Metric,
                 validation_params: Optional[MetaValidationInTrainingParams] = None,
                 model_metadata: Metadata = Metadata(),
                 device: Optional[torch.device] = None,
                 ):

        self._device: Optional[torch.device] = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self._model: BaseModel = model.to(self._device)
        self._version: str = version
        self._seed: str = seed
        self._meta_dataloader: MetaDataLoader = meta_dataloader
        self._transform: nn.Module = transform.to(self._device)
        self._meta_opt: Optimizer = meta_opt
        self._inner_opt: Callable[[nn.Module], Optimizer] = inner_opt
        self._lr_scheduler: MultiStepLR = lr_scheduler
        self._loss: nn.Module = loss.to(self._device)
        self._epochs: int = epochs
        self._n_inner_iter: int = n_inner_iter
        self._score: Metric = score.to(self._device)
        self._validation_params = validation_params

    def run(self) -> None:
        cudnn.benchmark = True
        logger.info(f':arrow_forward: starting to train model {self._model.name()}'
                    f" version='{self._version}' seed='{self._seed}' with MAML")
        logger.info(f'task sets in an epoch: {len(self._meta_dataloader)}')

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
                validator: MetaValidator = self._make_validator()
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

    def _make_validator(self) -> MetaValidator:
        return MetaValidator(
            model=self._model,
            meta_dataloader=self._validation_params.meta_dataloader,
            inner_opt=self._inner_opt,
            n_inner_iter=self._n_inner_iter,
            transform=self._validation_params.transform,
            loss=self._loss,
            score=self._validation_params.score if self._validation_params.score is not None else self._score,
            device=self._device
        )

    def _move_batch_to_device(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._device is not None:
            data_batch = {k: v.to(device=self._device, non_blocking=True) for k, v in
                          data_batch.items()}
        return data_batch

    def _transform(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            if self._transform is not None:
                data_batch = self._transform(data_batch)
        return data_batch

    def _train_epoch(self, epoch: int) -> None:
        self._model.train()
        iterator = tqdm(self._meta_dataloader,
                        leave=False,
                        desc=emoji.emojize(f":repeat_one: MAML Training {epoch}/{self._epochs}", language='alias'))

        task_set: TaskSet
        for task_set in iterator:
            inner_optim: Optimizer = self._inner_opt(self._model)

            self._meta_opt.zero_grad()
            self._score.reset()
            query_loss_mean: MeanMetric = MeanMetric().to(self._device)

            task: Task
            inner_iterator = tqdm(task_set,
                                  leave=False,
                                  desc=emoji.emojize(f":repeat_one: Task set Training", language='alias'))
            for task in inner_iterator:
                with higher.innerloop_ctx(self._model, inner_optim, copy_initial_weights=False) as \
                        (f_model, diff_optim):
                    # Optimize the likelihood of the support set by taking
                    # gradient steps w.r.t. the model's parameters.
                    # This adapts the model's meta-parameters to the task.
                    # higher is able to automatically keep copies of
                    # your network's parameters as they are being updated.
                    support_batch: Dict[str, torch.Tensor] = next(iter(task.support))  # get one batch from dataloader
                    support_batch = self._move_batch_to_device(support_batch)
                    support_batch = self._transform(support_batch)
                    inputs: torch.Tensor  # (B,5,H,W) or (B,1,H,W) with float values
                    targets: torch.Tensor  # (B,H,W) with long values (0-4) or (0-1)
                    inputs, targets = self._model.preprocess(support_batch)
                    for _ in range(self._n_inner_iter):
                        outputs: torch.Tensor = f_model(inputs)
                        support_loss_value = self._loss(outputs, targets)
                        diff_optim.step(support_loss_value)

                    # The final set of adapted parameters will induce some
                    # final loss and accuracy on the query dataset.
                    # These will be used to update the model's meta-parameters.
                    query_batch: Dict[str, torch.Tensor] = next(iter(task.query))  # get one batch from dataloader
                    query_batch = self._move_batch_to_device(query_batch)
                    query_batch = self._transform(query_batch)
                    query_inputs: torch.Tensor
                    query_targets: torch.Tensor
                    query_inputs, query_targets = self._model.preprocess(query_batch)

                    query_outputs: torch.Tensor = f_model(query_inputs)
                    query_loss_value = self._loss(query_outputs, query_targets)
                    query_loss_mean.update(query_loss_value.detach())

                    with torch.no_grad():
                        activated_query_outputs: torch.Tensor = self._model.activate(query_outputs)
                        current_query_score: torch.Tensor = self._score(activated_query_outputs, query_targets).detach()

                    # Update the model's meta-parameters to optimize the query
                    # losses across all the tasks sampled in this batch.
                    # This unrolls through the gradient steps.
                    query_loss_value.backward()

                    iterator.set_postfix({
                        "support loss": support_loss_value.item(),
                        "query loss": query_loss_value.item(),
                        "query score": current_query_score.item(),
                    })

            self._meta_opt.step()
            iterator.set_postfix({
                "task set loss": query_loss_mean.compute().item(),
                "task set score": self._score.compute().item(),
                "lr": f"{self._lr_scheduler.get_last_lr()[-1]:.7f}"
            })
            logger.info(str({
                "task set loss": query_loss_mean.compute().item(),
                "task set score": self._score.compute().item(),
                "lr": f"{self._lr_scheduler.get_last_lr()[-1]:.7f}"
            }))

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
