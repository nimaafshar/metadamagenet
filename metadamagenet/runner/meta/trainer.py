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

from ...logging import EmojiAdapter
from ...models import BaseModel, Metadata, Checkpoint, ModelManager
from ...dataset import MetaDataLoader, TaskSet, Task
from ..base import Runner
from .validator import MetaValidator

logger = EmojiAdapter(logging.getLogger())


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

    def _prepare_batch(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._device is not None:
            data_batch = {k: v.to(device=self._device, non_blocking=True) for k, v in
                          data_batch.items()}
        with torch.no_grad():
            if self._transform is not None:
                data_batch = self._transform(data_batch)
            return self._model.preprocess(data_batch)

    def _train_epoch(self, epoch: int) -> None:
        self._model.train()

        total_support_loss: MeanMetric = MeanMetric().to(device=self._device)
        total_support_score: MeanMetric = MeanMetric().to(device=self._device)
        total_query_loss: MeanMetric = MeanMetric().to(device=self._device)
        total_query_score: MeanMetric = MeanMetric().to(device=self._device)

        with tqdm(total=self._meta_dataloader.total_tasks,
                  leave=False,
                  desc=emoji.emojize(f":repeat_one: MAML Training {epoch}/{self._epochs}", language='alias')) \
                as progress_bar:

            task_set: TaskSet
            for task_set in self._meta_dataloader:
                inner_optim: Optimizer = self._inner_opt(self._model)

                self._score.reset()
                query_loss_mean: MeanMetric = MeanMetric().to(self._device)

                task: Task
                for task in task_set:
                    with higher.innerloop_ctx(self._model, inner_optim, copy_initial_weights=False) as \
                            (f_model, diff_optim):
                        # Optimize the likelihood of the support set by taking
                        # gradient steps w.r.t. the model's parameters.
                        # This adapts the model's meta-parameters to the task.
                        # higher is able to automatically keep copies of
                        # your network's parameters as they are being updated.
                        self._score.reset()
                        for k in range(1, self._n_inner_iter + 1):
                            gc.collect()
                            torch.cuda.empty_cache()

                            support_loss_sum: torch.Tensor = 0
                            data_batch: Dict[str, torch.Tensor]
                            inputs: torch.Tensor  # (B,5,H,W) or (B,1,H,W) with float values
                            targets: torch.Tensor  # (B,H,W) with long values (0-4) or (0-1)
                            for data_batch in task.support:
                                inputs, targets = self._prepare_batch(data_batch)
                                outputs: torch.Tensor = f_model(inputs)
                                support_loss_sum += self._loss(outputs, targets)
                                with torch.no_grad():
                                    activated_outputs: torch.Tensor = self._model.activate(outputs)
                                    self._score.update(activated_outputs, targets)
                                del inputs, targets, outputs, activated_outputs
                            support_loss = support_loss_sum / len(task.support)
                            diff_optim.step(support_loss)
                            logger.info("%s", {
                                "task": task.name,
                                "mode": "adapt",
                                "k": f"{k}/{self._n_inner_iter}",
                                "loss": support_loss.item(),
                                "score": self._score.compute().item()
                            })
                            total_support_loss.update(support_loss.item())
                            total_support_score.update(self._score.compute().item())
                            progress_bar.set_postfix({
                                "sup_loss": total_support_loss.compute().item(),
                                "sup_sc": total_support_score.compute().item()
                            })
                        gc.collect()
                        torch.cuda.empty_cache()
                        # The final set of adapted parameters will induce some
                        # final loss and accuracy on the query dataset.
                        # These will be used to update the model's meta-parameters.
                        self._score.reset()
                        query_loss_sum: torch.Tensor = 0
                        for data_batch in task.query:
                            inputs, targets = self._prepare_batch(data_batch)
                            outputs = f_model(inputs).detach()
                            query_loss_sum += self._loss(outputs, targets).detach()
                            with torch.no_grad():
                                activated_outputs: torch.Tensor = self._model.activate(outputs)
                                self._score.update(activated_outputs, targets).detach()
                        query_loss: torch.Tensor = query_loss_sum / len(task.query)

                        # Update the model's meta-parameters to optimize the query
                        # losses across all the tasks sampled in this batch.
                        # This unrolls through the gradient steps.
                        query_loss.backward()

                        logger.info("%s", {
                            "task": task.name,
                            "mode": "check",
                            "loss": query_loss.item(),
                            "score": self._score.compute().item()
                        })

                        total_query_loss.update(query_loss.item())
                        total_query_score.update(self._score.compute().item())
                        progress_bar.set_postfix({
                            "que_loss": total_query_loss.compute().item(),
                            "que_sc": total_query_score.compute().item()
                        })

                    progress_bar.update(1)

                self._meta_opt.zero_grad()
                self._meta_opt.step()
                logger.info("%s", {
                    "support_loss": total_support_loss.compute().item(),
                    "support_score": total_support_score.compute().item(),
                    "query_loss": total_query_loss.compute().item(),
                    "query_score": total_query_score.compute().item(),
                    "lr": f"{self._lr_scheduler.get_last_lr()[-1]:.7f}"
                })

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
